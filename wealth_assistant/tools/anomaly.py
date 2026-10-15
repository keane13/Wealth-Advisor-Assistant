"""Transaction anomaly detection.

The detection is rule-based plus robust statistics rather than an ML model. With a few
months of history per client there is nothing to train on, and every flag has to be
explainable to an advisor (and to a compliance officer). Each detector is a small,
independently testable strategy. New ones plug in via the ``detectors`` list.

Detectors run in order and the order matters. Single-transaction detectors skip
transactions that an earlier detector already flagged, so one payment is never
reported three times under three different names.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..config import AnomalyThresholds
from ..core.tools import Tool
from ..domain.models import ClientSnapshot, Finding, FindingKind, Severity, Transaction
from ..logging_setup import get_logger
from .analytics import EXCLUDED_FROM_CASHFLOW, month_key

logger = get_logger("tools.anomaly")

SENSITIVE_CATEGORIES = {"gambling", "cash_advance", "payday_loan"}


@dataclass
class DetectionContext:
    snapshot: ClientSnapshot
    median_monthly_income: float | None
    thresholds: AnomalyThresholds
    flagged_tx_ids: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)

    @property
    def outflows(self) -> list[Transaction]:
        return [t for t in self.snapshot.transactions
                if t.is_outflow and t.category not in EXCLUDED_FROM_CASHFLOW]


class Detector(ABC):
    rule: str

    @abstractmethod
    def detect(self, ctx: DetectionContext) -> list[Finding]: ...


class CashStructuringDetector(Detector):
    """Several cash withdrawals just under the reporting threshold in a short window."""

    rule = "cash_structuring"

    def detect(self, ctx: DetectionContext) -> list[Finding]:
        t = ctx.thresholds
        floor = t.cash_reporting_threshold * t.cash_structuring_band
        candidates = sorted((tx for tx in ctx.outflows
                             if tx.channel == "cash" and floor <= -tx.amount < t.cash_reporting_threshold),
                            key=lambda tx: tx.date)
        findings: list[Finding] = []
        used: set[str] = set()
        for i, first in enumerate(candidates):
            if first.id in used:
                continue
            window = [tx for tx in candidates[i:]
                      if tx.date - first.date <= timedelta(days=t.cash_structuring_window_days) and tx.id not in used]
            if len(window) >= t.cash_structuring_count:
                ids = [tx.id for tx in window]
                used.update(ids)
                total = -sum(tx.amount for tx in window)
                findings.append(Finding(
                    rule=self.rule, kind=FindingKind.ANOMALY, severity=Severity.CRITICAL,
                    title="Possible structuring of cash withdrawals",
                    detail=f"{len(window)} cash withdrawals totalling {total:,.2f} between {window[0].date} and "
                           f"{window[-1].date}, each just under the {t.cash_reporting_threshold:,.0f} reporting threshold.",
                    evidence={"transaction_ids": ids, "total": round(total, 2),
                              "amounts": [-tx.amount for tx in window]},
                    fingerprint_basis=tuple(ids),
                ))
        ctx.flagged_tx_ids.update(used)
        return findings


class LargeTransactionDetector(Detector):
    """Outflows that are large relative to the client's own income, unless they recur."""

    rule = "large_transaction"

    def detect(self, ctx: DetectionContext) -> list[Finding]:
        t = ctx.thresholds
        income = ctx.median_monthly_income
        if income and income > 0:
            threshold = max(t.large_tx_min_amount, t.large_tx_income_ratio * income)
            basis = f"{t.large_tx_income_ratio:.0%} of median monthly income"
        else:
            threshold = t.large_tx_min_amount * 2.5
            ctx.skipped.append(f"large-transaction rule: income unknown, using absolute threshold {threshold:,.0f}")
            basis = "absolute threshold (income unknown)"

        outflows = ctx.outflows
        by_merchant: dict[str, list[Transaction]] = defaultdict(list)
        for tx in outflows:
            by_merchant[tx.merchant.lower()].append(tx)

        findings = []
        for tx in outflows:
            amount = -tx.amount
            if amount < threshold or tx.id in ctx.flagged_tx_ids:
                continue
            others = [-o.amount for o in by_merchant[tx.merchant.lower()] if o.id != tx.id]
            # A payee paid 3+ times in the window is a recurring obligation such as rent
            # or medical bills. A payment to it is only a surprise if it is far above the
            # usual level. A gradual rise in costs is caught by the cash-flow trend rules.
            # The whole window counts because this is a retrospective review.
            if len(others) >= 2 and amount <= 2 * statistics.median(others):
                continue
            first_time = not any(o.date < tx.date for o in by_merchant[tx.merchant.lower()] if o.id != tx.id)
            severity = Severity.HIGH if (income and amount >= income) or first_time else Severity.MEDIUM
            findings.append(Finding(
                rule=self.rule, kind=FindingKind.ANOMALY, severity=severity,
                title=f"Large outflow to {tx.merchant}",
                detail=f"{amount:,.2f} on {tx.date} via {tx.channel} exceeds {threshold:,.2f} ({basis})"
                       f"{'; first payment to this payee' if first_time else ''}.",
                evidence={"transaction_ids": [tx.id], "amount": amount, "threshold": round(threshold, 2),
                          "first_time_payee": first_time, "category": tx.category},
                fingerprint_basis=(tx.id,),
            ))
            ctx.flagged_tx_ids.add(tx.id)
        return findings


class DuplicateChargeDetector(Detector):
    """Same merchant and same amount within a few days. Cash is excluded (repeat ATM amounts are normal)."""

    rule = "duplicate_transaction"

    def detect(self, ctx: DetectionContext) -> list[Finding]:
        window = timedelta(days=ctx.thresholds.duplicate_window_days)
        groups: dict[tuple[str, float], list[Transaction]] = defaultdict(list)
        for tx in ctx.outflows:
            if tx.channel != "cash":
                groups[(tx.merchant.lower(), tx.amount)].append(tx)

        findings = []
        for (_, amount), txs in groups.items():
            txs.sort(key=lambda tx: tx.date)
            for a, b in zip(txs, txs[1:]):
                if b.date - a.date <= window:
                    findings.append(Finding(
                        rule=self.rule, kind=FindingKind.ANOMALY, severity=Severity.MEDIUM,
                        title=f"Possible duplicate charge at {a.merchant}",
                        detail=f"Two charges of {-amount:,.2f} on {a.date} and {b.date}.",
                        evidence={"transaction_ids": [a.id, b.id], "amount": -amount, "merchant": a.merchant},
                        fingerprint_basis=(a.id, b.id),
                    ))
                    ctx.flagged_tx_ids.update({a.id, b.id})
        return findings


class CategoryOutlierDetector(Detector):
    """Per-category modified z-score (median/MAD). Robust to the outliers it is looking for."""

    rule = "statistical_outlier"

    def detect(self, ctx: DetectionContext) -> list[Finding]:
        t = ctx.thresholds
        by_category: dict[str, list[Transaction]] = defaultdict(list)
        for tx in ctx.outflows:
            by_category[tx.category].append(tx)

        findings: list[Finding] = []
        thin = []
        for category, txs in sorted(by_category.items()):
            if len(txs) < t.robust_z_min_samples:
                thin.append(category)
                continue
            amounts = [-tx.amount for tx in txs]
            median = statistics.median(amounts)
            mad = statistics.median(abs(a - median) for a in amounts)
            # MAD is 0 when most values are identical. Fall back to the mean absolute
            # deviation (Iglewicz & Hoaglin's recommendation).
            scale = 0.6745 / mad if mad else (1 / (1.2533 * statistics.fmean(abs(a - median) for a in amounts))
                                              if any(a != median for a in amounts) else 0.0)
            if scale == 0.0:
                continue
            for tx, amount in zip(txs, amounts):
                z = scale * (amount - median)
                if z > t.robust_z_threshold and tx.id not in ctx.flagged_tx_ids:
                    findings.append(Finding(
                        rule=self.rule, kind=FindingKind.ANOMALY,
                        severity=Severity.MEDIUM if amount >= t.spike_min_amount else Severity.LOW,
                        title=f"Unusual {category} transaction",
                        detail=f"{amount:,.2f} at {tx.merchant} on {tx.date} vs typical {median:,.2f} "
                               f"(modified z-score {z:.1f}).",
                        evidence={"transaction_ids": [tx.id], "amount": amount, "category": category,
                                  "category_median": round(median, 2), "modified_z": round(z, 1)},
                        fingerprint_basis=(tx.id,),
                    ))
                    ctx.flagged_tx_ids.add(tx.id)
        if thin:
            ctx.skipped.append(f"statistical outliers: too few samples (<{t.robust_z_min_samples}) in "
                               f"{len(thin)} categor{'y' if len(thin) == 1 else 'ies'}: {', '.join(thin)}")
        return findings


class CategorySpikeDetector(Detector):
    """Latest month's spend in a category compared with the trailing monthly average."""

    rule = "spending_spike"

    def detect(self, ctx: DetectionContext) -> list[Finding]:
        t = ctx.thresholds
        spend: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for tx in ctx.outflows:
            # Transactions already explained by an earlier detector (e.g. a duplicate
            # charge) are left out, so one event cannot also show up as a spike.
            if tx.id not in ctx.flagged_tx_ids:
                spend[month_key(tx.date)][tx.category] += -tx.amount
        months = sorted(spend)
        if len(months) < 3:
            ctx.skipped.append("spending spikes: need at least 3 months of history")
            return []

        latest, history = months[-1], months[:-1]
        findings = []
        for category, amount in sorted(spend[latest].items()):
            trailing = statistics.fmean(spend[m].get(category, 0.0) for m in history)
            if amount - trailing < t.spike_min_amount:
                continue
            if trailing > 0 and amount < t.spike_ratio * trailing:
                continue
            sensitive = category in SENSITIVE_CATEGORIES
            # The spike explains that month's transactions in the category, so they are
            # claimed here and not re-reported one by one as statistical outliers.
            tx_ids = [tx.id for tx in ctx.outflows if tx.category == category
                      and month_key(tx.date) == latest and tx.id not in ctx.flagged_tx_ids]
            ctx.flagged_tx_ids.update(tx_ids)
            findings.append(Finding(
                rule=self.rule, kind=FindingKind.TREND,
                severity=Severity.HIGH if sensitive else Severity.MEDIUM,
                title=f"Spending spike in {category}",
                detail=f"{amount:,.2f} in {latest} vs trailing average {trailing:,.2f} "
                       f"({'new category' if trailing == 0 else f'{amount / trailing:.1f}x'}).",
                evidence={"month": latest, "amount": round(amount, 2), "trailing_avg": round(trailing, 2),
                          "category": category, "sensitive_category": sensitive, "transaction_ids": tx_ids},
                fingerprint_basis=(latest, category),
            ))
        return findings


DEFAULT_DETECTORS: tuple[type[Detector], ...] = (
    CashStructuringDetector,   # most specific pattern first: it claims its transactions
    LargeTransactionDetector,
    DuplicateChargeDetector,
    CategorySpikeDetector,     # claims the spiking month's transactions in that category
    CategoryOutlierDetector,   # least specific: whatever is still unexplained
)


class AnomalyDetectionTool(Tool):
    name = "analysis.detect_anomalies"
    description = "Detect unusual transactions and patterns (large, duplicate, outlier, spike, cash structuring)."

    def __init__(self, thresholds: AnomalyThresholds, detectors: list[Detector] | None = None) -> None:
        self.thresholds = thresholds
        self.detectors = detectors if detectors is not None else [d() for d in DEFAULT_DETECTORS]

    def run(self, *, snapshot: ClientSnapshot, median_monthly_income: float | None = None) -> dict[str, Any]:
        ctx = DetectionContext(snapshot, median_monthly_income, self.thresholds)
        if not ctx.outflows:
            return {"findings": [], "skipped": ["anomaly detection: no outgoing transactions"], "detectors_run": []}
        findings: list[Finding] = []
        ran: list[str] = []
        for detector in self.detectors:
            try:
                findings.extend(detector.detect(ctx))
                ran.append(detector.rule)
            except Exception as exc:  # one faulty detector must not hide the others' results
                logger.exception("detector %s failed", detector.rule, extra={"event": "detector.crash"})
                ctx.skipped.append(f"{detector.rule}: detector failed ({type(exc).__name__}: {exc})")
        return {"findings": findings, "skipped": ctx.skipped, "detectors_run": ran}
