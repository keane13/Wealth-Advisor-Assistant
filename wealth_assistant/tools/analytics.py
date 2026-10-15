"""Financial analysis tools: cash flow and structural risk signals.

Every tool returns ``{"metrics": ..., "findings": [...], "skipped": [...]}``. ``skipped``
lists checks that could not run and why. A check that was not run is never reported
as a pass.
"""

from __future__ import annotations

import calendar
import statistics
from collections import defaultdict
from datetime import date
from typing import Any

from ..config import RiskThresholds
from ..core.tools import Tool
from ..domain.models import ClientSnapshot, Finding, FindingKind, Severity, Transaction

EXCLUDED_FROM_CASHFLOW = {"internal_transfer"}  # moving money between own accounts is not spend
DIVERSIFIED_SECTORS = {"diversified", "broad market"}
DEPOSIT_ACCOUNT_TYPES = {"checking", "savings", "money_market"}


def month_key(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def monthly_flows(transactions: list[Transaction]) -> dict[str, dict[str, float]]:
    months: dict[str, dict[str, float]] = defaultdict(lambda: {"inflow": 0.0, "outflow": 0.0})
    for tx in transactions:
        if tx.category in EXCLUDED_FROM_CASHFLOW:
            continue
        bucket = months[month_key(tx.date)]
        if tx.amount > 0:
            bucket["inflow"] += tx.amount
        else:
            bucket["outflow"] += -tx.amount
    return dict(sorted(months.items()))


def complete_months(transactions: list[Transaction], as_of: date) -> set[str]:
    """Months fully covered by the data. Partial edge months would distort averages."""
    if not transactions:
        return set()
    first = min(t.date for t in transactions)
    months = {month_key(t.date) for t in transactions}
    incomplete = set()
    if first.day > 5:
        incomplete.add(month_key(first))
    if as_of.day < calendar.monthrange(as_of.year, as_of.month)[1]:
        incomplete.add(month_key(as_of))
    return months - incomplete


def _slope(values: list[float]) -> float:
    n = len(values)
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    return sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denom if denom else 0.0


class CashflowAnalysisTool(Tool):
    name = "analysis.cashflow"
    description = "Monthly inflow/outflow, savings rate and net cash-flow trend."

    def run(self, *, snapshot: ClientSnapshot) -> dict[str, Any]:
        findings: list[Finding] = []
        skipped: list[str] = []
        flows = monthly_flows(snapshot.transactions)
        if not flows:
            return {"metrics": {"months": []}, "findings": [], "skipped": ["cash flow: no usable transactions"]}

        full = complete_months(snapshot.transactions, snapshot.as_of)
        basis_months = [m for m in flows if m in full] or list(flows)
        if len(basis_months) < len(flows):
            skipped.append(f"cash flow averages exclude partial month(s): "
                           f"{', '.join(m for m in flows if m not in basis_months)}")

        incomes = [flows[m]["inflow"] for m in basis_months]
        spends = [flows[m]["outflow"] for m in basis_months]
        nets = [flows[m]["inflow"] - flows[m]["outflow"] for m in basis_months]
        avg_income, avg_spend = statistics.fmean(incomes), statistics.fmean(spends)
        savings_rate = (avg_income - avg_spend) / avg_income * 100 if avg_income > 0 else None
        trend = _slope(nets) if len(nets) >= 3 else None

        metrics = {
            "months": [{"month": m, "inflow": round(v["inflow"], 2), "outflow": round(v["outflow"], 2),
                        "net": round(v["inflow"] - v["outflow"], 2)} for m, v in flows.items()],
            "months_in_averages": len(basis_months),
            "avg_monthly_income": round(avg_income, 2),
            "median_monthly_income": round(statistics.median(incomes), 2),
            "avg_monthly_spend": round(avg_spend, 2),
            "savings_rate_pct": round(savings_rate, 1) if savings_rate is not None else None,
            "net_trend_per_month": round(trend, 2) if trend is not None else None,
        }
        window = (basis_months[0], basis_months[-1])

        if avg_income <= 0 and avg_spend > 0:
            findings.append(Finding(
                rule="negative_cashflow", kind=FindingKind.RISK_SIGNAL, severity=Severity.HIGH,
                title="No income recorded while spending continues",
                detail=f"Average monthly spend {avg_spend:,.2f} with no recorded income over {len(basis_months)} month(s).",
                evidence={"avg_monthly_spend": round(avg_spend, 2)}, fingerprint_basis=window,
            ))
        elif savings_rate is not None and savings_rate < 0:
            findings.append(Finding(
                rule="negative_cashflow", kind=FindingKind.RISK_SIGNAL,
                severity=Severity.HIGH if savings_rate < -20 else Severity.MEDIUM,
                title="Spending exceeds income",
                detail=f"Average monthly spend {avg_spend:,.2f} vs income {avg_income:,.2f} "
                       f"(savings rate {savings_rate:.1f}%).",
                evidence={"savings_rate_pct": round(savings_rate, 1), "months": len(basis_months)},
                fingerprint_basis=window,
            ))

        if trend is None:
            skipped.append("cash-flow trend: needs at least 3 complete months")
        elif (len(nets) >= 4 and trend < 0 and nets[-1] < nets[-2] < nets[-3]
              and abs(trend) * len(nets) > 0.1 * max(avg_income, 1.0)):
            findings.append(Finding(
                rule="declining_cashflow", kind=FindingKind.TREND, severity=Severity.MEDIUM,
                title="Net cash flow is declining",
                detail=f"Net cash flow fell for three consecutive months; trend {trend:,.2f} per month.",
                evidence={"net_by_month": dict(zip(basis_months, [round(n, 2) for n in nets]))},
                fingerprint_basis=window,
            ))
        return {"metrics": metrics, "findings": findings, "skipped": skipped}


class RiskSignalsTool(Tool):
    name = "analysis.risk_signals"
    description = "Portfolio concentration, suitability vs risk profile, liquidity, leverage and KYC signals."

    def __init__(self, thresholds: RiskThresholds) -> None:
        self.t = thresholds

    def run(self, *, snapshot: ClientSnapshot, cashflow: dict[str, Any] | None) -> dict[str, Any]:
        findings: list[Finding] = []
        skipped: list[str] = []
        profile = snapshot.profile

        deposits = sum(a.balance for a in snapshot.accounts if a.type in DEPOSIT_ACCOUNT_TYPES)
        invested = sum(h.market_value for h in snapshot.holdings)
        debt = sum(liab.balance for liab in snapshot.liabilities)
        metrics: dict[str, Any] = {
            "cash_deposits": round(deposits, 2),
            "investments_total": round(invested, 2),
            "liabilities_total": round(debt, 2),
            "net_worth": round(deposits + invested - debt, 2),
        }

        # --- portfolio composition -------------------------------------------------
        if invested <= 0:
            skipped.append("portfolio checks: no holdings")
        else:
            by_class: dict[str, float] = defaultdict(float)
            for h in snapshot.holdings:
                by_class[h.asset_class] += h.market_value
            allocation = {k: round(v / invested * 100, 1) for k, v in sorted(by_class.items())}
            equity_pct = allocation.get("equity", 0.0)
            metrics["allocation_pct"] = allocation
            metrics["equity_pct"] = equity_pct
            top = max(snapshot.holdings, key=lambda h: h.market_value)
            metrics["largest_position"] = {"symbol": top.symbol, "pct": round(top.market_value / invested * 100, 1)}

            for h in snapshot.holdings:
                pct = h.market_value / invested * 100
                if (pct > self.t.max_single_position_pct and h.asset_class != "cash"
                        and (h.sector or "").lower() not in DIVERSIFIED_SECTORS):
                    findings.append(Finding(
                        rule="concentration", kind=FindingKind.RISK_SIGNAL,
                        severity=Severity.HIGH if pct > 1.75 * self.t.max_single_position_pct else Severity.MEDIUM,
                        title=f"Concentrated position in {h.symbol}",
                        detail=f"{h.symbol} is {pct:.1f}% of investments (limit {self.t.max_single_position_pct:.0f}%).",
                        evidence={"symbol": h.symbol, "pct": round(pct, 1), "market_value": h.market_value},
                        fingerprint_basis=(h.symbol,),
                    ))

            risk = profile.risk_tolerance if profile else None
            if risk is None:
                skipped.append("suitability check: risk tolerance unknown (no usable CRM profile)")
            else:
                low, high = self.t.equity_bands[risk.value]
                if not low <= equity_pct <= high:
                    gap = equity_pct - high if equity_pct > high else low - equity_pct
                    direction = "above" if equity_pct > high else "below"
                    findings.append(Finding(
                        rule="suitability_mismatch", kind=FindingKind.RISK_SIGNAL,
                        severity=Severity.HIGH if gap > 20 else Severity.MEDIUM,
                        title=f"Equity allocation {direction} {risk.value} risk band",
                        detail=f"Equity is {equity_pct:.1f}% vs target band {low:.0f}-{high:.0f}% "
                               f"for a {risk.value} investor.",
                        evidence={"equity_pct": equity_pct, "band": [low, high], "risk_tolerance": risk.value},
                        fingerprint_basis=(risk.value, direction),
                    ))

        # --- liquidity and leverage (need cash-flow context) ---------------------------
        cf = cashflow or {}
        avg_spend, avg_income = cf.get("avg_monthly_spend"), cf.get("avg_monthly_income")
        cash_holdings = sum(h.market_value for h in snapshot.holdings if h.asset_class == "cash")
        if not avg_spend:
            skipped.append("liquidity check: average monthly spend unknown")
        else:
            months = (deposits + cash_holdings) / avg_spend
            metrics["liquidity_months"] = round(months, 1)
            if months < self.t.min_liquidity_months:
                findings.append(Finding(
                    rule="low_liquidity", kind=FindingKind.RISK_SIGNAL,
                    severity=Severity.HIGH if months < self.t.min_liquidity_months / 2 else Severity.MEDIUM,
                    title="Emergency fund below target",
                    detail=f"Liquid assets cover {months:.1f} months of spending "
                           f"(target {self.t.min_liquidity_months:.0f}).",
                    evidence={"liquidity_months": round(months, 1), "liquid_assets": round(deposits + cash_holdings, 2)},
                ))

        payments = sum(liab.monthly_payment for liab in snapshot.liabilities)
        if payments and not avg_income:
            skipped.append("debt-to-income check: income unknown")
        elif payments:
            dti = payments / avg_income * 100
            metrics["debt_to_income_pct"] = round(dti, 1)
            if dti > self.t.max_debt_to_income_pct:
                findings.append(Finding(
                    rule="high_debt_to_income", kind=FindingKind.RISK_SIGNAL,
                    severity=Severity.HIGH if dti > 50 else Severity.MEDIUM,
                    title="Debt service is high relative to income",
                    detail=f"Debt payments are {dti:.1f}% of income (guideline {self.t.max_debt_to_income_pct:.0f}%).",
                    evidence={"debt_to_income_pct": round(dti, 1), "monthly_payments": round(payments, 2)},
                ))

        # --- profile / compliance hygiene --------------------------------------------
        if profile is None:
            skipped.append("KYC / review-date checks: no CRM profile")
        else:
            kyc = (profile.kyc_status or "").lower()
            if kyc in {"expired", "pending"}:
                findings.append(Finding(
                    rule="kyc_not_current", kind=FindingKind.RISK_SIGNAL,
                    severity=Severity.HIGH if kyc == "expired" else Severity.MEDIUM,
                    title=f"KYC status is {kyc}",
                    detail="Client due-diligence must be current before acting on new recommendations.",
                    evidence={"kyc_status": kyc}, fingerprint_basis=(kyc,),
                ))
            if profile.last_review_date:
                try:
                    days = (snapshot.as_of - date.fromisoformat(profile.last_review_date)).days
                    if days > 365:
                        findings.append(Finding(
                            rule="review_overdue", kind=FindingKind.RISK_SIGNAL, severity=Severity.LOW,
                            title="Annual review overdue",
                            detail=f"Last advisor review was {days} days ago.",
                            evidence={"last_review_date": profile.last_review_date, "days": days},
                            fingerprint_basis=(profile.last_review_date,),
                        ))
                except ValueError:
                    skipped.append(f"review-date check: unparseable date {profile.last_review_date!r}")

        return {"metrics": metrics, "findings": findings, "skipped": skipped}
