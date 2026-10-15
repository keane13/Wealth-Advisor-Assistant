"""Domain model. Plain dataclasses keep the core dependency-free and easy to serialise.

Money is represented as ``float`` rounded to cents at the edges. The analysis is
statistical (medians, z-scores, ratios) rather than ledger arithmetic, so ``Decimal``
would add friction without improving any decision this system makes. See README trade-offs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ["low", "medium", "high", "critical"].index(self.value)


class FindingKind(str, Enum):
    ANOMALY = "anomaly"          # an unusual transaction or pattern
    RISK_SIGNAL = "risk_signal"  # a structural risk in the client's position
    TREND = "trend"              # a directional change over time


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class Transaction:
    id: str
    date: date
    amount: float          # positive = inflow, negative = outflow (base currency)
    description: str
    category: str
    merchant: str
    channel: str = "card"  # card | transfer | cash | ach | wire

    @property
    def is_outflow(self) -> bool:
        return self.amount < 0


@dataclass(frozen=True)
class Account:
    account_id: str
    type: str              # checking | savings | brokerage | retirement
    balance: float


@dataclass(frozen=True)
class Holding:
    symbol: str
    asset_class: str       # equity | bond | cash | real_estate | alternative
    market_value: float
    sector: str | None = None


@dataclass(frozen=True)
class Liability:
    type: str
    balance: float
    monthly_payment: float


@dataclass(frozen=True)
class ClientProfile:
    """Context from the CRM, not from the client's own financial data."""

    client_id: str
    risk_tolerance: RiskTolerance | None
    age: int | None = None
    investment_horizon_years: int | None = None
    goals: tuple[str, ...] = ()
    advisor: str | None = None
    last_review_date: str | None = None
    kyc_status: str | None = None
    notes: tuple[str, ...] = ()


@dataclass
class DataQuality:
    sources: dict[str, str] = field(default_factory=dict)  # source -> ok | degraded | unavailable
    warnings: list[str] = field(default_factory=list)
    rejected_records: int = 0

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class ClientSnapshot:
    client_id: str
    name: str
    as_of: date
    base_currency: str
    accounts: list[Account]
    holdings: list[Holding]
    liabilities: list[Liability]
    transactions: list[Transaction]
    profile: ClientProfile | None = None
    data_quality: DataQuality = field(default_factory=DataQuality)


@dataclass
class Finding:
    rule: str
    kind: FindingKind
    severity: Severity
    title: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    fingerprint_basis: tuple[str, ...] = ()
    client_id: str = ""
    suppressed: bool = False
    suppression_reason: str | None = None
    review: dict[str, Any] | None = None

    @property
    def fingerprint(self) -> str:
        """Stable identity across runs, so a reviewer's dismissal can be remembered."""
        basis = "|".join((self.client_id, self.rule, *sorted(self.fingerprint_basis)))
        return hashlib.sha1(basis.encode()).hexdigest()[:10]

    @property
    def id(self) -> str:
        return f"F-{self.fingerprint}"

    @property
    def needs_human_review(self) -> bool:
        return not self.suppressed and self.severity.rank >= Severity.HIGH.rank

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("fingerprint_basis")
        out["id"] = self.id
        out["kind"] = self.kind.value
        out["severity"] = self.severity.value
        return out


@dataclass
class AnalysisResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)
    history: dict[str, Any] = field(default_factory=dict)

    @property
    def active_findings(self) -> list[Finding]:
        return [f for f in self.findings if not f.suppressed]


@dataclass
class ReviewOutcome:
    status: str                         # not_required | completed | pending
    reviewer: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    pending_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    priority: int                      # 1 = act first
    title: str
    rationale: str
    actions: list[str]
    related_findings: list[str] = field(default_factory=list)
    status: str = "proposed"           # proposed | awaiting_validation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
