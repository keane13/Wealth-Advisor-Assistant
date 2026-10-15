"""Recommendation Agent: turns validated findings into prioritised, actionable advice.

Recommendations come from a declarative playbook (rule -> advice). That makes them
consistent, reviewable by compliance, and easy to extend, which an LLM generating free
text would not be. The LLM, when enabled, only writes the narrative summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from ..core.context import RunContext
from ..domain.models import AnalysisResult, ClientSnapshot, Finding, Recommendation, ReviewOutcome
from .analyzer import ANALYSIS_KEY
from .data_fetcher import SNAPSHOT_KEY
from .reviewer import REVIEW_KEY

RECOMMENDATIONS_KEY = "recommendations"
NARRATIVE_KEY = "narrative"

SEVERITY_TO_PRIORITY = {"critical": 1, "high": 2, "medium": 3, "low": 4}


@dataclass(frozen=True)
class Play:
    title: Callable[[list[Finding]], str]
    rationale: str
    actions: tuple[str, ...]


PLAYBOOK: dict[str, Play] = {
    "cash_structuring": Play(
        lambda fs: "Escalate cash-withdrawal pattern to Compliance",
        "Repeated withdrawals just below the reporting threshold are a recognised AML red flag.",
        ("Refer to the Compliance/AML officer under the firm's suspicious-activity procedure.",
         "Do not discuss the referral with the client (tipping-off risk).",
         "Hold discretionary recommendations until Compliance responds."),
    ),
    "large_transaction": Play(
        lambda fs: f"Verify {len(fs)} large outflow(s) with the client",
        "Outflows that are large relative to income and not recurring can indicate fraud, a life event "
        "or a liquidity need the plan does not reflect.",
        ("Confirm purpose and authorisation of each payment with the client.",
         "If unauthorised, start the fraud/recall process with the bank immediately.",
         "If planned (e.g. property, tuition), update the cash-flow plan and goals."),
    ),
    "duplicate_transaction": Play(
        lambda fs: "Dispute probable duplicate charge at " + ", ".join(sorted({str(f.evidence.get("merchant"))
                                                                              for f in fs})),
        "Identical charges at the same merchant within days are usually billing errors.",
        ("Ask the client to confirm whether both purchases were intended.",
         "If not, raise a dispute with the card issuer/merchant for a refund."),
    ),
    "statistical_outlier": Play(
        lambda fs: "Review unusual transactions in " + ", ".join(sorted({str(f.evidence.get("category")) for f in fs})),
        "These amounts are far outside the client's normal pattern for the category.",
        ("Confirm the transactions are genuine.", "Check for new recurring costs that should enter the budget."),
    ),
    "spending_spike": Play(
        lambda fs: "Discuss spending increase in " + ", ".join(sorted(str(f.evidence.get("category")) for f in fs)),
        "A sharp month-on-month rise erodes savings capacity; some categories also signal wellbeing risk.",
        ("Discuss the increase and whether it is temporary.",
         "For gambling or high-cost credit, offer spending limits/support resources per firm policy."),
    ),
    "negative_cashflow": Play(
        lambda fs: "Close the gap between spending and income",
        "Sustained negative cash flow draws down savings and investments.",
        ("Build a budget with the client and identify discretionary cuts.",
         "Avoid selling long-term investments to fund recurring spend."),
    ),
    "declining_cashflow": Play(
        lambda fs: "Investigate declining net cash flow",
        "Net cash flow has fallen for consecutive months.",
        ("Identify the driver (lower income vs higher spend).", "Revisit savings targets if the trend persists."),
    ),
    "concentration": Play(
        lambda fs: "Reduce concentration in " + ", ".join(str(f.evidence.get("symbol")) for f in fs),
        "A single position above the concentration limit exposes the portfolio to idiosyncratic risk.",
        ("Propose a staged diversification plan (e.g. over 2-4 quarters).",
         "Assess capital-gains tax impact before selling; consider hedging or exchange funds."),
    ),
    "suitability_mismatch": Play(
        lambda fs: f"Realign portfolio with the client's {fs[0].evidence.get('risk_tolerance')} risk profile",
        "Allocation outside the agreed risk band is a suitability issue.",
        ("Re-confirm risk tolerance and horizon with the client (refresh questionnaire).",
         "Rebalance towards the target equity band if the profile is confirmed."),
    ),
    "low_liquidity": Play(
        lambda fs: "Build the emergency fund to at least 3 months of spending",
        "Thin liquidity forces asset sales at bad times when an unexpected cost arrives.",
        ("Redirect monthly surplus to a high-yield savings account.",
         "Keep near-term needs out of volatile assets."),
    ),
    "high_debt_to_income": Play(
        lambda fs: "Reduce debt-service burden",
        "High debt payments relative to income limit resilience and saving capacity.",
        ("Review refinancing/consolidation options.", "Prioritise paying down highest-rate debt."),
    ),
    "kyc_not_current": Play(
        lambda fs: "Bring KYC up to date before acting",
        "Recommendations cannot be executed for a client whose due diligence is not current.",
        ("Request updated KYC documents.", "Hold new transactions until KYC is refreshed."),
    ),
    "review_overdue": Play(
        lambda fs: "Schedule the overdue annual review",
        "Annual reviews keep goals, risk profile and plan in sync.",
        ("Book a review meeting.",),
    ),
}


class RecommendationAgent(BaseAgent):
    name = "recommender"
    capabilities = frozenset({"recommend"})

    def _run(self, task: AgentTask, ctx: RunContext) -> AgentResult:
        analysis: AnalysisResult = ctx.session.require(ANALYSIS_KEY)
        snapshot: ClientSnapshot = ctx.session.require(SNAPSHOT_KEY)
        review: ReviewOutcome | None = ctx.session.get(REVIEW_KEY)
        pending = set(review.pending_ids) if review else set()
        warnings: list[str] = []

        recs = self._build(analysis.active_findings, pending, warnings)
        ctx.session.put(RECOMMENDATIONS_KEY, recs, written_by=self.name)

        facts = build_facts(snapshot, analysis, recs, pending)
        narrative = ctx.tools.call("report.narrative", facts=facts)
        if narrative.ok:
            payload = dict(narrative.data)
            if narrative.degraded:
                payload["note"] = f"LLM narrative unavailable ({narrative.error}); template used"
                warnings.append(payload["note"])
                self.decision("narrative fell back to deterministic template", reason=narrative.error)
            ctx.session.put(NARRATIVE_KEY, payload, written_by=self.name)
        else:
            warnings.append(f"narrative unavailable: {narrative.error}")

        status = AgentStatus.SUCCESS if narrative.ok and not narrative.degraded else AgentStatus.PARTIAL
        provisional = sum(1 for r in recs if r.status == "awaiting_validation")
        return self.result(task, status, f"{len(recs)} recommendation(s), {provisional} awaiting validation",
                           warnings=warnings)

    def _build(self, findings: list[Finding], pending: set[str], warnings: list[str]) -> list[Recommendation]:
        grouped: dict[str, list[Finding]] = {}
        for f in findings:
            grouped.setdefault(f.rule, []).append(f)

        recs: list[Recommendation] = []
        for rule, group in grouped.items():
            play = PLAYBOOK.get(rule)
            if play is None:  # a new detector without a playbook entry still surfaces
                warnings.append(f"no playbook entry for rule '{rule}'; generic recommendation used")
                play = Play(lambda fs: f"Review: {fs[0].title}", "Flagged by analysis.", ("Review with the client.",))
            worst = max(group, key=lambda f: f.severity.rank)
            recs.append(Recommendation(
                priority=SEVERITY_TO_PRIORITY[worst.severity.value],
                title=play.title(group),
                rationale=play.rationale,
                actions=list(play.actions),
                related_findings=[f.id for f in group],
                status="awaiting_validation" if any(f.id in pending for f in group) else "proposed",
            ))

        if not recs:
            recs.append(Recommendation(
                priority=4, title="No action required; keep the current plan",
                rationale="No active findings in this review.", actions=["Schedule the next routine review."],
            ))
        recs.sort(key=lambda r: (r.priority, r.title))
        return recs


def build_facts(snapshot: ClientSnapshot, analysis: AnalysisResult, recs: list[Recommendation],
                pending: set[str]) -> dict[str, Any]:
    """The minimal, de-identified fact sheet given to the narrative tool (no name, no raw transactions)."""
    cf, risk = analysis.metrics.get("cashflow", {}), analysis.metrics.get("risk", {})
    key_metrics = {
        "net_worth": risk.get("net_worth"),
        "avg_monthly_income": cf.get("avg_monthly_income"),
        "avg_monthly_spend": cf.get("avg_monthly_spend"),
        "savings_rate_pct": cf.get("savings_rate_pct"),
        "equity_pct": risk.get("equity_pct"),
        "liquidity_months": risk.get("liquidity_months"),
        "debt_to_income_pct": risk.get("debt_to_income_pct"),
    }
    return {
        "client_id": snapshot.client_id,
        "as_of": snapshot.as_of.isoformat(),
        "currency": snapshot.base_currency,
        "risk_tolerance": snapshot.profile.risk_tolerance.value
        if snapshot.profile and snapshot.profile.risk_tolerance else None,
        "key_metrics": {k: v for k, v in key_metrics.items() if v is not None},
        "findings": [{"id": f.id, "severity": f.severity.value, "title": f.title, "detail": f.detail,
                      "review_status": "pending" if f.id in pending else ("validated" if f.review else "n/a")}
                     for f in analysis.active_findings],
        "recommendations": [{"priority": r.priority, "title": r.title, "status": r.status} for r in recs],
        "data_quality_issues": len(snapshot.data_quality.warnings),
        "since_last_review": {k: v for k, v in analysis.history.items() if k != "previous_run"}
        if analysis.history.get("previous_run") else None,
    }
