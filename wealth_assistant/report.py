"""Assemble the structured advisory report from session memory and the execution trace.

The report is the system's contract with downstream consumers (an advisor UI, a CRM
note, an audit store), so it has an explicit ``report_version``. Every piece of it
says where it came from: data sources, which checks were skipped, who reviewed what,
and whether the narrative came from the LLM or the template.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .core.agent import AgentResult
from .core.context import RunContext
from .domain.models import AnalysisResult, ClientSnapshot, Recommendation, ReviewOutcome

REPORT_VERSION = "1.0"


def build_report(ctx: RunContext, plan: str, status: str, trace: list[AgentResult],
                 errors: list[str] | None = None) -> dict[str, Any]:
    snapshot: ClientSnapshot | None = ctx.session.get("snapshot")
    analysis: AnalysisResult | None = ctx.session.get("analysis")
    review: ReviewOutcome | None = ctx.session.get("review")
    recs: list[Recommendation] = ctx.session.get("recommendations") or []
    narrative: dict[str, Any] | None = ctx.session.get("narrative")

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "run_id": ctx.run_id,
        "client_id": ctx.client_id,
        "plan": plan,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if snapshot:
        profile = snapshot.profile
        report["client"] = {
            "name": snapshot.name,
            "as_of": snapshot.as_of.isoformat(),
            "base_currency": snapshot.base_currency,
            "risk_tolerance": profile.risk_tolerance.value if profile and profile.risk_tolerance else None,
            "age": profile.age if profile else None,
            "goals": list(profile.goals) if profile else [],
            "advisor": profile.advisor if profile else None,
            "crm_notes": list(profile.notes) if profile else [],
        }
        report["data_quality"] = {
            "sources": snapshot.data_quality.sources,
            "rejected_records": snapshot.data_quality.rejected_records,
            "warnings": snapshot.data_quality.warnings,
            "transactions_analysed": len(snapshot.transactions),
        }

    if analysis:
        report["metrics"] = analysis.metrics
        report["findings"] = [f.to_dict() for f in analysis.active_findings]
        report["suppressed_findings"] = [f.to_dict() for f in analysis.findings if f.suppressed]
        report["skipped_checks"] = analysis.skipped_checks
        report["history"] = analysis.history

    if review:
        report["human_review"] = {
            "status": review.status,
            "reviewer": review.reviewer,
            "decisions": review.decisions,
            "pending_finding_ids": review.pending_ids,
            "issues": review.issues,
        }
    if recs:
        report["recommendations"] = [r.to_dict() for r in recs]
    if narrative:
        report["summary"] = narrative

    report["execution_trace"] = [r.to_dict() for r in trace]
    report["errors"] = list(errors or [])
    return report


def run_summary(ctx: RunContext, report: dict[str, Any]) -> dict[str, Any]:
    """The compact record written to long-term memory after each run."""
    analysis: AnalysisResult | None = ctx.session.get("analysis")
    findings = analysis.active_findings if analysis else []
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    return {
        "run_id": ctx.run_id,
        "at": report["generated_at"],
        "plan": report["plan"],
        "status": report["status"],
        "as_of": report.get("client", {}).get("as_of"),
        "net_worth": report.get("metrics", {}).get("risk", {}).get("net_worth"),
        "finding_counts": counts,
        "active_fingerprints": [f.fingerprint for f in findings],
        "top_findings": [f.title for f in findings[:5]],
        "top_recommendations": [r["title"] for r in report.get("recommendations", [])[:3]],
    }
