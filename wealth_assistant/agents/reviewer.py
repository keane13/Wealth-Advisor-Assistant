"""Human Review Agent: the human-in-the-loop checkpoint.

It does not judge anything itself. It brings the human's judgement into the pipeline:
it selects the findings that need sign-off, collects decisions from a pluggable
``Reviewer``, validates them, applies them, and writes "remember" dismissals to
long-term memory.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from ..core.context import RunContext
from ..domain.models import AnalysisResult, ReviewOutcome, Severity
from ..hitl.review import Reviewer, ReviewUnavailable
from .analyzer import ANALYSIS_KEY, findings_by_id

REVIEW_KEY = "review"


class HumanReviewAgent(BaseAgent):
    name = "human_review"
    capabilities = frozenset({"human_review"})

    def __init__(self, reviewer: Reviewer) -> None:
        super().__init__()
        self.reviewer = reviewer

    def _run(self, task: AgentTask, ctx: RunContext) -> AgentResult:
        analysis: AnalysisResult = ctx.session.require(ANALYSIS_KEY)
        items = [f for f in analysis.findings if f.needs_human_review]
        if not items:
            ctx.session.put(REVIEW_KEY, ReviewOutcome(status="not_required"), written_by=self.name)
            return self.result(task, AgentStatus.SKIPPED, "no high-severity findings; review not required")

        self.decision(f"{len(items)} finding(s) routed to human review via {self.reviewer.name}",
                      findings=[f.id for f in items])
        outcome = ReviewOutcome(status="pending", reviewer=self.reviewer.name)
        try:
            decisions = self.reviewer.review(items)
        except ReviewUnavailable as exc:
            outcome.issues.append(str(exc))
            decisions = []
        outcome.reviewer = self.reviewer.name  # a file reviewer learns its name while reading

        lookup = findings_by_id(items)
        decided: set[str] = set()
        for d in decisions:
            problem = d.validate()
            if d.finding_id not in lookup:
                problem = problem or f"finding {d.finding_id} is not awaiting review"
            if d.finding_id in decided:
                problem = problem or f"duplicate decision for {d.finding_id}; first one kept"
            if problem:
                outcome.issues.append(problem)
                continue

            finding = lookup[d.finding_id]
            record = {"finding_id": d.finding_id, "action": d.action, "note": d.note,
                      "reviewer": self.reviewer.name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if d.action == "dismiss":
                finding.suppressed = True
                finding.suppression_reason = f"dismissed in review: {d.note or 'no note'}"
                if d.remember:
                    ctx.long_term.suppress(ctx.client_id, finding.fingerprint, d.note or "expected activity",
                                           self.reviewer.name)
                    record["remembered"] = True
            elif d.action == "set_severity":
                record["previous_severity"] = finding.severity.value
                finding.severity = Severity(str(d.severity))
                record["new_severity"] = finding.severity.value
            finding.review = record
            outcome.decisions.append(record)
            decided.add(d.finding_id)
            self.decision(f"reviewer {d.action} {d.finding_id}", finding=d.finding_id, action=d.action)

        outcome.pending_ids = [f.id for f in items if f.id not in decided]
        outcome.status = "pending" if outcome.pending_ids else "completed"
        ctx.session.put(REVIEW_KEY, outcome, written_by=self.name)

        summary = f"{len(decided)} decided, {len(outcome.pending_ids)} pending"
        status = AgentStatus.PENDING if outcome.pending_ids else AgentStatus.SUCCESS
        return self.result(task, status, summary, warnings=outcome.issues)
