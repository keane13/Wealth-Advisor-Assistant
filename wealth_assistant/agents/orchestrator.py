"""Orchestrator Agent: the central coordinator.

Responsibilities:
* **Planning.** It picks an execution plan for the request type. Plans are data (a list
  of ``Step``), so adding a workflow does not touch agent code.
* **Routing.** It sends each step's task to the one agent that declares the capability.
  The routing table is checked when the orchestrator is built, so a missing agent is a
  startup error and never a mid-run surprise.
* **Flow control.** It checks each step's dependencies on the shared session memory. A
  failed *critical* step aborts the run. A failed *optional* step only skips the steps
  that depend on it.
* **Outcome.** It derives the overall run status, assembles the report, and stores a
  run summary in long-term memory.

The plan is deterministic on purpose. An LLM planner would add little to a fixed
advisory workflow, and it would make runs non-reproducible and harder to audit, which
matters in a regulated domain. Because plans are data, an LLM planner could later
emit a ``Plan`` without changing anything else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from ..core.context import RunContext
from ..core.errors import RoutingError
from ..domain.validation import is_valid_client_id
from ..logging_setup import agent_var, get_logger, log_event, run_id_var
from ..report import build_report, run_summary
from .analyzer import ANALYSIS_KEY
from .data_fetcher import SNAPSHOT_KEY
from .recommender import RECOMMENDATIONS_KEY
from .reviewer import REVIEW_KEY

logger = get_logger("agents.orchestrator")


@dataclass(frozen=True)
class Step:
    name: str
    task: AgentTask
    requires: tuple[str, ...] = ()
    produces: str | None = None
    critical: bool = False


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    steps: tuple[Step, ...] = field(default_factory=tuple)


FETCH = Step("fetch", AgentTask("fetch_client_data"), produces=SNAPSHOT_KEY, critical=True)
REVIEW = Step("review", AgentTask("human_review"), requires=(ANALYSIS_KEY,), produces=REVIEW_KEY)

PLANS: dict[str, Plan] = {
    "full_review": Plan(
        "full_review", "Complete advisory review: data, analysis, human validation, recommendations.",
        (
            FETCH,
            Step("analyze", AgentTask("analyze"), requires=(SNAPSHOT_KEY,), produces=ANALYSIS_KEY, critical=True),
            REVIEW,
            Step("recommend", AgentTask("recommend"), requires=(SNAPSHOT_KEY, ANALYSIS_KEY),
                 produces=RECOMMENDATIONS_KEY),
        ),
    ),
    "anomaly_scan": Plan(
        "anomaly_scan", "Fast transaction monitoring: anomalies and cash flow only, with human validation.",
        (
            FETCH,
            Step("analyze", AgentTask("analyze", {"checks": ["cashflow", "anomalies"]}),
                 requires=(SNAPSHOT_KEY,), produces=ANALYSIS_KEY, critical=True),
            REVIEW,
        ),
    ),
}


class OrchestratorAgent:
    name = "orchestrator"

    def __init__(self, agents: list[BaseAgent], plans: dict[str, Plan] | None = None) -> None:
        self.plans = plans or PLANS
        self.routes: dict[str, BaseAgent] = {}
        for agent in agents:
            for capability in agent.capabilities:
                if capability in self.routes:
                    raise RoutingError(f"capability '{capability}' claimed by both "
                                       f"{self.routes[capability].name} and {agent.name}")
                self.routes[capability] = agent
        for plan in self.plans.values():
            for step in plan.steps:
                if step.task.type not in self.routes:
                    raise RoutingError(f"plan '{plan.name}' step '{step.name}' needs capability "
                                       f"'{step.task.type}' but no agent provides it")

    def run(self, ctx: RunContext, plan_name: str = "full_review") -> dict[str, Any]:
        run_token, agent_token = run_id_var.set(ctx.run_id), agent_var.set(self.name)
        try:
            return self._run(ctx, plan_name)
        finally:
            agent_var.reset(agent_token)
            run_id_var.reset(run_token)

    def _run(self, ctx: RunContext, plan_name: str) -> dict[str, Any]:
        trace: list[AgentResult] = []
        if not is_valid_client_id(ctx.client_id):
            log_event(logger, logging.ERROR, "run.rejected", f"invalid client id {ctx.client_id!r}")
            return build_report(ctx, plan_name, "failed", trace, errors=[f"invalid client id {ctx.client_id!r}"])
        plan = self.plans.get(plan_name)
        if plan is None:
            return build_report(ctx, plan_name, "failed", trace,
                                errors=[f"unknown plan '{plan_name}'; available: {sorted(self.plans)}"])

        log_event(logger, logging.INFO, "run.start", f"client {ctx.client_id}, plan '{plan.name}'",
                  client_id=ctx.client_id, plan=plan.name)
        aborted_by: str | None = None
        for step in plan.steps:
            if aborted_by:
                trace.append(self._skipped(step, f"run aborted after critical step '{aborted_by}' failed"))
                continue
            missing = [k for k in step.requires if not ctx.session.has(k)]
            if missing:
                log_event(logger, logging.WARNING, "decision",
                          f"skipping '{step.name}': missing inputs {missing}", step=step.name)
                trace.append(self._skipped(step, f"missing inputs: {', '.join(missing)}"))
                continue

            agent = self.routes[step.task.type]
            log_event(logger, logging.INFO, "route", f"step '{step.name}' -> {agent.name}",
                      step=step.name, target=agent.name)
            result = agent.handle(step.task, ctx)
            result.step = step.name
            trace.append(result)

            if result.status is AgentStatus.FAILED and step.critical:
                aborted_by = step.name
                log_event(logger, logging.ERROR, "decision",
                          f"critical step '{step.name}' failed; aborting run", step=step.name)

        status = self._overall_status(trace, aborted_by)
        errors = [e for r in trace for e in r.errors]
        report = build_report(ctx, plan.name, status, trace, errors=errors)
        if status != "failed":
            try:
                ctx.long_term.append_summary(ctx.client_id, run_summary(ctx, report))
            except Exception as exc:  # losing history must not lose the report
                logger.exception("could not persist run summary", extra={"event": "memory.write_failed"})
                report["errors"].append(f"run summary not persisted: {exc}")
        log_event(logger, logging.INFO, "run.finish", f"status={status}", status=status,
                  findings=len(report.get("findings", [])))
        return report

    @staticmethod
    def _skipped(step: Step, reason: str) -> AgentResult:
        return AgentResult(agent="-", task=step.task.type, status=AgentStatus.SKIPPED, summary=reason, step=step.name)

    @staticmethod
    def _overall_status(trace: list[AgentResult], aborted_by: str | None) -> str:
        """failed > pending_review > completed_with_warnings > completed.

        Skipped *checks* inside a successful analysis (e.g. too little history) are
        informational and listed in the report. They do not degrade the status. A
        degraded *agent* or a skipped *step* does.
        """
        if aborted_by:
            return "failed"
        if any(r.status is AgentStatus.PENDING for r in trace):
            return "pending_review"
        degraded = any(r.status in (AgentStatus.PARTIAL, AgentStatus.FAILED) for r in trace)
        degraded |= any(r.status is AgentStatus.SKIPPED and r.agent == "-" for r in trace)
        return "completed_with_warnings" if degraded else "completed"
