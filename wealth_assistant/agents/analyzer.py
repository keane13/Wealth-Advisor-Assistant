"""Analyzer Agent: financial analysis, anomaly detection and comparison with history.

Each analysis module is called through the tool registry and isolated from the
others: if anomaly detection fails, cash-flow and risk results are still delivered and
the run is marked partial. Long-term memory supplies two things: findings a reviewer
previously dismissed as expected (suppressed, but kept for audit), and the previous
run's summary (used to label findings new or recurring and to compute net-worth change).
"""

from __future__ import annotations

from typing import Any

from ..core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from ..core.context import RunContext
from ..domain.models import AnalysisResult, ClientSnapshot, Finding
from .data_fetcher import SNAPSHOT_KEY

ANALYSIS_KEY = "analysis"
ALL_CHECKS = ("cashflow", "risk", "anomalies")


class AnalyzerAgent(BaseAgent):
    name = "analyzer"
    capabilities = frozenset({"analyze"})

    def _run(self, task: AgentTask, ctx: RunContext) -> AgentResult:
        snapshot: ClientSnapshot = ctx.session.require(SNAPSHOT_KEY)
        checks = tuple(task.params.get("checks", ALL_CHECKS))
        unknown = set(checks) - set(ALL_CHECKS)
        if unknown or not checks:
            return self.result(task, AgentStatus.FAILED,
                               errors=[f"invalid checks requested: {sorted(unknown) or 'none'}"])

        result = AnalysisResult()
        errors: list[str] = []

        # Cash flow always runs: it supplies the income baseline for anomaly thresholds
        # and the monthly spend for liquidity, even when it was not requested directly.
        cashflow_metrics = self._apply(ctx, "analysis.cashflow", result, errors, "cashflow", snapshot=snapshot)
        if "risk" in checks:
            self._apply(ctx, "analysis.risk_signals", result, errors, "risk",
                        snapshot=snapshot, cashflow=cashflow_metrics)
        if "anomalies" in checks:
            income = (cashflow_metrics or {}).get("median_monthly_income")
            if income is None:
                self.decision("income baseline unavailable; anomaly thresholds fall back to absolute values")
            self._apply(ctx, "analysis.detect_anomalies", result, errors, None,
                        snapshot=snapshot, median_monthly_income=income)

        for f in result.findings:
            f.client_id = ctx.client_id
        self._apply_suppressions(ctx, result)
        result.history = self._compare_with_history(ctx, result)
        result.findings.sort(key=lambda f: (-f.severity.rank, f.rule, f.id))
        ctx.session.put(ANALYSIS_KEY, result, written_by=self.name)

        modules_run = 1 + ("risk" in checks) + ("anomalies" in checks)
        if len(errors) >= modules_run:
            return self.result(task, AgentStatus.FAILED, "all analysis modules failed", errors=errors)
        active = result.active_findings
        summary = (f"{len(active)} active finding(s), {len(result.findings) - len(active)} suppressed, "
                   f"{len(result.skipped_checks)} check(s) skipped")
        return self.result(task, AgentStatus.PARTIAL if errors else AgentStatus.SUCCESS, summary,
                           warnings=result.skipped_checks, errors=errors)

    def _apply(self, ctx: RunContext, tool: str, result: AnalysisResult, errors: list[str],
               metrics_key: str | None, **kwargs: Any) -> dict[str, Any] | None:
        res = ctx.tools.call(tool, **kwargs)
        if not res.ok:
            errors.append(f"{tool}: {res.error}")
            result.skipped_checks.append(f"{tool} unavailable: {res.error}")
            return None
        data = res.data
        result.findings.extend(data.get("findings", []))
        result.skipped_checks.extend(data.get("skipped", []))
        if metrics_key:
            result.metrics[metrics_key] = data.get("metrics", {})
            return data.get("metrics", {})
        return None

    def _apply_suppressions(self, ctx: RunContext, result: AnalysisResult) -> None:
        suppressions = ctx.long_term.suppressions(ctx.client_id)
        for f in result.findings:
            if f.fingerprint in suppressions:
                s = suppressions[f.fingerprint]
                f.suppressed = True
                f.suppression_reason = f"dismissed by {s.get('reviewer')} on {s.get('at', '?')[:10]}: {s.get('reason')}"
                self.decision(f"suppressing {f.id} ({f.rule}) per earlier reviewer decision", finding=f.id)

    def _compare_with_history(self, ctx: RunContext, result: AnalysisResult) -> dict[str, Any]:
        previous = ctx.long_term.last_summary(ctx.client_id)
        if not previous:
            return {"previous_run": None}
        before = set(previous.get("active_fingerprints", []))
        active = result.active_findings
        for f in active:
            f.evidence.setdefault("history", "recurring" if f.fingerprint in before else "new")
        now_fps = {f.fingerprint for f in active}
        history: dict[str, Any] = {
            "previous_run": {"run_id": previous.get("run_id"), "at": previous.get("at"), "as_of": previous.get("as_of")},
            "new_findings": sum(1 for f in active if f.fingerprint not in before),
            "recurring_findings": sum(1 for f in active if f.fingerprint in before),
            "no_longer_active": len(before - now_fps),
        }
        nw_now = result.metrics.get("risk", {}).get("net_worth")
        nw_before = previous.get("net_worth")
        if nw_now is not None and nw_before is not None:
            history["net_worth_change"] = round(nw_now - nw_before, 2)
        return history


def findings_by_id(findings: list[Finding]) -> dict[str, Finding]:
    return {f.id: f for f in findings}
