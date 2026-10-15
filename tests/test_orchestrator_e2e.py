"""End-to-end runs through the orchestrator against the bundled mock data."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wealth_assistant.agents.analyzer import AnalyzerAgent
from wealth_assistant.agents.data_fetcher import DataFetcherAgent
from wealth_assistant.agents.orchestrator import OrchestratorAgent
from wealth_assistant.agents.recommender import RecommendationAgent
from wealth_assistant.agents.reviewer import HumanReviewAgent
from wealth_assistant.app import build_assistant
from wealth_assistant.config import PROJECT_ROOT
from wealth_assistant.core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from wealth_assistant.core.errors import RoutingError
from wealth_assistant.hitl.review import AutoConfirmReviewer, DeferredReviewer, FileReviewer

SAMPLE_DECISIONS = PROJECT_ROOT / "samples" / "review_C001.json"


def rules(report):
    return {f["rule"] for f in report["findings"]}


def statuses(report):
    return {t["step"]: t["status"] for t in report["execution_trace"]}


def test_full_review_pending_when_no_human_available(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C001")
    assert report["status"] == "pending_review"
    assert rules(report) == {"concentration", "large_transaction", "spending_spike", "duplicate_transaction",
                             "statistical_outlier", "suitability_mismatch"}
    assert report["human_review"]["status"] == "pending"
    provisional = [r for r in report["recommendations"] if r["status"] == "awaiting_validation"]
    assert provisional, "recommendations depending on unreviewed findings must be marked provisional"
    json.dumps(report)  # the report must always be serialisable


def test_review_decisions_apply_and_dismissals_are_remembered(settings, memory):
    assistant = build_assistant(settings, FileReviewer(SAMPLE_DECISIONS), memory)
    first = assistant.review_client("C001")
    assert first["status"] == "completed"
    assert [f["rule"] for f in first["suppressed_findings"]] == ["large_transaction"]
    nvda = next(f for f in first["findings"] if f["rule"] == "concentration")
    assert nvda["severity"] == "critical" and nvda["review"]["previous_severity"] == "high"
    assert first["recommendations"][0]["title"].startswith("Reduce concentration")

    second = build_assistant(settings, DeferredReviewer(), memory).review_client("C001")
    assert "large_transaction" not in rules(second)                 # suppressed from long-term memory
    assert second["suppressed_findings"][0]["suppression_reason"].startswith("dismissed by file:j.rivera")
    assert second["history"]["previous_run"]["run_id"] == first["run_id"]
    assert second["history"]["new_findings"] == 0


def test_crm_outage_falls_back_to_cached_profile(settings, memory):
    build_assistant(settings, AutoConfirmReviewer(), memory).review_client("C002")  # warms the cache
    outage = replace(settings, crm_outage=True)
    report = build_assistant(outage, AutoConfirmReviewer(), memory).review_client("C002")
    assert report["data_quality"]["sources"]["crm"].startswith("degraded")
    assert report["client"]["risk_tolerance"] == "conservative"
    assert "suitability_mismatch" in rules(report)                  # still possible thanks to the cache
    assert report["status"] == "completed_with_warnings"


def test_crm_outage_without_cache_degrades_gracefully(settings, memory):
    report = build_assistant(replace(settings, crm_outage=True), AutoConfirmReviewer(), memory).review_client("C002")
    assert report["data_quality"]["sources"]["crm"] == "unavailable"
    assert "suitability_mismatch" not in rules(report)
    assert any("suitability check" in s for s in report["skipped_checks"])
    assert "cash_structuring" in rules(report)                      # transaction analysis is unaffected
    assert report["status"] == "completed_with_warnings"


def test_flaky_crm_recovers_with_retries(settings, memory):
    flaky = replace(settings, crm_failure_rate=0.5, random_seed=3)
    report = build_assistant(flaky, AutoConfirmReviewer(), memory).review_client("C001")
    assert report["data_quality"]["sources"]["crm"] == "ok"


def test_compliance_escalation_is_top_priority(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C002")
    top = report["recommendations"][0]
    assert top["priority"] == 1 and "Compliance" in top["title"]
    assert {"cash_structuring", "suitability_mismatch", "low_liquidity", "kyc_not_current"} <= rules(report)
    assert "large_transaction" not in rules(report)                  # recurring medical payee is not "large"


def test_messy_data_is_cleaned_not_fatal(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C003")
    assert report["status"] == "completed_with_warnings"
    assert report["data_quality"]["rejected_records"] == 9
    assert any("portfolio checks" in s for s in report["skipped_checks"])
    assert any("risk_tolerance" in w for w in report["data_quality"]["warnings"])


def test_healthy_client_has_no_false_positives(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C004")
    assert report["findings"] == []
    assert report["human_review"]["status"] == "not_required"
    assert report["data_quality"]["sources"]["crm"] == "unavailable"   # 404: not in CRM


def test_unknown_client_fails_cleanly_and_skips_downstream(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C999")
    assert report["status"] == "failed"
    assert statuses(report) == {"fetch": "failed", "analyze": "skipped", "review": "skipped", "recommend": "skipped"}
    assert memory.last_summary("C999") is None


@pytest.mark.parametrize("bad_id", ["../../etc/passwd", "", "A" * 40, "C 001"])
def test_invalid_client_ids_rejected_before_any_agent_runs(settings, memory, bad_id):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client(bad_id)
    assert report["status"] == "failed" and report["execution_trace"] == []


def test_anomaly_scan_plan_routes_a_subset(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C001", plan="anomaly_scan")
    assert list(statuses(report)) == ["fetch", "analyze", "review"]
    assert "risk" not in report["metrics"] and "concentration" not in rules(report)


def test_unknown_plan(settings, memory):
    report = build_assistant(settings, DeferredReviewer(), memory).review_client("C001", plan="nope")
    assert report["status"] == "failed" and "unknown plan" in report["errors"][0]


class CrashingAgent(BaseAgent):
    def __init__(self, name: str, capability: str) -> None:
        self.name = name  # type: ignore[misc]
        self.capabilities = frozenset({capability})  # type: ignore[misc]
        super().__init__()

    def _run(self, task: AgentTask, ctx) -> AgentResult:
        raise RuntimeError("simulated bug")


def make_orchestrator(**replace_agents):
    agents = {"fetch": DataFetcherAgent(), "analyze": AnalyzerAgent(),
              "review": HumanReviewAgent(DeferredReviewer()), "recommend": RecommendationAgent()}
    agents.update(replace_agents)
    return OrchestratorAgent(list(agents.values()))


def run_with(orchestrator, settings, memory, client="C001"):
    assistant = build_assistant(settings, DeferredReviewer(), memory)
    assistant.orchestrator = orchestrator
    return assistant.review_client(client)


def test_crash_in_optional_agent_still_yields_a_report(settings, memory):
    report = run_with(make_orchestrator(recommend=CrashingAgent("recommender", "recommend")), settings, memory)
    assert statuses(report)["recommend"] == "failed"
    assert report["status"] == "pending_review" and report["findings"]
    assert any("simulated bug" in e for e in report["errors"])


def test_crash_in_critical_agent_aborts_the_run(settings, memory):
    report = run_with(make_orchestrator(analyze=CrashingAgent("analyzer", "analyze")), settings, memory)
    assert report["status"] == "failed"
    assert statuses(report)["review"] == "skipped" and statuses(report)["recommend"] == "skipped"


def test_routing_is_validated_at_construction():
    with pytest.raises(RoutingError, match="no agent provides"):
        OrchestratorAgent([DataFetcherAgent(), AnalyzerAgent()])
    with pytest.raises(RoutingError, match="claimed by both"):
        OrchestratorAgent([DataFetcherAgent(), DataFetcherAgent()])


def test_agent_status_enum_is_stable():
    assert {s.value for s in AgentStatus} == {"success", "partial", "pending", "skipped", "failed"}
