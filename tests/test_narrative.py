"""LLM narrative: grounding guardrail and fallback behaviour, using a fake Claude client."""

from __future__ import annotations

from types import SimpleNamespace

from wealth_assistant.core.tools import ToolPolicy, ToolRegistry
from wealth_assistant.tools.narrative import ClaudeNarrativeTool, TemplateNarrativeTool, ungrounded_numbers

FACTS = {
    "client_id": "C1", "as_of": "2026-09-30", "currency": "USD",
    "key_metrics": {"net_worth": 461700.0, "savings_rate_pct": 21.3},
    "findings": [{"id": "F-1", "severity": "high", "title": "Concentrated position in NVDA",
                  "detail": "NVDA is 37.8% of investments", "review_status": "pending"}],
    "recommendations": [{"priority": 2, "title": "Reduce concentration in NVDA", "status": "awaiting_validation"}],
    "data_quality_issues": 0,
}


def fake_client(text: str, stop_reason: str = "end_turn"):
    response = SimpleNamespace(stop_reason=stop_reason, model="claude-test",
                               content=[SimpleNamespace(type="text", text=text)])
    create = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def registry_with(text: str, stop_reason: str = "end_turn") -> ToolRegistry:
    reg = ToolRegistry(sleep=lambda s: None)
    reg.register(TemplateNarrativeTool())
    reg.register(ClaudeNarrativeTool("claude-test", client_factory=lambda: fake_client(text, stop_reason)),
                 ToolPolicy(fallback="report.narrative_template", fallback_on_permanent=True), name="report.narrative")
    return reg


def test_grounding_check_accepts_figures_from_facts_and_rounding():
    text = "Net worth is 461,700.00 USD; NVDA is about 38% of investments, savings rate 21.3%."
    assert ungrounded_numbers(text, FACTS) == []


def test_grounding_check_flags_invented_figures():
    assert ungrounded_numbers("The client could save 2,500 a month.", FACTS) == ["2,500"]


def test_grounded_llm_text_is_used():
    result = registry_with("Net worth 461,700.00. Priority: reduce NVDA concentration (37.8%).").call(
        "report.narrative", facts=FACTS)
    assert result.ok and not result.degraded and result.data["source"] == "llm"


def test_hallucinated_figure_falls_back_to_template():
    result = registry_with("Net worth is 500,000 and rising.").call("report.narrative", facts=FACTS)
    assert result.ok and result.degraded and result.data["source"] == "template"
    assert "grounding" in result.error


def test_refusal_falls_back_to_template():
    result = registry_with("", stop_reason="refusal").call("report.narrative", facts=FACTS)
    assert result.degraded and result.data["source"] == "template"


def test_template_mentions_pending_review():
    text = TemplateNarrativeTool().run(facts=FACTS)["text"]
    assert "await advisor validation" in text and "461,700.00" in text
