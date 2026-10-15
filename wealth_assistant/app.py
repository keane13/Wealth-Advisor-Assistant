"""Composition root: the one place that knows the concrete classes.

Everything else depends on abstractions (``Tool`` names, ``BaseAgent``, ``Reviewer``,
``LongTermMemory``). Swapping the mock CRM for a real client, the JSON store for a
database, or the console reviewer for a web UI is a change here only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents.analyzer import AnalyzerAgent
from .agents.data_fetcher import DataFetcherAgent
from .agents.orchestrator import OrchestratorAgent
from .agents.recommender import RecommendationAgent
from .agents.reviewer import HumanReviewAgent
from .config import Settings
from .core.context import RunContext
from .core.tools import ToolPolicy, ToolRegistry
from .hitl.review import DeferredReviewer, Reviewer
from .memory.long_term import JsonFileMemoryStore, LongTermMemory
from .tools.analytics import CashflowAnalysisTool, RiskSignalsTool
from .tools.anomaly import AnomalyDetectionTool
from .tools.client_data import ClientDataLoadTool, ClientDataParseTool
from .tools.crm import CachedCRMProfileTool, CRMProfileTool, MockCRMService
from .tools.narrative import ClaudeNarrativeTool, TemplateNarrativeTool


def build_tool_registry(settings: Settings, memory: LongTermMemory) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ClientDataLoadTool(settings.data_dir / "clients"))
    registry.register(ClientDataParseTool())

    crm = MockCRMService(
        settings.data_dir / "crm" / "profiles.json",
        failure_rate=settings.crm_failure_rate,
        outage=settings.crm_outage,
        latency_s=settings.crm_latency_s,
        seed=settings.random_seed,
    )
    registry.register(CachedCRMProfileTool(memory))
    registry.register(CRMProfileTool(crm), ToolPolicy(
        max_attempts=settings.crm_max_attempts, backoff_base_s=0.2,
        timeout_s=settings.crm_timeout_s, fallback=CachedCRMProfileTool.name,
    ))

    registry.register(CashflowAnalysisTool())
    registry.register(RiskSignalsTool(settings.risk))
    registry.register(AnomalyDetectionTool(settings.anomaly))

    registry.register(TemplateNarrativeTool())
    if settings.use_llm:
        # One attempt: the Anthropic SDK already retries transient errors internally.
        # Any failure, including a permanent one like a grounding-check rejection,
        # falls back to the deterministic template.
        registry.register(
            ClaudeNarrativeTool(settings.llm_model, timeout_s=settings.llm_timeout_s),
            ToolPolicy(max_attempts=1, fallback=TemplateNarrativeTool.name, fallback_on_permanent=True),
            name="report.narrative",
        )
    else:
        registry.register(TemplateNarrativeTool(), name="report.narrative")
    return registry


@dataclass
class WealthAssistant:
    settings: Settings
    memory: LongTermMemory
    tools: ToolRegistry
    orchestrator: OrchestratorAgent

    def review_client(self, client_id: str, plan: str = "full_review") -> dict[str, Any]:
        ctx = RunContext(client_id=client_id, tools=self.tools, long_term=self.memory, settings=self.settings)
        return self.orchestrator.run(ctx, plan)


def build_assistant(settings: Settings | None = None, reviewer: Reviewer | None = None,
                    memory: LongTermMemory | None = None) -> WealthAssistant:
    settings = settings or Settings.from_env()
    memory = memory or JsonFileMemoryStore(settings.memory_dir)
    tools = build_tool_registry(settings, memory)
    agents = [
        DataFetcherAgent(),
        AnalyzerAgent(),
        HumanReviewAgent(reviewer or DeferredReviewer()),
        RecommendationAgent(),
    ]
    return WealthAssistant(settings, memory, tools, OrchestratorAgent(agents))
