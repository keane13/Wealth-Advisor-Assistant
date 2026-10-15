"""Advisor-facing narrative summary.

There are two implementations behind one capability (``report.narrative``):

* ``TemplateNarrativeTool`` is deterministic, needs no network, and is always available.
* ``ClaudeNarrativeTool`` asks Claude for a more readable briefing. It is *optional* and
  sandboxed: it only sees the structured facts that were already computed, without the
  client's name or raw transactions. Its output must pass a grounding check (every
  number in the text must appear in the facts) or it is rejected. The registry then
  falls back to the template.

The LLM only phrases conclusions. It never makes them, so a hallucination can at worst
cost readability and never correctness.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..core.errors import ToolError, TransientToolError
from ..core.tools import Tool

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


class TemplateNarrativeTool(Tool):
    name = "report.narrative_template"
    description = "Deterministic narrative summary built from the structured facts."

    def run(self, *, facts: dict[str, Any]) -> dict[str, Any]:
        m = facts.get("key_metrics", {})
        ccy = facts.get("currency", "")
        parts = [f"Review of client {facts.get('client_id')} as of {facts.get('as_of')}."]

        money = []
        if m.get("net_worth") is not None:
            money.append(f"net worth {ccy} {m['net_worth']:,.2f}")
        if m.get("avg_monthly_income") is not None and m.get("avg_monthly_spend") is not None:
            rate = m.get("savings_rate_pct")
            money.append(f"average monthly income {m['avg_monthly_income']:,.2f} vs spend "
                         f"{m['avg_monthly_spend']:,.2f}" + (f" (savings rate {rate}%)" if rate is not None else ""))
        if money:
            parts.append("Position: " + "; ".join(money) + ".")

        findings = facts.get("findings", [])
        if findings:
            counts: dict[str, int] = {}
            for f in findings:
                counts[f["severity"]] = counts.get(f["severity"], 0) + 1
            order = ["critical", "high", "medium", "low"]
            parts.append(f"{len(findings)} active finding(s): "
                         + ", ".join(f"{counts[s]} {s}" for s in order if s in counts) + ".")
        else:
            parts.append("No active findings.")

        since = facts.get("since_last_review")
        if since:
            change = since.get("net_worth_change")
            parts.append(f"Since the last review: {since.get('new_findings', 0)} new and "
                         f"{since.get('recurring_findings', 0)} recurring finding(s)"
                         + (f", net worth change {change:+,.2f}" if change is not None else "") + ".")

        recs = facts.get("recommendations", [])[:3]
        if recs:
            parts.append("Top priorities: " + "; ".join(f"({n}) {r['title']}" for n, r in enumerate(recs, 1)) + ".")
        pending = sum(1 for f in findings if f.get("review_status") == "pending")
        if pending:
            parts.append(f"{pending} high-severity finding(s) await advisor validation; "
                         "related recommendations are provisional.")
        if facts.get("data_quality_issues"):
            parts.append(f"Data quality: {facts['data_quality_issues']} issue(s) noted; see report.")
        return {"text": " ".join(parts), "source": "template"}


SYSTEM_PROMPT = """You draft internal briefings for a licensed wealth advisor.

You receive structured facts that an analysis pipeline has already computed. Write a briefing of at most 150 words in plain text (no markdown, no bullet characters) that:
- states the client's overall position in one or two sentences,
- explains the most important findings and why they matter,
- names the top recommended actions in priority order,
- says clearly if any findings still await advisor validation.

Rules:
- Use only the facts provided. Do not infer facts that are not there.
- Copy any figure exactly as it appears in the facts (same digits, no rounding, no "k"/"m" abbreviations). Prefer fewer figures over new ones.
- Do not add recommendations beyond those listed.
- Address the advisor, not the client."""


def ungrounded_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that cannot be traced to ``facts``.

    Small counts (0-10) are allowed because they arise naturally from counting listed
    items. Any other number must be a rounding of some fact at the precision it is
    written with: "38%" may stand for 37.8, but "37.5" may not.
    """
    def to_float(token: str) -> float | None:
        try:
            return float(token.replace(",", ""))
        except ValueError:
            return None

    known = [v for v in (to_float(t) for t in _NUMBER.findall(json.dumps(facts))) if v is not None]
    bad = []
    for token in _NUMBER.findall(text):
        value = to_float(token)
        if value is None or (value.is_integer() and 0 <= value <= 10):
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(value - k) <= tolerance for k in known):
            bad.append(token)
    return bad


class ClaudeNarrativeTool(Tool):
    name = "report.narrative_llm"
    description = "Claude-written advisor briefing, grounded in (and checked against) the structured facts."

    def __init__(self, model: str, timeout_s: float = 60.0, client_factory: Callable[[], Any] | None = None) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self._client_factory = client_factory
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                try:
                    import anthropic
                except ImportError as exc:
                    raise ToolError("anthropic SDK not installed (pip install -e '.[llm]')", tool=self.name) from exc
                try:
                    # The SDK already retries 429/5xx/connection errors, so the registry
                    # policy for this tool uses a single attempt to avoid compounding retries.
                    self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=2)
                except Exception as exc:  # e.g. no credentials configured
                    raise ToolError(f"cannot initialise Claude client: {exc}", tool=self.name) from exc
        return self._client

    def run(self, *, facts: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content":
                           f"<facts>\n{json.dumps(facts, indent=2, default=str)}\n</facts>\n\nWrite the briefing."}],
                output_config={"effort": "low"},
                # Opt in to server-side refusal fallbacks: if a safety classifier declines,
                # the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:
            raise _map_sdk_error(exc, self.name) from exc

        if response.stop_reason == "refusal":
            raise ToolError("model declined to write the briefing", tool=self.name)
        if response.stop_reason == "max_tokens":
            raise ToolError("briefing truncated (max_tokens)", tool=self.name)
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        if not text:
            raise ToolError("model returned no text", tool=self.name)
        bad = ungrounded_numbers(text, facts)
        if bad:
            raise ToolError(f"briefing failed grounding check; unsupported figures: {', '.join(bad[:5])}",
                            tool=self.name)
        return {"text": text, "source": "llm", "model": response.model}


def _map_sdk_error(exc: Exception, tool: str) -> ToolError:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - a fake client is in use
        return ToolError(str(exc), tool=tool)
    if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)):
        return TransientToolError(f"Claude API temporarily unavailable: {exc}", tool=tool)
    if isinstance(exc, anthropic.APIStatusError):
        return ToolError(f"Claude API error {exc.status_code}: {exc.message}", tool=tool)
    if "authentication" in str(exc).lower():
        return ToolError("no Anthropic credentials configured (set ANTHROPIC_API_KEY)", tool=tool)
    return ToolError(f"Claude call failed: {type(exc).__name__}: {exc}", tool=tool)
