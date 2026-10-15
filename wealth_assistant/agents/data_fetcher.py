"""Data Fetcher Agent: gathers client data from every source and prepares a validated snapshot.

The two sources have different criticality:
* client financial data is *required*: without it there is nothing to analyse.
* the CRM profile is *enrichment*: without it the run continues degraded, and the
  checks that need it (suitability, KYC) are explicitly skipped.
"""

from __future__ import annotations

from datetime import date, datetime

from ..core.agent import AgentResult, AgentStatus, AgentTask, BaseAgent
from ..core.concurrency import submit_with_context
from ..core.context import RunContext
from ..core.errors import DataValidationError
from ..domain.models import ClientSnapshot
from ..domain.parsing import parse_profile

SNAPSHOT_KEY = "snapshot"


class DataFetcherAgent(BaseAgent):
    name = "data_fetcher"
    capabilities = frozenset({"fetch_client_data"})

    def _run(self, task: AgentTask, ctx: RunContext) -> AgentResult:
        cid = ctx.client_id
        # The two sources are independent, so fetch them concurrently. CRM latency then
        # does not add to the file read.
        raw_future = submit_with_context(ctx.tools.call, "client_data.load", client_id=cid)
        crm_future = submit_with_context(ctx.tools.call, "crm.get_profile", client_id=cid)
        raw_res, crm_res = raw_future.result(), crm_future.result()

        if not raw_res.ok:
            return self.result(task, AgentStatus.FAILED, "client financial data unavailable",
                               errors=[f"client_data.load: {raw_res.error}"])
        parsed = ctx.tools.call("client_data.parse", raw=raw_res.data, client_id=cid)
        if not parsed.ok:
            return self.result(task, AgentStatus.FAILED, "client financial data invalid",
                               errors=[f"client_data.parse: {parsed.error}"])

        snapshot: ClientSnapshot = parsed.data
        dq = snapshot.data_quality
        dq.sources["client_data"] = "ok"
        if dq.rejected_records:
            self.decision(f"dropped {dq.rejected_records} invalid record(s); continuing with valid data",
                          rejected=dq.rejected_records)

        crm_ok = self._attach_profile(snapshot, crm_res, ctx)
        ctx.session.put(SNAPSHOT_KEY, snapshot, written_by=self.name)

        status = AgentStatus.SUCCESS if crm_ok and not dq.rejected_records else AgentStatus.PARTIAL
        summary = (f"{len(snapshot.transactions)} transactions, {len(snapshot.holdings)} holdings; "
                   f"CRM: {dq.sources.get('crm')}")
        return self.result(task, status, summary, warnings=list(dq.warnings))

    def _attach_profile(self, snapshot: ClientSnapshot, crm_res, ctx: RunContext) -> bool:
        """Attach the CRM profile if possible. Returns True only for a fresh, valid profile."""
        dq = snapshot.data_quality
        if not crm_res.ok:
            reason = ("client not found in CRM" if crm_res.error_type == "NotFoundError"
                      else f"CRM unavailable and no cached profile ({crm_res.error})")
            dq.sources["crm"] = "unavailable"
            dq.warn(f"{reason}; suitability and KYC checks will be skipped")
            self.decision(f"proceeding without CRM profile: {reason}", error_type=crm_res.error_type)
            return False

        payload = crm_res.data
        try:
            profile, warnings = parse_profile(payload["profile"], ctx.client_id)
        except (DataValidationError, KeyError, TypeError) as exc:
            dq.sources["crm"] = "invalid"
            dq.warn(f"CRM profile rejected: {exc}")
            return False
        for w in warnings:
            dq.warn(w)
        snapshot.profile = profile

        if payload.get("source") == "cache":
            age = _age_days(payload.get("cached_at"), snapshot.as_of)
            dq.sources["crm"] = f"degraded (cached profile from {payload.get('cached_at')})"
            dq.warn(f"CRM unavailable; using cached profile ({age} day(s) old)")
            if age is not None and age > ctx.settings.crm_cache_max_age_days:
                dq.warn(f"cached CRM profile is older than {ctx.settings.crm_cache_max_age_days} days; "
                        "verify risk tolerance with the client")
            self.decision("CRM down: using cached profile from long-term memory", cached_at=payload.get("cached_at"))
            return False

        dq.sources["crm"] = "ok"
        ctx.long_term.cache_crm_profile(ctx.client_id, payload["profile"])
        return not warnings


def _age_days(cached_at: str | None, as_of: date) -> int | None:
    if not cached_at:
        return None
    try:
        cached = datetime.fromisoformat(cached_at).date()
    except ValueError:
        return None
    return max(0, (max(as_of, date.today()) - cached).days)
