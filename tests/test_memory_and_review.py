"""Long-term memory persistence and the human-in-the-loop checkpoint."""

from __future__ import annotations

import json

import pytest

from wealth_assistant.domain.models import Finding, FindingKind, Severity
from wealth_assistant.hitl.review import ConsoleReviewer, FileReviewer, ReviewDecision
from wealth_assistant.memory.long_term import MAX_SUMMARIES, JsonFileMemoryStore
from wealth_assistant.memory.short_term import SessionMemory


def test_json_store_round_trip_and_bounded_history(tmp_path):
    store = JsonFileMemoryStore(tmp_path)
    for i in range(MAX_SUMMARIES + 5):
        store.append_summary("C1", {"run_id": str(i)})
    store.cache_crm_profile("C1", {"client_id": "C1"})
    store.suppress("C1", "abc", "expected", "tester")

    fresh = JsonFileMemoryStore(tmp_path)  # a new process sees the same state
    assert len(fresh.summaries("C1")) == MAX_SUMMARIES
    assert fresh.last_summary("C1")["run_id"] == str(MAX_SUMMARIES + 4)
    assert fresh.cached_crm_profile("C1")["profile"] == {"client_id": "C1"}
    assert "abc" in fresh.suppressions("C1")


def test_corrupt_memory_file_is_quarantined_not_fatal(tmp_path):
    (tmp_path / "C1.json").write_text("{ not json", encoding="utf-8")
    store = JsonFileMemoryStore(tmp_path)
    assert store.last_summary("C1") is None
    assert list(tmp_path.glob("C1.corrupt-*.json"))
    store.append_summary("C1", {"run_id": "after"})  # and it keeps working
    assert store.last_summary("C1")["run_id"] == "after"


def test_memory_store_rejects_unsafe_ids(tmp_path):
    with pytest.raises(ValueError):
        JsonFileMemoryStore(tmp_path).last_summary("../escape")


def test_session_memory_records_provenance():
    mem = SessionMemory()
    mem.put("snapshot", 1, written_by="data_fetcher")
    assert mem.require("snapshot") == 1
    assert mem.history()[0].written_by == "data_fetcher"
    with pytest.raises(KeyError):
        mem.require("missing")


def finding(rule="large_transaction"):
    return Finding(rule=rule, kind=FindingKind.ANOMALY, severity=Severity.HIGH, title="t", detail="d",
                   fingerprint_basis=("x",), client_id="C1")


def test_file_reviewer_targets_by_id_and_rule(tmp_path):
    a, b = finding("large_transaction"), finding("spending_spike")
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"reviewer": "me", "decisions": [
        {"finding_id": a.id, "action": "dismiss", "remember": True},
        {"rule": "spending_spike", "action": "confirm"},
    ]}), encoding="utf-8")
    reviewer = FileReviewer(path)
    decisions = reviewer.review([a, b])
    assert [(d.finding_id, d.action) for d in decisions] == [(a.id, "dismiss"), (b.id, "confirm")]
    assert reviewer.name == "file:me"


def test_decision_validation():
    assert ReviewDecision("F", "approve").validate() is not None
    assert ReviewDecision("F", "set_severity", severity="extreme").validate() is not None
    assert ReviewDecision("F", "set_severity", severity="low").validate() is None


def test_console_reviewer_handles_eof_by_leaving_items_pending():
    answers = iter(["c", "looks right"])

    def fake_input(prompt):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    reviewer = ConsoleReviewer(input_fn=fake_input, output_fn=lambda s: None)
    decisions = reviewer.review([finding("a"), finding("b")])
    assert len(decisions) == 1 and decisions[0].action == "confirm"
