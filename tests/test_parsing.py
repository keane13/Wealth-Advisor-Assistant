"""Untrusted input: record-level rejection, document-level failure."""

from __future__ import annotations

import pytest

from wealth_assistant.core.errors import DataValidationError
from wealth_assistant.domain.models import RiskTolerance
from wealth_assistant.domain.parsing import parse_amount, parse_client_document, parse_profile


def doc(**overrides):
    base = {
        "client_id": "X1", "name": "X", "base_currency": "USD", "as_of": "2026-09-30",
        "accounts": [{"account_id": "a", "type": "checking", "balance": 100}],
        "holdings": [], "liabilities": [],
        "transactions": [{"id": "t1", "date": "2026-09-01", "amount": -10, "merchant": "m", "category": "c"}],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("value,expected", [(10, 10.0), ("1,250.50", 1250.5), (-3.456, -3.46)])
def test_parse_amount_accepts_numbers_and_numeric_strings(value, expected):
    assert parse_amount(value) == expected


@pytest.mark.parametrize("value", [True, None, "abc", float("nan"), float("inf")])
def test_parse_amount_rejects_garbage(value):
    with pytest.raises((ValueError, TypeError)):
        parse_amount(value)


def test_bad_records_are_dropped_and_reported_not_fatal():
    snap = parse_client_document(doc(transactions=[
        {"id": "ok", "date": "2026-09-01", "amount": -10},
        {"id": "bad_amount", "date": "2026-09-01", "amount": "abc"},
        {"id": "bad_date", "date": "2026-13-45", "amount": -1},
        {"id": "ok", "date": "2026-09-02", "amount": -5},              # duplicate id
        {"id": "fx", "date": "2026-09-02", "amount": -5, "currency": "EUR"},
        {"id": "zero", "date": "2026-09-02", "amount": 0},
        {"id": "future", "date": "2026-10-15", "amount": -5},
        "not-a-record",
        {"date": "2026-09-03", "amount": -7},                           # missing id -> synthetic id
    ]), "X1")
    assert [t.id for t in snap.transactions] == ["ok", "ROW-8"]
    assert snap.data_quality.rejected_records == 7
    assert any("dated after as_of" in w for w in snap.data_quality.warnings)
    assert any("no FX conversion" in w for w in snap.data_quality.warnings)


def test_client_id_mismatch_is_fatal():
    with pytest.raises(DataValidationError):
        parse_client_document(doc(client_id="OTHER"), "X1")


def test_non_object_document_is_fatal():
    with pytest.raises(DataValidationError):
        parse_client_document(["not", "an", "object"], "X1")


def test_missing_fields_get_safe_defaults_with_warnings():
    raw = doc()
    del raw["base_currency"], raw["as_of"], raw["holdings"]
    snap = parse_client_document(raw, "X1")
    assert snap.base_currency == "USD"
    assert snap.as_of.isoformat() == "2026-09-01"  # falls back to the latest transaction
    assert snap.holdings == []
    assert len(snap.data_quality.warnings) == 3


def test_negative_holding_value_rejected():
    snap = parse_client_document(doc(holdings=[{"symbol": "A", "asset_class": "equity", "market_value": -1}]), "X1")
    assert snap.holdings == [] and snap.data_quality.rejected_records == 1


def test_profile_with_unknown_risk_tolerance_degrades_gracefully():
    profile, warnings = parse_profile({"client_id": "X1", "risk_tolerance": "balanced", "age": "old"}, "X1")
    assert profile.risk_tolerance is None and profile.age is None
    assert len(warnings) == 2


def test_profile_parses_valid_payload():
    profile, warnings = parse_profile({"client_id": "X1", "risk_tolerance": "Aggressive", "age": 30}, "X1")
    assert profile.risk_tolerance is RiskTolerance.AGGRESSIVE and not warnings


def test_profile_for_wrong_client_rejected():
    with pytest.raises(DataValidationError):
        parse_profile({"client_id": "OTHER"}, "X1")
