"""Cash-flow and structural risk analysis."""

from __future__ import annotations

from datetime import date

from factories import monthly_salary, profile, snapshot, tx

from wealth_assistant.config import RiskThresholds
from wealth_assistant.domain.models import Account, Holding, Liability, RiskTolerance, Severity
from wealth_assistant.tools.analytics import CashflowAnalysisTool, RiskSignalsTool


def rules(result):
    return sorted(f.rule for f in result["findings"])


def test_cashflow_metrics_and_negative_cashflow():
    txs = monthly_salary(amount=5_000) + [tx(date(2026, m, 10), -6_500, "Costs", "living") for m in range(4, 10)]
    result = CashflowAnalysisTool().run(snapshot=snapshot(txs))
    m = result["metrics"]
    assert m["avg_monthly_income"] == 5_000 and m["avg_monthly_spend"] == 6_500
    assert m["savings_rate_pct"] == -30.0
    assert rules(result) == ["negative_cashflow"]
    assert result["findings"][0].severity is Severity.HIGH


def test_internal_transfers_are_not_spending():
    txs = monthly_salary() + [tx(date(2026, m, 16), -9_000, "Savings", "internal_transfer", "transfer")
                              for m in range(4, 10)]
    result = CashflowAnalysisTool().run(snapshot=snapshot(txs))
    assert result["metrics"]["avg_monthly_spend"] == 0 and result["findings"] == []


def test_declining_cashflow_trend():
    spend = [1_000, 2_000, 3_000, 4_500, 6_000, 7_500]
    txs = monthly_salary() + [tx(date(2026, 4 + i, 10), -s, "Costs", "living") for i, s in enumerate(spend)]
    assert "declining_cashflow" in rules(CashflowAnalysisTool().run(snapshot=snapshot(txs)))


def test_no_income_month_does_not_divide_by_zero():
    txs = [tx(date(2026, m, 10), -500, "Costs", "living") for m in range(7, 10)]
    result = CashflowAnalysisTool().run(snapshot=snapshot(txs))
    assert result["metrics"]["savings_rate_pct"] is None
    assert rules(result) == ["negative_cashflow"]


def test_partial_month_excluded_from_averages():
    txs = monthly_salary(months=6) + [tx("2026-09-10", -100, "x")]
    result = CashflowAnalysisTool().run(snapshot=snapshot(txs, as_of=date(2026, 9, 15)))
    assert result["metrics"]["months_in_averages"] == 5
    assert any("partial month" in s for s in result["skipped"])


def test_empty_transactions():
    result = CashflowAnalysisTool().run(snapshot=snapshot([]))
    assert result["findings"] == [] and result["skipped"]


CF = {"avg_monthly_income": 10_000.0, "avg_monthly_spend": 5_000.0}


def test_concentration_suitability_and_diversified_exemption():
    holdings = [Holding("NVDA", "equity", 400_000, "Technology"), Holding("VTI", "equity", 400_000, "Diversified"),
                Holding("BND", "bond", 200_000)]
    result = RiskSignalsTool(RiskThresholds()).run(
        snapshot=snapshot(holdings=holdings, profile=profile(RiskTolerance.CONSERVATIVE)), cashflow=CF)
    assert rules(result) == ["concentration", "suitability_mismatch"]
    conc = next(f for f in result["findings"] if f.rule == "concentration")
    assert conc.evidence["symbol"] == "NVDA" and conc.severity is Severity.HIGH
    assert result["metrics"]["equity_pct"] == 80.0


def test_suitability_skipped_without_profile_not_passed():
    result = RiskSignalsTool(RiskThresholds()).run(
        snapshot=snapshot(holdings=[Holding("VTI", "equity", 100, "Diversified")]), cashflow=CF)
    assert "suitability_mismatch" not in rules(result)
    assert any("suitability check" in s for s in result["skipped"])


def test_low_liquidity_and_high_dti_and_kyc():
    snap = snapshot(accounts=[Account("a", "checking", 4_000)],
                    liabilities=[Liability("loan", 100_000, 5_500)],
                    profile=profile(kyc_status="expired", last_review_date="2024-01-01"))
    result = RiskSignalsTool(RiskThresholds()).run(snapshot=snap, cashflow=CF)
    assert rules(result) == ["high_debt_to_income", "kyc_not_current", "low_liquidity", "review_overdue"]
    assert result["metrics"]["liquidity_months"] == 0.8


def test_missing_cashflow_context_skips_liquidity():
    result = RiskSignalsTool(RiskThresholds()).run(snapshot=snapshot(), cashflow=None)
    assert any("liquidity check" in s for s in result["skipped"])
