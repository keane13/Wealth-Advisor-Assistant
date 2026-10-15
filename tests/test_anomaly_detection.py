"""Anomaly detectors: each must fire on its pattern and stay quiet on normal behaviour."""

from __future__ import annotations

from datetime import date

from factories import monthly_salary, snapshot, tx

from wealth_assistant.config import AnomalyThresholds
from wealth_assistant.domain.models import Severity
from wealth_assistant.tools.anomaly import AnomalyDetectionTool, Detector

T = AnomalyThresholds()


def run(transactions, income=10_000.0, detectors=None):
    tool = AnomalyDetectionTool(T, detectors)
    return tool.run(snapshot=snapshot(transactions), median_monthly_income=income)


def rules(result):
    return sorted(f.rule for f in result["findings"])


def regular_spending():
    """Six months of unremarkable groceries/dining: enough samples for statistics, nothing unusual."""
    out = monthly_salary()
    for m in range(4, 10):
        for d, amt in ((3, 80), (10, 95), (17, 70), (24, 88)):
            out.append(tx(date(2026, m, d), -amt, "FreshMart", "groceries"))
        for d, amt in ((5, 30), (12, 42), (19, 35)):
            out.append(tx(date(2026, m, d), -amt, "Cafe", "dining"))
    return out


def test_normal_behaviour_produces_no_findings():
    assert run(regular_spending())["findings"] == []


def test_large_first_time_outflow_is_high_severity():
    result = run(regular_spending() + [tx("2026-08-14", -18_500, "Escrow Ltd", "transfer", "wire")])
    [finding] = result["findings"]
    assert finding.rule == "large_transaction" and finding.severity is Severity.HIGH
    assert finding.evidence["first_time_payee"] is True


def test_recurring_large_payee_is_not_flagged():
    rent = [tx(date(2026, m, 2), -6_000, "Landlord", "housing", "ach") for m in range(4, 10)]
    assert run(regular_spending() + rent)["findings"] == []


def test_large_threshold_falls_back_to_absolute_when_income_unknown():
    result = run([tx("2026-09-01", -4_000, "Shop")], income=None)
    assert result["findings"] == []  # 4,000 < 5,000 absolute fallback
    assert any("income unknown" in s for s in result["skipped"])


def test_duplicate_charge_detected_but_repeat_cash_is_not():
    txs = regular_spending() + [
        tx("2026-09-05", -1249.99, "BestElectro"), tx("2026-09-06", -1249.99, "BestElectro"),
        tx("2026-09-07", -200, "ATM", "cash_withdrawal", "cash"), tx("2026-09-07", -200, "ATM", "cash_withdrawal", "cash"),
    ]
    result = run(txs)
    dupes = [f for f in result["findings"] if f.rule == "duplicate_transaction"]
    assert len(dupes) == 1 and dupes[0].evidence["merchant"] == "BestElectro"


def test_cash_structuring_is_critical_and_claims_its_transactions():
    txs = regular_spending() + [tx(f"2026-09-{d:02d}", -a, "Branch", "cash_withdrawal", "cash")
                                for d, a in ((8, 9_500), (10, 9_200), (12, 9_800))]
    result = run(txs)
    assert rules(result) == ["cash_structuring"]  # not also reported as three large transactions
    assert result["findings"][0].severity is Severity.CRITICAL


def test_two_withdrawals_are_not_structuring():
    txs = regular_spending() + [tx(f"2026-09-{d:02d}", -9_500, "Branch", "cash_withdrawal", "cash") for d in (8, 10)]
    assert "cash_structuring" not in rules(run(txs))


def test_statistical_outlier_uses_robust_scores():
    result = run(regular_spending() + [tx("2026-07-19", -1_150, "Le Grand", "dining")])
    [finding] = result["findings"]
    assert finding.rule == "statistical_outlier" and finding.evidence["modified_z"] > T.robust_z_threshold


def test_sensitive_category_spike_is_high_and_not_double_reported_as_outliers():
    txs = regular_spending()
    txs += [tx(date(2026, m, 22), -40, "BetNow", "gambling") for m in range(4, 9)]
    txs += [tx(f"2026-09-{d:02d}", -600, "BetNow", "gambling") for d in (3, 10, 17, 24)]
    result = run(txs)
    assert rules(result) == ["spending_spike"]
    assert result["findings"][0].severity is Severity.HIGH


def test_spike_needs_three_months_of_history():
    txs = [tx("2026-08-01", -50, "Shop"), tx("2026-09-01", -5_000, "Shop")]
    result = run(txs, income=None)
    assert any("3 months" in s for s in result["skipped"])


def test_no_outflows_is_handled():
    result = run(monthly_salary())
    assert result["findings"] == [] and result["skipped"]


def test_a_crashing_detector_does_not_hide_the_others():
    class Broken(Detector):
        rule = "broken"

        def detect(self, ctx):
            raise RuntimeError("bad detector")

    from wealth_assistant.tools.anomaly import LargeTransactionDetector
    result = run(regular_spending() + [tx("2026-08-14", -18_500, "Escrow", "transfer", "wire")],
                 detectors=[Broken(), LargeTransactionDetector()])
    assert rules(result) == ["large_transaction"]
    assert any("broken: detector failed" in s for s in result["skipped"])


def test_fingerprints_are_stable_across_runs():
    txs = regular_spending() + [tx("2026-08-14", -18_500, "Escrow", "transfer", "wire", tx_id="W1")]
    a, b = run(txs)["findings"][0], run(txs)["findings"][0]
    assert a.fingerprint == b.fingerprint
