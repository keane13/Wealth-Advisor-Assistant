"""Generate the mock client files in data/clients and the CRM data in data/crm.

The data is synthetic and seeded, so it is reproducible. Each client is designed to
exercise a different part of the system. The injected anomalies are listed in
``EXPECTED`` so the README and the tests can refer to them.

    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MONTHS = [(2026, m) for m in range(4, 10)]  # Apr-Sep 2026
AS_OF = "2026-09-30"


class Ledger:
    def __init__(self, client_id: str, seed: int) -> None:
        self.cid = client_id
        self.rng = random.Random(seed)
        self.rows: list[dict] = []

    def add(self, d: date, amount: float, merchant: str, category: str, channel: str = "card",
            description: str | None = None) -> None:
        self.rows.append({
            "id": f"{self.cid}-T{len(self.rows) + 1:04d}",
            "date": d.isoformat(),
            "amount": round(amount, 2),
            "merchant": merchant,
            "category": category,
            "channel": channel,
            "description": description or merchant,
        })

    def spread(self, year: int, month: int, count: int, low: float, high: float, merchants: list[str],
               category: str, channel: str = "card") -> None:
        for _ in range(count):
            day = self.rng.randint(1, 28)
            self.add(date(year, month, day), -self.rng.uniform(low, high), self.rng.choice(merchants), category, channel)

    def sorted_rows(self) -> list[dict]:
        return sorted(self.rows, key=lambda r: (r["date"], r["id"]))


def everyday_spending(lg: Ledger, y: int, m: int, scale: float = 1.0) -> None:
    lg.spread(y, m, 9, 60 * scale, 180 * scale, ["FreshMart", "Green Grocer", "Costco"], "groceries")
    lg.spread(y, m, 7, 20, 90, ["Cafe Uno", "Noodle Bar", "Pizza Place"], "dining")
    lg.spread(y, m, 4, 40, 70, ["Shell", "Chevron"], "transport")
    lg.spread(y, m, 4, 30, 250, ["Uniqlo", "Amazon", "Target"], "shopping")
    lg.add(date(y, m, 12), -lg.rng.uniform(180, 260), "City Power & Water", "utilities", "ach")
    lg.add(date(y, m, 18), -85.0, "Telco Mobile", "utilities", "ach")
    lg.add(date(y, m, 20), -15.99, "StreamFlix", "subscriptions")


def client_c001() -> dict:
    """Mid-career professional, moderate risk. Concentrated portfolio, several transaction anomalies."""
    lg = Ledger("C001", seed=1)
    for y, m in MONTHS:
        lg.add(date(y, m, 1), 6000.0, "Acme Corp Payroll", "salary", "ach")
        lg.add(date(y, m, 15), 6000.0, "Acme Corp Payroll", "salary", "ach")
        lg.add(date(y, m, 3), -2100.0, "First Home Mortgage", "housing", "ach")
        lg.add(date(y, m, 5), -450.0, "AutoFinance Co", "loan_payment", "ach")
        lg.add(date(y, m, 8), -240.0, "SafeLife Insurance", "insurance", "ach")
        lg.add(date(y, m, 16), -1500.0, "Transfer to Savings", "internal_transfer", "transfer")
        everyday_spending(lg, y, m)
        if m < 9:
            lg.add(date(y, m, 22), -lg.rng.uniform(20, 60), "BetNow Online", "gambling")

    # Injected anomalies
    lg.add(date(2026, 8, 14), -18500.0, "Overseas Property Escrow Ltd", "transfer", "wire",
           "International wire - property deposit")                         # large, first-time payee
    lg.add(date(2026, 9, 5), -1249.99, "BestElectro", "shopping")            # duplicate charge...
    lg.add(date(2026, 9, 6), -1249.99, "BestElectro", "shopping")            # ...one day later
    for d in (3, 10, 17, 24):
        lg.add(date(2026, 9, d), -600.0, "BetNow Online", "gambling")        # gambling spike in Sep
    lg.add(date(2026, 7, 19), -1150.0, "Le Grand Restaurant", "dining")      # dining outlier

    return {
        "client_id": "C001",
        "name": "Ayu Pratama",
        "base_currency": "USD",
        "as_of": AS_OF,
        "accounts": [
            {"account_id": "C001-CHK", "type": "checking", "balance": 14200.00},
            {"account_id": "C001-SAV", "type": "savings", "balance": 26500.00},
        ],
        "holdings": [
            {"symbol": "NVDA", "asset_class": "equity", "sector": "Technology", "market_value": 310000.00},
            {"symbol": "VTI", "asset_class": "equity", "sector": "Diversified", "market_value": 280000.00},
            {"symbol": "BND", "asset_class": "bond", "sector": "Diversified", "market_value": 150000.00},
            {"symbol": "AAPL", "asset_class": "equity", "sector": "Technology", "market_value": 60000.00},
            {"symbol": "MMF", "asset_class": "cash", "market_value": 20000.00},
        ],
        "liabilities": [
            {"type": "mortgage", "balance": 385000.00, "monthly_payment": 2100.00},
            {"type": "auto_loan", "balance": 14000.00, "monthly_payment": 450.00},
        ],
        "transactions": lg.sorted_rows(),
    }


def client_c002() -> dict:
    """Retiree, conservative. Aggressive portfolio, thin liquidity, rising costs, cash structuring, expired KYC."""
    lg = Ledger("C002", seed=2)
    for i, (y, m) in enumerate(MONTHS):
        lg.add(date(y, m, 1), 4200.0, "State Pension Fund", "pension", "ach")
        lg.add(date(y, m, 3), 2300.0, "Social Security", "pension", "ach")
        lg.add(date(y, m, 6), -650.0, "HomeEquity Bank", "loan_payment", "ach")
        lg.add(date(y, m, 9), -410.0, "County Property Tax", "housing", "ach")
        everyday_spending(lg, y, m, scale=0.8)
        # Medical and home-repair costs rise steadily: this drives the declining-cash-flow trend.
        lg.add(date(y, m, 14), -(1800 + 650 * i), "Riverside Medical Group", "healthcare", "card")
        lg.add(date(y, m, 25), -(900 + 300 * i), "HandyPro Home Repairs", "home_maintenance", "card")

    # Injected: three cash withdrawals just under 10,000 within five days.
    for d, amt in ((8, 9500.0), (10, 9200.0), (12, 9800.0)):
        lg.add(date(2026, 9, d), -amt, "Branch Cash Withdrawal", "cash_withdrawal", "cash")

    return {
        "client_id": "C002",
        "name": "Robert Hale",
        "base_currency": "USD",
        "as_of": AS_OF,
        "accounts": [
            {"account_id": "C002-CHK", "type": "checking", "balance": 6800.00},
            {"account_id": "C002-SAV", "type": "savings", "balance": 3000.00},
        ],
        "holdings": [
            {"symbol": "VOO", "asset_class": "equity", "sector": "Diversified", "market_value": 180000.00},
            {"symbol": "QQQ", "asset_class": "equity", "sector": "Diversified", "market_value": 140000.00},
            {"symbol": "TSLA", "asset_class": "equity", "sector": "Consumer", "market_value": 45000.00},
            {"symbol": "BND", "asset_class": "bond", "sector": "Diversified", "market_value": 50000.00},
            {"symbol": "MMF", "asset_class": "cash", "market_value": 5000.00},
        ],
        "liabilities": [{"type": "home_equity_line", "balance": 40000.00, "monthly_payment": 650.00}],
        "transactions": lg.sorted_rows(),
    }


def client_c003() -> dict:
    """Freelancer with messy data. Exercises record-level validation and degraded analysis."""
    lg = Ledger("C003", seed=3)
    for y, m in MONTHS[-3:]:  # only three months of history
        if m != 8:  # no income at all in August
            lg.add(date(y, m, 10), 7800.0, "Client Invoice", "freelance_income", "transfer")
        lg.add(date(y, m, 2), -2600.0, "Harbour View Rentals", "housing", "transfer")
        everyday_spending(lg, y, m)
    rows = lg.sorted_rows()
    rows += [
        {"date": "2026-09-11", "amount": -42.5, "merchant": "Kopi Corner", "category": "dining"},           # no id
        {"id": "C003-BAD1", "date": "2026-09-12", "amount": "abc", "merchant": "?", "category": "misc"},   # bad amount
        {"id": "C003-BAD2", "date": "2026-13-45", "amount": -10, "merchant": "?", "category": "misc"},     # bad date
        {"id": rows[0]["id"], "date": "2026-07-03", "amount": -99, "merchant": "Dup", "category": "misc"},  # duplicate id
        {"id": "C003-FUT1", "date": "2026-10-15", "amount": -300, "merchant": "Future Co", "category": "misc"},
        {"id": "C003-FX1", "date": "2026-09-14", "amount": -120, "currency": "USD", "merchant": "US Store",
         "category": "shopping"},                                                                              # wrong currency
        {"id": "C003-STR1", "date": "2026-09-15", "amount": "-1,250.00", "merchant": "Design Supplies",
         "category": "business"},                                                                              # numeric string: accepted
        {"id": "C003-ZERO", "date": "2026-09-16", "amount": 0, "merchant": "Auth Hold", "category": "misc"},
        "not-a-record",
    ]
    return {
        "client_id": "C003",
        "name": "Mei Lin",
        "base_currency": "SGD",
        "as_of": AS_OF,
        "accounts": [{"account_id": "C003-SAV", "type": "savings", "balance": 22000.00},
                     {"account_id": "C003-CHK", "type": "checking", "balance": "n/a"}],
        "holdings": [{"symbol": "STI-ETF", "asset_class": "equity", "market_value": "unknown"}],
        "transactions": rows,
    }


def client_c004() -> dict:
    """Healthy, diversified client who is not in the CRM. Should produce no false positives."""
    lg = Ledger("C004", seed=4)
    for y, m in MONTHS:
        lg.add(date(y, m, 1), 9000.0, "Globex Payroll", "salary", "ach")
        lg.add(date(y, m, 4), -2300.0, "Maple Apartments", "housing", "ach")
        lg.add(date(y, m, 16), -1000.0, "Transfer to Savings", "internal_transfer", "transfer")
        everyday_spending(lg, y, m)
    return {
        "client_id": "C004",
        "name": "Daniel Okafor",
        "base_currency": "USD",
        "as_of": AS_OF,
        "accounts": [
            {"account_id": "C004-CHK", "type": "checking", "balance": 12000.00},
            {"account_id": "C004-SAV", "type": "savings", "balance": 30000.00},
        ],
        "holdings": [
            {"symbol": "VTI", "asset_class": "equity", "sector": "Diversified", "market_value": 120000.00},
            {"symbol": "VXUS", "asset_class": "equity", "sector": "Diversified", "market_value": 60000.00},
            {"symbol": "BND", "asset_class": "bond", "sector": "Diversified", "market_value": 70000.00},
            {"symbol": "MMF", "asset_class": "cash", "market_value": 10000.00},
        ],
        "liabilities": [],
        "transactions": lg.sorted_rows(),
    }


CRM_PROFILES = {
    "C001": {
        "client_id": "C001", "risk_tolerance": "moderate", "age": 41, "investment_horizon_years": 20,
        "goals": ["Retire at 60", "Children's university fund", "Holiday home"],
        "advisor": "J. Rivera", "last_review_date": "2025-11-20", "kyc_status": "verified",
        "notes": ["2026-08-02: Client mentioned paying a deposit on a holiday apartment abroad in August."],
    },
    "C002": {
        "client_id": "C002", "risk_tolerance": "conservative", "age": 68, "investment_horizon_years": 7,
        "goals": ["Stable retirement income", "Leave inheritance"],
        "advisor": "J. Rivera", "last_review_date": "2025-03-10", "kyc_status": "expired",
        "notes": ["Prefers phone contact.", "Son holds power of attorney (pending documentation)."],
    },
    "C003": {
        "client_id": "C003", "risk_tolerance": "balanced", "age": "thirty-four",
        "goals": ["Buy first home"], "advisor": "A. Tan", "kyc_status": "pending",
    },
    # C004 intentionally absent -> CRM returns 404.
}

EXPECTED = {
    "C001": ["large_transaction (18,500 wire)", "duplicate_transaction (BestElectro)",
             "spending_spike (gambling, Sep)", "statistical_outlier (dining 1,150)",
             "concentration (NVDA)", "suitability_mismatch (equity above moderate band)"],
    "C002": ["cash_structuring (3 withdrawals < 10,000)", "suitability_mismatch (equity vs conservative)",
             "low_liquidity", "negative_cashflow", "declining_cashflow", "kyc_not_current (expired)",
             "review_overdue"],
    "C003": ["data-quality warnings and rejected records", "suitability skipped (unrecognised risk tolerance)",
             "kyc_not_current (pending)", "portfolio checks skipped (no valid holdings)"],
    "C004": ["no findings; CRM 404 -> suitability/KYC checks skipped"],
}


def main() -> None:
    clients_dir = ROOT / "data" / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    for builder in (client_c001, client_c002, client_c003, client_c004):
        doc = builder()
        (clients_dir / f"{doc['client_id']}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {doc['client_id']}: {len(doc['transactions'])} transactions")
    crm_dir = ROOT / "data" / "crm"
    crm_dir.mkdir(parents=True, exist_ok=True)
    (crm_dir / "profiles.json").write_text(json.dumps(CRM_PROFILES, indent=2), encoding="utf-8")
    print("wrote CRM profiles")


if __name__ == "__main__":
    main()
