"""Small builders for domain objects used across tests."""

from __future__ import annotations

from datetime import date

from wealth_assistant.domain.models import (
    Account, ClientProfile, ClientSnapshot, Holding, Liability, RiskTolerance, Transaction,
)

_counter = iter(range(1, 1_000_000))


def tx(d: date | str, amount: float, merchant: str = "Shop", category: str = "shopping",
       channel: str = "card", tx_id: str | None = None) -> Transaction:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return Transaction(id=tx_id or f"T{next(_counter)}", date=d, amount=amount, description=merchant,
                       category=category, merchant=merchant, channel=channel)


def monthly_salary(months: int = 6, amount: float = 10_000.0, start: date = date(2026, 4, 1)) -> list[Transaction]:
    out = []
    for i in range(months):
        m = (start.month - 1 + i) % 12 + 1
        y = start.year + (start.month - 1 + i) // 12
        out.append(tx(date(y, m, 1), amount, "Payroll", "salary", "ach"))
    return out


def snapshot(transactions: list[Transaction] | None = None, *, holdings: list[Holding] | None = None,
             accounts: list[Account] | None = None, liabilities: list[Liability] | None = None,
             profile: ClientProfile | None = None, as_of: date = date(2026, 9, 30)) -> ClientSnapshot:
    return ClientSnapshot(
        client_id="TEST", name="Test Client", as_of=as_of, base_currency="USD",
        accounts=accounts if accounts is not None else [Account("A1", "checking", 20_000.0)],
        holdings=holdings or [], liabilities=liabilities or [], transactions=transactions or [],
        profile=profile,
    )


def profile(risk: RiskTolerance | None = RiskTolerance.MODERATE, **kw) -> ClientProfile:
    return ClientProfile(client_id="TEST", risk_tolerance=risk, **kw)


