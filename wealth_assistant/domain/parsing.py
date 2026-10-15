"""Turn untrusted raw JSON into validated domain objects.

Policy: be strict at the *record* level and lenient at the *document* level. A single
bad transaction is dropped and reported in ``DataQuality``. It never fails the whole
run. Only a document that is unusable as a whole (wrong shape, wrong client) raises.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Callable, TypeVar

from ..core.errors import DataValidationError
from .models import (
    Account, ClientProfile, ClientSnapshot, DataQuality, Holding, Liability, RiskTolerance, Transaction,
)

T = TypeVar("T")

KNOWN_ASSET_CLASSES = {"equity", "bond", "cash", "real_estate", "alternative"}
KNOWN_CHANNELS = {"card", "transfer", "cash", "ach", "wire"}


def parse_amount(value: Any) -> float:
    """Accept numbers and numeric strings ("1,200.50"). Reject bools, NaN and infinity."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"not a number: {value!r}")
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return round(number, 2)


def parse_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError(f"date must be an ISO string, got {value!r}")
    return date.fromisoformat(value.strip()[:10])


def parse_client_document(raw: Any, expected_client_id: str) -> ClientSnapshot:
    if not isinstance(raw, dict):
        raise DataValidationError("client document must be a JSON object")
    doc_client = raw.get("client_id")
    if doc_client != expected_client_id:
        raise DataValidationError(
            f"client_id mismatch: requested {expected_client_id!r}, document contains {doc_client!r}"
        )

    dq = DataQuality()
    name = raw.get("name") or "Unknown client"
    if not raw.get("name"):
        dq.warn("client name missing")

    base_currency = str(raw.get("base_currency") or "").upper()
    if not base_currency:
        base_currency = "USD"
        dq.warn("base_currency missing; assuming USD")

    transactions = _parse_transactions(raw.get("transactions"), base_currency, dq)

    as_of: date | None = None
    try:
        as_of = parse_date(raw.get("as_of"))
    except (ValueError, TypeError):
        dq.warn("as_of missing or invalid; using latest transaction date")
    if as_of is None:
        as_of = max((t.date for t in transactions), default=date.today())

    # Future-dated transactions are dropped only once ``as_of`` is known.
    future = [t for t in transactions if t.date > as_of]
    if future:
        dq.warn(f"{len(future)} transaction(s) dated after as_of {as_of} were excluded: "
                f"{', '.join(t.id for t in future)}")
        dq.rejected_records += len(future)
        transactions = [t for t in transactions if t.date <= as_of]

    return ClientSnapshot(
        client_id=expected_client_id,
        name=str(name),
        as_of=as_of,
        base_currency=base_currency,
        accounts=_parse_list(raw.get("accounts"), "account", _parse_account, dq),
        holdings=_parse_list(raw.get("holdings"), "holding", _parse_holding, dq),
        liabilities=_parse_list(raw.get("liabilities"), "liability", _parse_liability, dq),
        transactions=sorted(transactions, key=lambda t: (t.date, t.id)),
        data_quality=dq,
    )


def parse_profile(raw: Any, expected_client_id: str) -> tuple[ClientProfile, list[str]]:
    """Parse a CRM payload. Returns the profile plus non-fatal warnings."""
    if not isinstance(raw, dict):
        raise DataValidationError("CRM profile must be a JSON object")
    if raw.get("client_id") != expected_client_id:
        raise DataValidationError(f"CRM returned profile for {raw.get('client_id')!r}, expected {expected_client_id!r}")

    warnings: list[str] = []
    risk: RiskTolerance | None = None
    try:
        risk = RiskTolerance(str(raw.get("risk_tolerance", "")).lower())
    except ValueError:
        warnings.append(f"CRM risk_tolerance {raw.get('risk_tolerance')!r} is missing or unrecognised; "
                        "suitability checks will be skipped")

    def _opt_int(key: str) -> int | None:
        value = raw.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            warnings.append(f"CRM field {key!r} is not an integer: {value!r}")
            return None

    profile = ClientProfile(
        client_id=expected_client_id,
        risk_tolerance=risk,
        age=_opt_int("age"),
        investment_horizon_years=_opt_int("investment_horizon_years"),
        goals=tuple(str(g) for g in raw.get("goals") or ()),
        advisor=raw.get("advisor"),
        last_review_date=raw.get("last_review_date"),
        kyc_status=raw.get("kyc_status"),
        notes=tuple(str(n) for n in raw.get("notes") or ()),
    )
    return profile, warnings


# ---------------------------------------------------------------------- helpers

def _parse_list(items: Any, label: str, parser: Callable[[dict[str, Any]], T], dq: DataQuality) -> list[T]:
    if items is None:
        dq.warn(f"no {label}s provided")
        return []
    if not isinstance(items, list):
        dq.warn(f"'{label}s' is not a list; ignored")
        return []
    parsed: list[T] = []
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError("record is not an object")
            parsed.append(parser(item))
        except (ValueError, TypeError, KeyError) as exc:
            dq.rejected_records += 1
            dq.warn(f"{label} #{index} rejected: {exc}")
    return parsed


def _parse_account(item: dict[str, Any]) -> Account:
    return Account(
        account_id=str(item.get("account_id") or "unknown"),
        type=str(item.get("type") or "unknown").lower(),
        balance=parse_amount(item["balance"]),
    )


def _parse_holding(item: dict[str, Any]) -> Holding:
    value = parse_amount(item["market_value"])
    if value < 0:
        raise ValueError(f"negative market_value {value}")
    asset_class = str(item.get("asset_class") or "").lower()
    return Holding(
        symbol=str(item.get("symbol") or "UNKNOWN"),
        asset_class=asset_class if asset_class in KNOWN_ASSET_CLASSES else "alternative",
        market_value=value,
        sector=item.get("sector"),
    )


def _parse_liability(item: dict[str, Any]) -> Liability:
    return Liability(
        type=str(item.get("type") or "unknown"),
        balance=abs(parse_amount(item["balance"])),
        monthly_payment=abs(parse_amount(item.get("monthly_payment", 0))),
    )


def _parse_transactions(items: Any, base_currency: str, dq: DataQuality) -> list[Transaction]:
    if not isinstance(items, list):
        dq.warn("transactions missing or not a list; transaction analysis will be limited")
        return []

    seen: set[str] = set()
    out: list[Transaction] = []
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError("record is not an object")
            tx_id = str(item.get("id") or f"ROW-{index}")
            if not item.get("id"):
                dq.warn(f"transaction #{index} has no id; assigned {tx_id}")
            if tx_id in seen:
                raise ValueError(f"duplicate transaction id {tx_id}")
            currency = str(item.get("currency") or base_currency).upper()
            if currency != base_currency:
                # No FX source is available, so mixing currencies would corrupt every ratio.
                raise ValueError(f"currency {currency} differs from base {base_currency} (no FX conversion)")
            amount = parse_amount(item.get("amount"))
            if amount == 0:
                raise ValueError("zero amount")
            channel = str(item.get("channel") or "card").lower()
            tx = Transaction(
                id=tx_id,
                date=parse_date(item.get("date")),
                amount=amount,
                description=str(item.get("description") or ""),
                category=str(item.get("category") or "uncategorized").lower(),
                merchant=str(item.get("merchant") or item.get("description") or "unknown"),
                channel=channel if channel in KNOWN_CHANNELS else "card",
            )
            seen.add(tx_id)
            out.append(tx)
        except (ValueError, TypeError) as exc:
            dq.rejected_records += 1
            dq.warn(f"transaction #{index} ({item.get('id') if isinstance(item, dict) else '?'}) rejected: {exc}")
    return out
