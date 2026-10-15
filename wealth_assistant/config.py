"""Runtime configuration. Defaults come from the environment (``WA_*``) and the CLI can override them."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AnomalyThresholds:
    # Single outflow larger than this share of median monthly income.
    large_tx_income_ratio: float = 0.5
    large_tx_min_amount: float = 2_000.0
    # Modified z-score (Iglewicz & Hoaglin): |z| > 3.5 is the usual outlier cut-off.
    robust_z_threshold: float = 3.5
    robust_z_min_samples: int = 8
    # Same merchant + same amount within N days.
    duplicate_window_days: int = 3
    # Repeated cash withdrawals just under a reporting threshold.
    cash_reporting_threshold: float = 10_000.0
    cash_structuring_band: float = 0.8   # withdrawals within [80%, 100%) of the threshold
    cash_structuring_count: int = 3
    cash_structuring_window_days: int = 7
    # Last month's category spend compared with the trailing average.
    spike_ratio: float = 2.5
    spike_min_amount: float = 500.0


@dataclass(frozen=True)
class RiskThresholds:
    max_single_position_pct: float = 20.0
    min_liquidity_months: float = 3.0
    max_debt_to_income_pct: float = 36.0
    # Equity allocation bands by risk tolerance (inclusive low, inclusive high).
    equity_bands: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "conservative": (15.0, 40.0),
        "moderate": (40.0, 70.0),
        "aggressive": (65.0, 100.0),
    })


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    memory_dir: Path = PROJECT_ROOT / ".memory"
    log_dir: Path = PROJECT_ROOT / "logs"
    output_dir: Path = PROJECT_ROOT / "output"

    # Mock CRM behaviour (to demonstrate resilience).
    crm_failure_rate: float = 0.0
    crm_outage: bool = False
    crm_latency_s: float = 0.05
    crm_timeout_s: float = 2.0
    crm_max_attempts: int = 3
    crm_cache_max_age_days: int = 90
    random_seed: int | None = None

    # Optional LLM narrative.
    use_llm: bool = False
    llm_model: str = "claude-opus-5-5"
    llm_timeout_s: float = 60.0

    anomaly: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    risk: RiskThresholds = field(default_factory=RiskThresholds)

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        """Read ``WA_*`` environment variables; explicit overrides (CLI flags) win."""
        env: dict[str, object] = {
            "crm_failure_rate": _env_float("WA_CRM_FAILURE_RATE", cls.crm_failure_rate),
            "crm_outage": _env_bool("WA_CRM_OUTAGE", cls.crm_outage),
            "crm_latency_s": _env_float("WA_CRM_LATENCY_S", cls.crm_latency_s),
            "crm_timeout_s": _env_float("WA_CRM_TIMEOUT_S", cls.crm_timeout_s),
            "use_llm": _env_bool("WA_USE_LLM", cls.use_llm),
            "llm_model": os.environ.get("WA_LLM_MODEL", cls.llm_model),
            "llm_timeout_s": _env_float("WA_LLM_TIMEOUT_S", cls.llm_timeout_s),
        }
        env.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env)  # type: ignore[arg-type]
