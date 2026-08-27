from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

DEFAULT_FEATURE_COLUMNS = (
    "trailing_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "benchmark_correlation",
    "beta",
    "capm_alpha",
    "information_ratio",
    "historical_var_5pct",
)


@dataclass(frozen=True)
class PanelConfig:
    dataset_version: str
    universe_id: str
    benchmark_symbol: str
    lookback_sessions: int = 252
    min_observations: int = 126
    target_horizon_sessions: int = 20
    entry_lag_sessions: int = 1
    annualization_sessions: int = 252
    annual_risk_free_rate: float = 0.0
    feature_columns: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_FEATURE_COLUMNS
    )

    def __post_init__(self) -> None:
        for field_name in (
            "lookback_sessions",
            "min_observations",
            "target_horizon_sessions",
            "annualization_sessions",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")

        lag = self.entry_lag_sessions
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ValueError("entry_lag_sessions must be a non-negative integer.")
        if self.min_observations > self.lookback_sessions:
            raise ValueError("min_observations cannot exceed lookback_sessions.")
        if (
            not isfinite(float(self.annual_risk_free_rate))
            or self.annual_risk_free_rate <= -1.0
        ):
            raise ValueError(
                "annual_risk_free_rate must be finite and greater than -1."
            )

        dataset_version = str(self.dataset_version).strip()
        universe_id = str(self.universe_id).strip()
        benchmark_symbol = str(self.benchmark_symbol).upper().strip()
        if not dataset_version or not universe_id or not benchmark_symbol:
            raise ValueError(
                "dataset_version, universe_id, and benchmark_symbol are required."
            )

        feature_columns = tuple(self.feature_columns)
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns must be non-empty and unique.")
        unsupported = sorted(set(feature_columns) - set(DEFAULT_FEATURE_COLUMNS))
        if unsupported:
            raise ValueError(f"Unsupported feature columns: {', '.join(unsupported)}")

        object.__setattr__(self, "dataset_version", dataset_version)
        object.__setattr__(self, "universe_id", universe_id)
        object.__setattr__(self, "benchmark_symbol", benchmark_symbol)
        object.__setattr__(self, "feature_columns", feature_columns)
