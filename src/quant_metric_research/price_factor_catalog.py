"""Opt-in price representations, not exact French factors or Qlib replicas."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class PriceFactorSpec:
    """Versioned formula and its inclusive supplied-calendar source interval."""

    name: str
    formula_version: str
    formula: str
    start_lag_sessions: int
    end_lag_sessions: int
    required_price_count: int
    source_references: tuple[str, ...]
    required_input: str
    availability_assumption: str
    adjustment_assumption: str


_FRENCH_DAILY_MOMENTUM = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "Data_Library/det_mom_factor_daily.html"
)
_FRENCH_DAILY_REVERSAL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "Data_Library/det_st_rev_factor_daily.html"
)
_QLIB_LOADER = "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"


def _spec(
    name: str,
    formula: str,
    start_lag: int,
    end_lag: int,
    references: tuple[str, ...],
) -> PriceFactorSpec:
    return PriceFactorSpec(
        name=name,
        formula_version="1",
        formula=formula,
        start_lag_sessions=start_lag,
        end_lag_sessions=end_lag,
        required_price_count=start_lag - end_lag + 1,
        source_references=references,
        required_input="adjusted_close",
        availability_assumption=(
            "Calculated after the as_of_date close from supplied observations. "
            "Historical publication and revision availability are not verified."
        ),
        adjustment_assumption=(
            "Caller supplies consistently adjusted closes. Corporate actions, "
            "revisions and provider provenance are not verified by this calculator."
        ),
    )


PRICE_FACTOR_CATALOG = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            _spec(
                "return_21s",
                "P[t] / P[t-21] - 1",
                21,
                0,
                (_FRENCH_DAILY_REVERSAL, _QLIB_LOADER),
            ),
            _spec(
                "momentum_252s_skip_21s",
                "P[t-21] / P[t-252] - 1",
                252,
                21,
                (_FRENCH_DAILY_MOMENTUM,),
            ),
            _spec(
                "ma_distance_63s",
                "P[t] / mean(P[t-62], ..., P[t]) - 1",
                62,
                0,
                (_QLIB_LOADER,),
            ),
        )
    }
)
