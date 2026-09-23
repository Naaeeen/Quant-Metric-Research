"""Offline pure-factor example; invented prices, no training or market evidence."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from quant_metric_research import __version__, compute_price_factors


def _timestamp(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def main() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=253)
    position = np.arange(len(calendar))
    prices = pd.Series(
        100.0 * np.exp(0.001 * position + 0.02 * np.sin(position / 10)),
        index=calendar,
        name="INVENTED",
    )
    result = compute_price_factors(
        prices,
        calendar=calendar,
        as_of_date=calendar[-1],
        factor_names=("return_21s", "momentum_252s_skip_21s", "ma_distance_63s"),
    )
    print(
        json.dumps(
            {
                "claim_scope": "synthetic_factor_example",
                "package_version": __version__,
                "network_accessed": False,
                "training_performed": False,
                "final_outcomes_evaluated": False,
                "result": asdict(result),
            },
            default=_timestamp,
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
