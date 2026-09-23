"""Invented panel enrichment only: no market data, fitting, or file writes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_metric_research import (
    FEATURE_BUNDLES,
    PanelConfig,
    __version__,
    build_factor_panel,
    build_point_in_time_panel,
)

FACTOR_NAMES = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")


def invented_inputs():
    dates = pd.bdate_range("2020-01-01", periods=280)
    position = np.arange(len(dates))
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(0.001 * position + 0.01 * np.sin(position / (number + 3))),
                }
            )
            for number, symbol in enumerate(("BENCH", "A", "B", "C"))
        ],
        ignore_index=True,
    )
    memberships = pd.DataFrame(
        [
            {
                "universe_id": "INVENTED",
                "symbol": symbol,
                "effective_from": dates[0],
                "effective_to": None,
                "source": "invented example",
            }
            for symbol in ("A", "B", "C", "MISSING")
        ]
    )
    config = PanelConfig(
        dataset_version="synthetic-factor-panel-v1",
        universe_id="INVENTED",
        benchmark_symbol="BENCH",
    )
    return prices, memberships, [dates[index] for index in (20, 63, 252, 279)], config


def main() -> None:
    prices, memberships, dates, config = invented_inputs()
    legacy = build_point_in_time_panel(
        prices, memberships, as_of_dates=dates, config=config
    )
    enriched = build_factor_panel(
        prices, memberships, as_of_dates=dates, config=config, factor_names=FACTOR_NAMES
    )
    pd.testing.assert_frame_equal(enriched.loc[:, legacy.columns], legacy)
    print(
        json.dumps(
            {
                "claim_scope": "synthetic_factor_panel_example",
                "package_version": __version__,
                "legacy_panel_preserved": True,
                "training_performed": False,
                "network_accessed": False,
                "final_outcomes_evaluated": False,
                "row_count": len(enriched),
                "candidate_feature_counts": {
                    name: len(bundle.feature_columns)
                    for name, bundle in FEATURE_BUNDLES.items()
                },
                "factor_status_counts": {
                    name: {
                        status: int(count)
                        for status, count in enriched[f"{name}_status"]
                        .value_counts()
                        .items()
                    }
                    for name in FACTOR_NAMES
                },
            },
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
