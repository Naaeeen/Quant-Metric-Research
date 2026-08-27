from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture()
def market_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-02", periods=15)
    price_rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        price_rows.extend(
            [
                {
                    "date": date,
                    "symbol": "AAA",
                    "adjusted_close": 100.0 * (1.02**index),
                },
                {
                    "date": date,
                    "symbol": "BBB",
                    "adjusted_close": 100.0 * (0.995**index),
                },
                {
                    "date": date,
                    "symbol": "BENCH",
                    "adjusted_close": 100.0 * (1.01**index),
                },
            ]
        )

    memberships = pd.DataFrame(
        [
            {
                "universe_id": "TEST",
                "symbol": "AAA",
                "effective_from": dates[0],
                "effective_to": None,
                "source": "history",
            },
            {
                "universe_id": "TEST",
                "symbol": "BBB",
                "effective_from": dates[8],
                "effective_to": None,
                "source": "history",
            },
        ]
    )
    prices = pd.DataFrame(price_rows)
    return prices, memberships, dates
