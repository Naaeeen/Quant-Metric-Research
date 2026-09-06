from __future__ import annotations

import pandas as pd
import pytest

from quant_metric_research.contracts import DataContractError
from quant_metric_research.io import read_as_of_dates


@pytest.mark.parametrize(
    "value", ["2025-01-02 12:00:00", "2025-01-02T00:00:00Z", 0, True, "invalid"]
)
def test_date_file_rejects_non_daily_dates(tmp_path, value: object) -> None:
    path = tmp_path / "dates.csv"
    pd.DataFrame({"as_of_date": [value]}).to_csv(path, index=False)

    with pytest.raises(DataContractError, match="as_of_dates"):
        read_as_of_dates(path)


def test_date_reader_rejects_duplicate_frame_columns(monkeypatch) -> None:
    frame = pd.DataFrame(
        [["2025-01-02", "2025-01-03"]], columns=["as_of_date", "as_of_date"]
    )
    monkeypatch.setattr("quant_metric_research.io.read_table", lambda path: frame)

    with pytest.raises(DataContractError, match="[Dd]uplicate.*columns"):
        read_as_of_dates("unused.csv")


def test_date_reader_preserves_order(tmp_path) -> None:
    path = tmp_path / "dates.csv"
    pd.DataFrame({"as_of_date": ["2025-01-03", "2025-01-02"]}).to_csv(path, index=False)

    assert read_as_of_dates(path) == (
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-02"),
    )
