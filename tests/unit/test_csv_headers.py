from __future__ import annotations

import gzip

import pandas as pd
import pytest

from quant_metric_research.contracts import DataContractError
from quant_metric_research.io import read_as_of_dates, read_table


def _write_csv(tmp_path, suffix: str, contents: str):
    path = tmp_path / f"input{suffix}"
    encoded = contents.encode("utf-8")
    path.write_bytes(gzip.compress(encoded) if suffix.endswith(".gz") else encoded)
    return path


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
@pytest.mark.parametrize(
    "contents",
    [
        "date,symbol,adjusted_close,adjusted_close\n2025-01-02,A,100,200\n",
        "\ufeffdate,date,symbol\n2025-01-02,2025-01-03,A\n",
        '\n \t\n"date","date",symbol\n2025-01-02,2025-01-03,A\n',
        '"quoted,name","quoted,name"\n1,2\n',
        '"multiline\nname","multiline\nname"\n1,2\n',
        "date,,\n2025-01-02,1,2\n",
    ],
)
def test_read_table_rejects_duplicate_raw_csv_headers(
    tmp_path, suffix: str, contents: str
) -> None:
    path = _write_csv(tmp_path, suffix, contents)

    with pytest.raises(DataContractError, match="[Dd]uplicate.*columns"):
        read_table(path)


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_date_reader_rejects_duplicate_raw_date_headers(tmp_path, suffix) -> None:
    path = _write_csv(
        tmp_path,
        suffix,
        "as_of_date,as_of_date\n2025-01-02,2025-01-03\n",
    )

    with pytest.raises(DataContractError, match="[Dd]uplicate.*columns"):
        read_as_of_dates(path)


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
@pytest.mark.parametrize(
    ("contents", "expected_columns"),
    [
        (
            "adjusted_close,adjusted_close.1\n100,200\n",
            ["adjusted_close", "adjusted_close.1"],
        ),
        (
            '\ufeff\n"quoted,name","multiline\nname"\n100,200\n',
            ["quoted,name", "multiline\nname"],
        ),
        ("1,1.0\n100,200\n", ["1", "1.0"]),
    ],
)
def test_read_table_preserves_distinct_literal_csv_headers(
    tmp_path, suffix: str, contents: str, expected_columns: list[str]
) -> None:
    path = _write_csv(tmp_path, suffix, contents)

    result = read_table(path)

    assert result.columns.tolist() == expected_columns
    assert result.iloc[0].tolist() == [100, 200]


def test_csv_header_validation_does_not_change_parquet_loading(tmp_path) -> None:
    path = tmp_path / "input.parquet"
    expected = pd.DataFrame({"adjusted_close": [100.0], "adjusted_close.1": [200.0]})
    expected.to_parquet(path, index=False)

    pd.testing.assert_frame_equal(read_table(path), expected)
