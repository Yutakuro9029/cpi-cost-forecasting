import pandas as pd
import numpy as np
import pytest

from data_utils import fill_single_month_gaps, load_cpi, report_missing


def _write_csv(path, dates, values):
    pd.DataFrame({"Date": dates, "All_Items": values}).to_csv(path, index=False)
    return path


def test_missing_month_is_inserted_as_nan(tmp_path):
    # 2024-03 is absent from the file entirely
    path = _write_csv(tmp_path / "cpi.csv",
                      ["2024-01-01", "2024-02-01", "2024-04-01"], [1.0, 2.0, 4.0])
    frame = load_cpi(path)
    assert len(frame) == 4
    assert pd.isna(frame.loc["2024-03-01", "All_Items"])
    assert frame.index.freqstr == "MS"


def test_empty_cell_stays_nan(tmp_path):
    path = tmp_path / "cpi.csv"
    path.write_text("Date,All_Items\n2024-01-01,1.0\n2024-02-01,\n2024-03-01,3.0\n")
    frame = load_cpi(path)
    assert len(frame) == 3
    assert pd.isna(frame.loc["2024-02-01", "All_Items"])


def test_unsorted_dates_are_sorted(tmp_path):
    path = _write_csv(tmp_path / "cpi.csv",
                      ["2024-03-01", "2024-01-01", "2024-02-01"], [3.0, 1.0, 2.0])
    frame = load_cpi(path)
    assert frame["All_Items"].tolist() == [1.0, 2.0, 3.0]


def test_duplicate_dates_raise(tmp_path):
    path = _write_csv(tmp_path / "cpi.csv",
                      ["2024-01-01", "2024-01-01", "2024-02-01"], [1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="Duplicate"):
        load_cpi(path)


def test_report_missing_lists_nan_months(tmp_path, capsys):
    path = _write_csv(tmp_path / "cpi.csv",
                      ["2024-01-01", "2024-02-01", "2024-04-01"], [1.0, 2.0, 4.0])
    report_missing(load_cpi(path))
    assert "2024-03" in capsys.readouterr().out

def _monthly(values):
    idx = pd.date_range("2025-07-01", periods=len(values), freq="MS")
    return pd.DataFrame({"A": values}, index=idx)


def test_single_gap_is_midpoint():
    filled, mask = fill_single_month_gaps(_monthly([5.0, 10.0, np.nan, 20.0, 25.0]))
    assert filled["A"].iloc[2] == pytest.approx(15.0)
    assert mask["A"].tolist() == [False, False, True, False, False]


def test_no_gap_returns_unchanged_copy():
    frame = _monthly([1.0, 2.0, 3.0])
    filled, mask = fill_single_month_gaps(frame)
    pd.testing.assert_frame_equal(filled, frame)
    assert not mask.to_numpy().any()
    assert filled is not frame


def test_two_month_gap_raises():
    with pytest.raises(ValueError, match="longer than one month"):
        fill_single_month_gaps(_monthly([1.0, np.nan, np.nan, 4.0]))


@pytest.mark.parametrize("values", [[np.nan, 2.0, 3.0], [1.0, 2.0, np.nan]], ids=["first", "last"])
def test_gap_at_edge_raises(values):
    with pytest.raises(ValueError, match="first or last"):
        fill_single_month_gaps(_monthly(values))


def test_original_frame_is_not_modified():
    frame = _monthly([1.0, np.nan, 3.0])
    fill_single_month_gaps(frame)
    assert np.isnan(frame["A"].iloc[1])


def test_rejects_non_monthly_index():
    idx = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-04-01"])
    frame = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=idx)
    with pytest.raises(ValueError, match="freq"):
        fill_single_month_gaps(frame)
