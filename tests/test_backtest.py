import numpy as np
import pandas as pd
import pytest

from backtest import make_splits, run_backtest, run_interval_backtest
from models import drift_forecast, naive_forecast, seasonal_naive_forecast

HORIZON = 12
BASELINE_MODELS = {
    "Naive": naive_forecast,
    "SeasonalNaive": seasonal_naive_forecast,
    "Drift": drift_forecast,
}


# make_splits

def test_splits_align_with_cutoffs(cpi_like_df, cutoffs):
    for cutoff, train, test in make_splits(cpi_like_df["A"], cutoffs, HORIZON):
        cutoff_dt = pd.Timestamp(cutoff)
        assert train.index[-1] == cutoff_dt
        assert test.index[0] == cutoff_dt + pd.DateOffset(months=1)
        assert len(test) == HORIZON


def test_train_and_test_do_not_overlap(cpi_like_df, cutoffs):
    for _, train, test in make_splits(cpi_like_df["A"], cutoffs, HORIZON):
        assert train.index.intersection(test.index).empty
        assert train.index.max() < test.index.min()


def test_split_near_end_of_data_is_shorter(cpi_like_df):
    # Data ends 2026-09, so a 2026-03 cutoff leaves only 6 test months
    [(_, _, test)] = list(make_splits(cpi_like_df["A"], ["2026-03-01"], HORIZON))
    assert len(test) == 6


# run_backtest

def test_backtest_shape_and_columns(cpi_like_df, cutoffs):
    results = run_backtest(cpi_like_df, BASELINE_MODELS, cutoffs, HORIZON)
    expected_rows = cpi_like_df.shape[1] * len(cutoffs) * len(BASELINE_MODELS)
    assert len(results) == expected_rows
    assert list(results.columns) == ["series", "cutoff", "model", "mae", "mase", "n_points"]


def test_backtest_skips_missing_month(cpi_like_df, cutoffs):
    """2025-10 is NaN, so only the last cutoff loses one evaluated point (12, 12, 12, 11)."""
    results = run_backtest(cpi_like_df, BASELINE_MODELS, cutoffs, HORIZON)
    points = results.groupby("cutoff")["n_points"].unique()
    for cutoff in cutoffs[:-1]:
        assert list(points[cutoff]) == [12]
    assert list(points[cutoffs[-1]]) == [11]


def test_models_only_see_train(cpi_like_df, cutoffs):
    """Spy model: records the last date it was given; it must equal the cutoff every time."""
    seen = []

    def spy_model(train, horizon):
        seen.append(train.index[-1])
        return naive_forecast(train, horizon)

    run_backtest(cpi_like_df, {"Spy": spy_model}, cutoffs, HORIZON)
    expected = [pd.Timestamp(c) for c in cutoffs] * cpi_like_df.shape[1]
    assert seen == expected


# run_interval_backtest (uses a fake interval model so no statistical fit is needed)

def _wide_interval(train, horizon, level):
    """Always covers the actual value: coverage must be exactly 1."""
    idx = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    last = train.iloc[-1]
    return pd.DataFrame({"mean": last, "lower": last - 1e6, "upper": last + 1e6}, index=idx)


def _point_interval(train, horizon, level):
    """Zero-width interval at the last value: on a trending series it never covers."""
    idx = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    last = train.iloc[-1]
    return pd.DataFrame({"mean": last, "lower": last, "upper": last}, index=idx)


def test_interval_backtest_counts_and_coverage(cpi_like_df, cutoffs):
    results = run_interval_backtest(
        cpi_like_df, cutoffs, HORIZON, levels=(0.80, 0.95),
        interval_models={"Wide": _wide_interval, "Point": _point_interval},
    )
    assert len(results) == cpi_like_df.shape[1] * len(cutoffs) * 2 * 2
    per_model = results.groupby("model")[["n_covered", "n_points"]].sum()
    total_points = cpi_like_df.shape[1] * 2 * (12 * 3 + 11)  # 2 levels, 47 points per series
    assert (per_model["n_points"] == total_points).all()
    assert per_model.loc["Wide", "n_covered"] == total_points
    assert per_model.loc["Point", "n_covered"] == 0
    assert np.isfinite(results["scaled_width"]).all()
