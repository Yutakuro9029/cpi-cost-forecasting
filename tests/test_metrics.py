import numpy as np
import pandas as pd
import pytest

from metrics import calc_mae, calc_mase, interval_metrics, mase_scale
from models import seasonal_naive_forecast

HORIZON = 12


def _split(series, horizon=HORIZON):
    return series.iloc[:-horizon], series.iloc[-horizon:]


# calc_mae

def test_mae_skips_nan_and_counts_points():
    idx = pd.date_range("2024-01-01", periods=4, freq="MS")
    actual = pd.Series([1.0, np.nan, 3.0, 5.0], index=idx)
    pred = pd.Series([2.0, 100.0, 3.0, 3.0], index=idx)
    mae, n_points = calc_mae(actual, pred)
    assert mae == pytest.approx(1.0)  # (1 + 0 + 2) / 3
    assert n_points == 3


def test_mae_is_nan_when_no_valid_points():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    mae, n_points = calc_mae(pd.Series([np.nan] * 3, index=idx), pd.Series([1.0] * 3, index=idx))
    assert np.isnan(mae)
    assert n_points == 0


# calc_mase (original Test 1 and Test 4)

def test_mase_is_nan_when_scale_is_zero(seasonal_series):
    train, test = _split(seasonal_series)
    pred = seasonal_naive_forecast(train, HORIZON)
    mae, n_points = calc_mae(test, pred)
    mase, _ = calc_mase(test, pred, train)
    assert mae == pytest.approx(0.0)
    assert np.isnan(mase)
    assert n_points == HORIZON


def test_mase_hand_calculated(seasonal_trend_series):
    # Train scale: every value is 1 above the same month last year, so scale = 1.
    # Seasonal naive misses each test month by 1, so MAE = 1 and MASE = 1.
    train, test = _split(seasonal_trend_series)
    pred = seasonal_naive_forecast(train, HORIZON)
    mae, _ = calc_mae(test, pred)
    mase, _ = calc_mase(test, pred, train)
    assert mae == pytest.approx(1.0)
    assert mase == pytest.approx(1.0)


# mase_scale

def test_mase_scale_hand_calculated(seasonal_trend_series):
    train, _ = _split(seasonal_trend_series)
    assert mase_scale(train) == pytest.approx(1.0)


def test_mase_scale_is_nan_when_zero(seasonal_series):
    train, _ = _split(seasonal_series)
    assert np.isnan(mase_scale(train))


def test_mase_scale_matches_calc_mase(noisy_seasonal_series):
    """Interval widths are scaled with mase_scale, so it must equal the scale inside calc_mase."""
    train, test = _split(noisy_seasonal_series)
    pred = seasonal_naive_forecast(train, HORIZON)
    mae, _ = calc_mae(test, pred)
    mase, _ = calc_mase(test, pred, train)
    assert mae / mase_scale(train) == pytest.approx(mase)


# interval_metrics

@pytest.fixture
def small_interval():
    idx = pd.date_range("2024-01-01", periods=4, freq="MS")
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    frame = pd.DataFrame({"mean": [1.0, 1.0, 1.0, 1.0],
                          "lower": [0.0, 0.0, 0.0, 0.0],
                          "upper": [2.0, 2.0, 2.0, 5.0]}, index=idx)
    return actual, frame


def test_interval_metrics_hand_calculated(small_interval):
    actual, frame = small_interval
    r = interval_metrics(actual, frame, scale=2.0)
    assert r["n_points"] == 4
    assert r["n_covered"] == 3  # 3.0 is above its upper bound of 2.0; boundary 2.0 counts as covered
    assert r["coverage"] == pytest.approx(0.75)
    assert r["mean_width"] == pytest.approx(2.75)
    assert r["scaled_width"] == pytest.approx(1.375)


def test_interval_metrics_skips_nan(small_interval):
    actual, frame = small_interval
    actual = actual.copy()
    actual.iloc[2] = np.nan  # the only uncovered point
    r = interval_metrics(actual, frame, scale=1.0)
    assert r["n_points"] == 3
    assert r["coverage"] == pytest.approx(1.0)


def test_interval_metrics_rejects_mismatched_index(small_interval):
    actual, frame = small_interval
    shifted = frame.copy()
    shifted.index = shifted.index + pd.DateOffset(months=1)
    with pytest.raises(AssertionError):
        interval_metrics(actual, shifted, scale=1.0)
