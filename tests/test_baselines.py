import numpy as np
import pytest

from metrics import calc_mae
from models import drift_forecast, naive_forecast, seasonal_naive_forecast

HORIZON = 12
BASELINES = [naive_forecast, seasonal_naive_forecast, drift_forecast]


def _split(series, horizon=HORIZON):
    return series.iloc[:-horizon], series.iloc[-horizon:]


# Original Test 2: naive error grows with the step, mean of 1..12 = 6.5
def test_naive_error_on_linear_trend(linear_series):
    train, test = _split(linear_series)
    mae, n_points = calc_mae(test, naive_forecast(train, HORIZON))
    assert mae == pytest.approx(6.5)
    assert n_points == HORIZON


def test_naive_uses_last_valid_value(linear_series):
    train, _ = _split(linear_series)
    train = train.copy()
    train.iloc[-1] = np.nan
    pred = naive_forecast(train, HORIZON)
    assert (pred == train.iloc[-2]).all()


def test_seasonal_naive_exact_on_repeating_pattern(seasonal_series):
    train, test = _split(seasonal_series)
    mae, _ = calc_mae(test, seasonal_naive_forecast(train, HORIZON))
    assert mae == pytest.approx(0.0)


# Original Test 3: a NaN month in the test set is skipped, not counted as an error
def test_seasonal_naive_with_nan_in_test(seasonal_series):
    series = seasonal_series.copy()
    series.loc["2024-10-01"] = np.nan
    train, test = _split(series)
    mae, n_points = calc_mae(test, seasonal_naive_forecast(train, HORIZON))
    assert mae == pytest.approx(0.0)
    assert n_points == HORIZON - 1


def test_seasonal_naive_matches_by_date_not_position(seasonal_series):
    """A missing month inside train must not shift which month is used one year later."""
    train, test = _split(seasonal_series)
    train = train.copy()
    train.loc["2023-03-01"] = np.nan
    pred = seasonal_naive_forecast(train, HORIZON)
    assert np.isnan(pred.loc["2024-03-01"])
    assert (pred.drop(pred.index[2]) == test.drop(test.index[2])).all()


def test_seasonal_naive_never_reads_past_train(seasonal_series):
    """Beyond 12 steps the value from one year earlier is not in train, so it must be NaN."""
    train, _ = _split(seasonal_series)
    pred = seasonal_naive_forecast(train, HORIZON + 1)
    assert pred.iloc[:HORIZON].notna().all()
    assert np.isnan(pred.iloc[-1])


# Original Test 5: drift on a perfect line has zero error
def test_drift_exact_on_linear_trend(linear_series):
    train, test = _split(linear_series)
    mae, n_points = calc_mae(test, drift_forecast(train, HORIZON))
    assert mae == pytest.approx(0.0)
    assert n_points == HORIZON


@pytest.mark.parametrize("model_fn", BASELINES, ids=lambda f: f.__name__)
def test_forecast_index_follows_train(linear_series, model_fn):
    train, test = _split(linear_series)
    pred = model_fn(train, HORIZON)
    assert pred.index.equals(test.index)
    assert len(pred) == HORIZON
