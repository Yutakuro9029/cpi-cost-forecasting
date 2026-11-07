import numpy as np
import pytest

from models import ml_hgb_forecast

HORIZON = 12


def _split(series, horizon=HORIZON):
    return series.iloc[:-horizon], series.iloc[-horizon:]


def test_ml_learns_seasonal_trend_pattern(seasonal_trend_long):
    """Original strict ML test: the month feature must let the trees learn the January drop."""
    train, test = _split(seasonal_trend_long)
    pred = ml_hgb_forecast(train, HORIZON)
    assert float(np.mean(np.abs(test - pred))) < 0.1


def test_ml_forecast_index_follows_train(seasonal_trend_long):
    train, test = _split(seasonal_trend_long)
    assert ml_hgb_forecast(train, HORIZON).index.equals(test.index)


def test_ml_is_deterministic(seasonal_trend_long):
    """random_state is fixed, so two runs must give identical forecasts."""
    train, _ = _split(seasonal_trend_long)
    first = ml_hgb_forecast(train, HORIZON)
    second = ml_hgb_forecast(train, HORIZON)
    np.testing.assert_array_equal(first.values, second.values)


def test_ml_rejects_nan_in_train(seasonal_trend_long):
    train, _ = _split(seasonal_trend_long)
    train = train.copy()
    train.iloc[5] = np.nan
    with pytest.raises(AssertionError):
        ml_hgb_forecast(train, HORIZON)
