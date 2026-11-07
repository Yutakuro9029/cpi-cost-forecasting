import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from models import (
    AUDIT_STATUSES,
    _fit_status,
    ets_forecast_audited,
    ets_interval,
    sarima_forecast_audited,
    sarima_interval,
)

HORIZON = 12
POINT_MODELS = [ets_forecast_audited, sarima_forecast_audited]
INTERVAL_FNS = [ets_interval, sarima_interval]


def _split(series, horizon=HORIZON):
    return series.iloc[:-horizon], series.iloc[-horizon:]


def _mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


# Point forecasts (original cell 24 tests)

def test_sarima_follows_linear_trend(linear_series_long):
    """The drift term (trend='c') must carry the slope forward instead of a flat line."""
    train, test = _split(linear_series_long)
    assert _mae(test, sarima_forecast_audited(train, HORIZON)) < 0.5


def test_ets_follows_seasonal_trend(seasonal_trend_long):
    train, test = _split(seasonal_trend_long)
    assert _mae(test, ets_forecast_audited(train, HORIZON)) < 0.5


@pytest.mark.parametrize("model_fn", POINT_MODELS, ids=lambda f: f.__name__)
def test_rejects_nan_in_train(seasonal_trend_long, model_fn):
    train, _ = _split(seasonal_trend_long)
    train = train.copy()
    train.iloc[5] = np.nan
    with pytest.raises(AssertionError):
        model_fn(train, HORIZON)


# Audit log

@pytest.mark.parametrize("model_fn", POINT_MODELS, ids=lambda f: f.__name__)
def test_audit_records_one_entry_per_fit(seasonal_trend_long, model_fn):
    train, _ = _split(seasonal_trend_long)
    audit = []
    model_fn(train, HORIZON, audit=audit)
    assert len(audit) == 1
    assert audit[0]["status"] in AUDIT_STATUSES
    assert audit[0]["train_end"] == train.index[-1]


def test_failed_fit_returns_nan_and_is_recorded():
    # ETS cannot initialise seasonals with fewer than two full years of data
    short = pd.Series(np.arange(1, 11, dtype=float),
                      index=pd.date_range("2020-01-01", periods=10, freq="MS"))
    audit = []
    pred = ets_forecast_audited(short, 3, audit=audit)
    assert pred.isna().all()
    assert len(pred) == 3
    assert audit[0]["status"] == "failed"
    assert "ValueError" in audit[0]["detail"]


def _caught(*categories):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for category in categories:
            warnings.warn("synthetic warning", category)
    return caught


@pytest.mark.parametrize(
    "retvals, categories, expected",
    [
        ({"converged": True}, (), "converged"),
        ({"converged": True}, (ConvergenceWarning,), "not_converged"),
        ({"converged": False}, (), "not_converged"),
        ({}, (), "unknown"),
        (None, (), "unknown"),
    ],
    ids=["flag_true", "warning_overrides_flag", "flag_false", "missing_key", "no_retvals"],
)
def test_fit_status_cases(retvals, categories, expected):
    status, _ = _fit_status(SimpleNamespace(mle_retvals=retvals), _caught(*categories))
    assert status == expected


def test_fit_status_keeps_other_warning_names():
    status, detail = _fit_status(SimpleNamespace(mle_retvals={"converged": True}), _caught(UserWarning))
    assert status == "converged"
    assert "UserWarning" in detail


# Prediction intervals (cell 32 tests)

@pytest.mark.parametrize("interval_fn", INTERVAL_FNS, ids=lambda f: f.__name__)
def test_interval_basic_properties(noisy_seasonal_series, interval_fn):
    train, test = _split(noisy_seasonal_series)
    f80 = interval_fn(train, HORIZON, 0.80)
    f95 = interval_fn(train, HORIZON, 0.95)
    width80 = f80["upper"] - f80["lower"]
    width95 = f95["upper"] - f95["lower"]

    assert f80.index.equals(test.index)
    assert list(f80.columns) == ["mean", "lower", "upper"]
    assert ((f80["lower"] <= f80["mean"]) & (f80["mean"] <= f80["upper"])).all()
    assert (width95 > width80).all()
    assert width80.iloc[-1] > width80.iloc[0]


@pytest.mark.parametrize(
    "interval_fn, point_fn",
    [(ets_interval, ets_forecast_audited), (sarima_interval, sarima_forecast_audited)],
    ids=["ETS", "SARIMA"],
)
def test_interval_mean_matches_point_forecast(noisy_seasonal_series, interval_fn, point_fn):
    """Guards against the interval and point versions drifting to different model specs."""
    train, _ = _split(noisy_seasonal_series)
    np.testing.assert_allclose(interval_fn(train, HORIZON)["mean"].values,
                               point_fn(train, HORIZON).values)
