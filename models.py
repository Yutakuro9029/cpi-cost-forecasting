import numpy as np
import pandas as pd
import warnings

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Baseline Forecasting Algorithms

def naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last non-NaN observed value from training data."""
    valid_train = train.dropna()
    if len(valid_train) == 0:
        last_val = np.nan
    else:
        last_val = valid_train.iloc[-1]

    future_dates = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
    return pd.Series([last_val] * horizon, index=future_dates)

def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 12) -> pd.Series:
    """Forecast value at month t as equal to value at month (t - season) in train."""
    future_dates = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
    forecast_vals = []

    for d in future_dates:
        target_date = d - pd.DateOffset(years=1)
        val = train.get(target_date, np.nan)
        forecast_vals.append(val)

    return pd.Series(forecast_vals, index=future_dates)

def drift_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """
    Naive forecast with Drift (Linear trend based on historical average growth rate).
    Forecast_h = Last_Value + h * ((Last_Value - First_Value) / (N - 1))
    Calculated using ONLY training data to avoid leakage.
    """
    valid_train = train.dropna()
    if len(valid_train) < 2:
        return naive_forecast(train, horizon)
        
    first_val = valid_train.iloc[0]
    last_val = valid_train.iloc[-1]
    n_obs = len(valid_train)
    
    # Calculate historical average slope per step
    drift_slope = (last_val - first_val) / (n_obs - 1)
    
    future_dates = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    steps = np.arange(1, horizon + 1)
    forecast_vals = last_val + (steps * drift_slope)
    
    return pd.Series(forecast_vals, index=future_dates)

# 1. Model fitting (same specs as cell 24; checked by a test below)

def fit_ets(train: pd.Series):
    """ETS(A,A,A), identical spec to ets_forecast_audited."""
    assert train.index.freqstr == "MS", "Train index must have freq='MS'"
    assert not train.isna().any(), "Train data contains NaN"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is audited in cell 28
        return ETSModel(train, error="add", trend="add", seasonal="add",
                        seasonal_periods=12).fit(disp=False)


def fit_sarima(train: pd.Series):
    """SARIMA(0,1,1)(0,1,1,12) with drift, identical spec to sarima_forecast_audited."""
    assert train.index.freqstr == "MS", "Train index must have freq='MS'"
    assert not train.isna().any(), "Train data contains NaN"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend="c",
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


# 2. Prediction intervals: (train, horizon, level) -> DataFrame[mean, lower, upper]

def ets_interval(train: pd.Series, horizon: int, level: float = 0.80) -> pd.DataFrame:
    res = fit_ets(train)
    start = len(train)
    frame = res.get_prediction(start=start, end=start + horizon - 1).summary_frame(alpha=1 - level)
    out = frame[["mean", "pi_lower", "pi_upper"]].copy()
    out.columns = ["mean", "lower", "upper"]
    out.index = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return out


def sarima_interval(train: pd.Series, horizon: int, level: float = 0.80) -> pd.DataFrame:
    res = fit_sarima(train)
    # For state space forecasts these bounds include observation noise (prediction interval)
    frame = res.get_forecast(steps=horizon).summary_frame(alpha=1 - level)
    out = frame[["mean", "mean_ci_lower", "mean_ci_upper"]].copy()
    out.columns = ["mean", "lower", "upper"]
    out.index = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return out


INTERVAL_MODELS = {"ETS": ets_interval, "SARIMA": sarima_interval}

AUDIT_STATUSES = ["converged", "not_converged", "unknown", "failed"]


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")


def _record_fit(audit, train, status, detail=""):
    """Append one fit record to the audit list. audit=None means do not record."""
    if audit is None:
        return
    audit.append({"series": train.name, "train_end": train.index[-1], "status": status, "detail": detail})

def sarima_forecast_audited(train: pd.Series, horizon: int, audit=None) -> pd.Series:
    """Airline-style SARIMA (0,1,1)(0,1,1,12) with drift. Spec fixed in advance."""
    assert train.index.freqstr == "MS", "Train index must have freq='MS'"
    assert not train.isna().any(), "Train data contains NaN"
    future_dates = _future_index(train, horizon)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(
                train,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
        status, detail = _fit_status(res, caught)
        _record_fit(audit, train, status, detail)
        pred = res.forecast(steps=horizon)
        pred.index = future_dates
        return pred
    except Exception as err:
        _record_fit(audit, train, "failed", f"{type(err).__name__}: {err}")
        return pd.Series([np.nan] * horizon, index=future_dates)


def ets_forecast_audited(train: pd.Series, horizon: int, audit=None) -> pd.Series:
    """ETS(A,A,A). Spec fixed in advance."""
    assert train.index.freqstr == "MS", "Train index must have freq='MS'"
    assert not train.isna().any(), "Train data contains NaN"
    future_dates = _future_index(train, horizon)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12)
            res = model.fit(disp=False)
        status, detail = _fit_status(res, caught)
        _record_fit(audit, train, status, detail)
        pred = res.forecast(steps=horizon)
        pred.index = future_dates
        return pred
    except Exception as err:
        _record_fit(audit, train, "failed", f"{type(err).__name__}: {err}")
        return pd.Series([np.nan] * horizon, index=future_dates)

def _fit_status(res, caught):
    """Decide convergence from BOTH the optimizer flag and any ConvergenceWarning.

    A missing 'converged' key is reported as 'unknown', never as converged.
    """
    retvals = getattr(res, "mle_retvals", None)
    flag = retvals.get("converged") if isinstance(retvals, dict) else None
    conv_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    other = sorted({w.category.__name__ for w in caught} - {"ConvergenceWarning"})
    detail = ", ".join(other)
    if flag is False or conv_warnings:
        return "not_converged", detail
    if flag is None:
        return "unknown", detail
    return "converged", detail

def ml_hgb_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """
    ML Model using HistGradientBoostingRegressor on MoM differences.
    Adjusted min_samples_leaf=5, max_iter=200 to capture seasonal splits on small sample sizes.
    """
    assert train.index.freqstr == 'MS', "Train index must have freq='MS'"
    assert not train.isna().any(), "Train set contains NaNs"
    
    # 1. Transform level to MoM difference
    diff_train = train.diff().dropna()
    
    # 2. Build Feature Matrix
    df_feat = pd.DataFrame({"target": diff_train})
    df_feat["lag_1"] = df_feat["target"].shift(1)
    df_feat["lag_2"] = df_feat["target"].shift(2)
    df_feat["lag_3"] = df_feat["target"].shift(3)
    df_feat["lag_12"] = df_feat["target"].shift(12)
    df_feat["rolling_mean_3"] = df_feat["target"].shift(1).rolling(3).mean()
    df_feat["month"] = df_feat.index.month
    
    df_train_clean = df_feat.dropna()
    
    X_train = df_train_clean.drop(columns=["target"])
    y_train = df_train_clean["target"]
    
    # 3. Fit Model with tuned min_samples_leaf
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Recursive Multi-step Forecasting
    history_diff = list(diff_train.values)
    history_dates = list(diff_train.index)
    
    pred_levels = []
    last_level = train.iloc[-1]
    
    future_dates = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    current_level = last_level
    
    for f_date in future_dates:
        s_diff = pd.Series(history_diff, index=history_dates)
        
        lag_1 = s_diff.iloc[-1]
        lag_2 = s_diff.iloc[-2]
        lag_3 = s_diff.iloc[-3]
        lag_12 = s_diff.iloc[-12] if len(s_diff) >= 12 else s_diff.iloc[-1]
        roll_3 = s_diff.iloc[-3:].mean()
        month_val = f_date.month
        
        X_test_step = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_12": lag_12,
            "rolling_mean_3": roll_3,
            "month": month_val
        }])
        
        pred_diff_step = model.predict(X_test_step)[0]
        
        history_diff.append(pred_diff_step)
        history_dates.append(f_date)
        
        current_level = current_level + pred_diff_step
        pred_levels.append(current_level)
        
    return pd.Series(pred_levels, index=future_dates)