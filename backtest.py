import pandas as pd

from metrics import calc_mae, calc_mase, interval_metrics, mase_scale
from models import INTERVAL_MODELS

def make_splits(data: pd.Series, cutoffs: list, horizon: int = 12):
    """Yields (cutoff_date_str, train_series, test_series) for each cutoff date safely."""
    for cutoff in cutoffs:
        cutoff_dt = pd.to_datetime(cutoff)
        
        train = data.loc[:cutoff_dt]
        test_start = cutoff_dt + pd.DateOffset(months=1)
        test_end = cutoff_dt + pd.DateOffset(months=horizon)
        test = data.loc[test_start:test_end]
        
        # Safety assert: ensure test length matches horizon exactly (except when test extends into NaNs at the end)
        assert len(test) <= horizon, f"Test slice length ({len(test)}) exceeds horizon ({horizon})"
        
        yield cutoff, train, test

def run_backtest(df: pd.DataFrame, models: dict, cutoffs: list, horizon: int = 12) -> pd.DataFrame:
    """
    Executes rolling-origin backtest across series, models, and cutoff dates.
    
    """
    records = []
    
    for series_name in df.columns:
        series_data = df[series_name]
        
        for cutoff, train, test in make_splits(series_data, cutoffs, horizon):
            
            for model_name, model_fn in models.items():
                # Generate forecast using ONLY training data
                pred = model_fn(train, horizon)
                
                # Calculate evaluation metrics
                mae_val, n_pts = calc_mae(test, pred)
                mase_val, _ = calc_mase(test, pred, train)
                
                records.append({
                    "series": series_name,
                    "cutoff": cutoff,
                    "model": model_name,
                    "mae": mae_val,
                    "mase": mase_val,
                    "n_points": n_pts
                })
                
    return pd.DataFrame(records)

# 6. Interval backtest on real data (same cutoffs as Milestone 2)

def run_interval_backtest(df: pd.DataFrame, cutoffs: list, horizon: int = 12,
                             levels=(0.80, 0.95), interval_models=None) -> pd.DataFrame:
    interval_models = interval_models or INTERVAL_MODELS
    records = []
    for series_name in df.columns:
        for cutoff, train, test in make_splits(df[series_name], cutoffs, horizon):
            scale = mase_scale(train)
            for model_name, fn in interval_models.items():
                for lv in levels:
                    frame = fn(train, horizon, lv)
                    r = interval_metrics(test, frame.loc[test.index], scale)
                    records.append({"series": series_name, "cutoff": cutoff, "model": model_name,
                                    "level": lv, **r})
    return pd.DataFrame(records)