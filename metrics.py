import numpy as np
import pandas as pd

# MAE and MASE Functions
def calc_mae(actual: pd.Series, pred: pd.Series):
    """Calculate MAE while skipping NaNs in actual, returning (mae, valid_points)."""
    mask = actual.notna() & pred.notna()
    n_points = mask.sum()
    if n_points == 0:
        return np.nan, 0
    mae_val = np.mean(np.abs(actual[mask] - pred[mask]))
    return mae_val, n_points

def calc_mase(actual: pd.Series, pred: pd.Series, train: pd.Series, season: int = 12):
    """Calculate MASE scaled by in-sample seasonal naive MAE on train."""
    mae_model, n_points = calc_mae(actual, pred)

    # Compute in-sample seasonal naive error on train
    train_dates = train.index[season:]
    train_target_dates = train_dates - pd.DateOffset(years=1)

    train_actual = train.loc[train_dates]
    train_s_naive = train.reindex(train_target_dates).values

    scale_mask = train_actual.notna() & pd.notna(train_s_naive)
    if scale_mask.sum() == 0:
        scale = np.nan
    else:
        scale = np.mean(np.abs(train_actual[scale_mask] - train_s_naive[scale_mask]))
    
    if np.isnan(scale) or scale == 0:
        return np.nan, n_points
        
    return mae_model / scale, n_points

# Metrics

def mase_scale(train: pd.Series, season: int = 12) -> float:
    """In-sample seasonal naive MAE, matched by date and skipping NaNs (same as calc_mase)."""
    dates = train.index[season:]
    actual = train.loc[dates]
    lagged = train.reindex(dates - pd.DateOffset(years=1)).values
    mask = actual.notna().values & pd.notna(lagged)
    if mask.sum() == 0:
        return np.nan
    scale = np.mean(np.abs(actual.values[mask] - lagged[mask]))
    return scale if scale > 0 else np.nan


def interval_metrics(actual: pd.Series, frame: pd.DataFrame, scale: float) -> dict:
    """Coverage and width on points where actual is observed."""
    assert actual.index.equals(frame.index), "Actual and interval indexes do not match"
    mask = actual.notna() & frame["lower"].notna() & frame["upper"].notna()
    n_points = int(mask.sum())
    if n_points == 0:
        return {"n_points": 0, "n_covered": 0, "coverage": np.nan,
                "mean_width": np.nan, "scaled_width": np.nan}
    a, lo, hi = actual[mask], frame["lower"][mask], frame["upper"][mask]
    n_covered = int(((a >= lo) & (a <= hi)).sum())
    mean_width = float((hi - lo).mean())
    return {"n_points": n_points, "n_covered": n_covered, "coverage": n_covered / n_points,
            "mean_width": mean_width, "scaled_width": mean_width / scale}