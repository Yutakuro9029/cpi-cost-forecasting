"""Interval calibration on synthetic series where the model form is appropriate.

Slow (about 800 model fits). Run with: pytest -m slow
"""
import numpy as np
import pandas as pd
import pytest

from metrics import interval_metrics
from models import INTERVAL_MODELS

LEVELS = (0.80, 0.95)
# Accepted pooled coverage. Slightly below nominal is expected: statsmodels intervals
# do not include parameter estimation uncertainty.
BOUNDS = {0.80: (0.72, 0.88), 0.95: (0.90, 0.99)}


def run_calibration(n_series=100, horizon=12, seed=42):
    """Pooled coverage (total covered / total points) per model and level."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=180, freq="MS")
    season = 3 * np.sin(2 * np.pi * dates.month / 12)
    totals = {(m, lv): [0, 0] for m in INTERVAL_MODELS for lv in LEVELS}
    for _ in range(n_series):
        level = np.cumsum(rng.normal(0, 0.5, len(dates)))
        y = pd.Series(100 + 0.2 * np.arange(len(dates)) + season + level
                      + rng.normal(0, 0.5, len(dates)), index=dates)
        train, test = y.iloc[:-horizon], y.iloc[-horizon:]
        for name, fn in INTERVAL_MODELS.items():
            for lv in LEVELS:
                r = interval_metrics(test, fn(train, horizon, lv), 1.0)
                totals[(name, lv)][0] += r["n_covered"]
                totals[(name, lv)][1] += r["n_points"]
    return {key: covered / points for key, (covered, points) in totals.items()}


@pytest.fixture(scope="module")
def calibration_coverage():
    # Module scope: the expensive run happens once and is shared by all cases below
    return run_calibration()


@pytest.mark.slow
@pytest.mark.parametrize("model", list(INTERVAL_MODELS))
@pytest.mark.parametrize("level", LEVELS)
def test_pooled_coverage_close_to_nominal(calibration_coverage, model, level):
    low, high = BOUNDS[level]
    coverage = calibration_coverage[(model, level)]
    assert low <= coverage <= high, f"{model} {level:.0%}: coverage {coverage:.3f}"
