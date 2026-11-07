"""Shared synthetic data for the test suite. No test uses real FRED data."""
import numpy as np
import pandas as pd
import pytest


# Short series (2020-2024), used by baseline and metric tests

@pytest.fixture
def monthly_dates():
    return pd.date_range("2020-01-01", "2024-12-01", freq="MS")


@pytest.fixture
def linear_series(monthly_dates):
    """Slope of 1 per month: 1, 2, ..., N."""
    return pd.Series(np.arange(1, len(monthly_dates) + 1, dtype=float), index=monthly_dates)


@pytest.fixture
def seasonal_series(monthly_dates):
    """Value equals the month number, repeating every year (seasonal naive scale = 0)."""
    return pd.Series(monthly_dates.month.astype(float), index=monthly_dates)


@pytest.fixture
def seasonal_trend_series(monthly_dates):
    """Month number plus 1 per year: 2020 is 1..12, 2021 is 2..13, and so on."""
    values = monthly_dates.month + (monthly_dates.year - 2020)
    return pd.Series(values.astype(float), index=monthly_dates)


# Longer series (2018-2024), same data as the original statistical and ML unit tests

@pytest.fixture
def long_dates():
    return pd.date_range("2018-01-01", "2024-12-01", freq="MS")


@pytest.fixture
def linear_series_long(long_dates):
    return pd.Series(np.arange(1, len(long_dates) + 1, dtype=float), index=long_dates)


@pytest.fixture
def seasonal_trend_long(long_dates):
    values = long_dates.month + (long_dates.year - 2018)
    return pd.Series(values.astype(float), index=long_dates)


@pytest.fixture
def noisy_seasonal_series():
    """Trend + seasonality + random walk level (2012-2024), used by interval tests."""
    dates = pd.date_range("2012-01-01", "2024-12-01", freq="MS")
    rng = np.random.default_rng(0)
    values = (100 + 0.3 * np.arange(len(dates)) + 2 * np.sin(2 * np.pi * dates.month / 12)
              + np.cumsum(rng.normal(0, 0.5, len(dates))))
    return pd.Series(values, index=dates)


# Multi-series DataFrame shaped like the real CPI data

@pytest.fixture
def cpi_like_df():
    """Two series from 2010-01 to 2026-09 with an all-NaN row at 2025-10, like the FRED data."""
    dates = pd.date_range("2010-01-01", "2026-09-01", freq="MS")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "A": 100 + 0.3 * np.arange(len(dates)) + rng.normal(0, 1, len(dates)),
            "B": 200 + 0.5 * np.arange(len(dates)) + 3 * np.sin(2 * np.pi * dates.month / 12),
        },
        index=dates,
    )
    frame.loc["2025-10-01"] = np.nan
    return frame.asfreq("MS")


@pytest.fixture
def cutoffs():
    return ["2022-09-01", "2023-09-01", "2024-09-01", "2025-09-01"]
