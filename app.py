import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import make_splits
from data_utils import cpi_series_dict, fill_single_month_gaps, load_cpi, load_source_info
from models import drift_forecast, ets_interval, sarima_interval

DATA_PATH = os.path.join("data", "cpi_data.csv")
RESULTS_DIR = "results"
HORIZON = 12
HISTORY_MONTHS = 48
INTERVALS = {"ETS": ets_interval, "SARIMA": sarima_interval}

# Mid-tone colors that stay readable on both the light and the dark theme
COLORS = {
    "actual": "#2CA02C",
    "forecast": "#1F77B4",
    "band": "rgba(31, 119, 180, 0.25)",
    "drift": "#7F7F7F",
    "filled": "#FF7F0E",
}
LEVELS = {"80%": 0.80, "95%": 0.95}
BASELINES = ["Naive", "SeasonalNaive", "Drift"]
MODEL_ORDER = BASELINES + ["ETS", "SARIMA", "ML_HGB"]
MODEL_COLORS = {
    "Naive": "#D62728", "SeasonalNaive": "#9467BD", "Drift": "#7F7F7F",
    "ETS": "#FF7F0E", "SARIMA": "#1F77B4", "ML_HGB": "#2CA02C",
}

st.set_page_config(page_title="CPI Cost Forecast", layout="wide")


# Data loading (cached: runs once per session, not on every widget change)

@st.cache_data
def load_data():
    raw = load_cpi(DATA_PATH)
    filled, filled_mask = fill_single_month_gaps(raw)
    return raw, filled, filled_mask


@st.cache_data
def load_results():
    m2 = pd.read_csv(os.path.join(RESULTS_DIR, "milestone2_backtest_results.csv"))
    m3 = pd.read_csv(os.path.join(RESULTS_DIR, "milestone3_interval_results.csv"))
    return m2, m3


@st.cache_data
def forecast(series_name: str, model_name: str, level: float) -> pd.DataFrame:
    """Fit on the gap-filled series up to the latest month and forecast the next 12 months."""
    _, filled, _ = load_data()
    frame = INTERVALS[model_name](filled[series_name], HORIZON, level)
    frame["drift"] = drift_forecast(filled[series_name], HORIZON)
    return frame


def backtest_coverage(m3, series_name, model_name, level):
    """Pooled backtest coverage: total covered points / total evaluated points."""
    rows = m3[(m3["series"] == series_name) & (m3["model"] == model_name)
              & np.isclose(m3["level"], level)]
    n_points = int(rows["n_points"].sum())
    if n_points == 0:
        return np.nan, 0
    return rows["n_covered"].sum() / n_points, n_points


def backtest_mase(m2, series_name, model_name):
    return m2[(m2["series"] == series_name) & (m2["model"] == model_name)]["mase"].agg(["mean", "max"])


# Tab 2 helpers

def mase_summary(m2, series_name):
    """Mean, best and worst MASE per model, plus wins against the best baseline per cutoff."""
    rows = m2[m2["series"] == series_name]
    best_baseline = rows[rows["model"].isin(BASELINES)].groupby("cutoff")["mase"].min()
    rows = rows.assign(beat=rows["mase"] < rows["cutoff"].map(best_baseline))
    table = rows.groupby("model").agg(
        mean_mase=("mase", "mean"), best_mase=("mase", "min"),
        worst_mase=("mase", "max"), wins=("beat", "sum"), cutoffs=("cutoff", "nunique"),
    )
    # A baseline cannot meaningfully "beat the best baseline", so label it instead
    table["wins"] = [
        "baseline" if model in BASELINES else f"{int(w)} of {int(n)}"
        for model, w, n in zip(table.index, table["wins"], table["cutoffs"])
    ]
    table = table.drop(columns="cutoffs")
    order = [m for m in MODEL_ORDER if m in table.index]
    return table.loc[order].sort_values("mean_mase")


def mase_by_cutoff_chart(m2, series_name):
    rows = m2[m2["series"] == series_name]
    fig = go.Figure()
    for model in MODEL_ORDER:
        part = rows[rows["model"] == model].sort_values("cutoff")
        if part.empty:
            continue
        dash = "dash" if model in BASELINES else "solid"
        fig.add_trace(go.Scatter(x=part["cutoff"], y=part["mase"], mode="lines+markers", name=model,
                                 line=dict(color=MODEL_COLORS[model], dash=dash)))
    fig.add_hline(y=1.0, line=dict(color="#7F7F7F", width=1, dash="dot"),
                  annotation_text="MASE = 1", annotation_position="top left")
    fig.update_layout(title=f"{series_name}: MASE by cutoff (lower is better)",
                      xaxis_title="Cutoff (forecast starts the next month)", yaxis_title="MASE",
                      xaxis=dict(type="category"), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=60, b=20))
    return fig


def coverage_summary(m3, series_name):
    """Pooled coverage and mean scaled width per model and level for one series."""
    rows = m3[m3["series"] == series_name]
    out = []
    for (model, level), g in rows.groupby(["model", "level"]):
        out.append({"model": model, "interval": f"{level:.0%}",
                    "coverage": g["n_covered"].sum() / g["n_points"].sum(),
                    "gap_vs_stated": g["n_covered"].sum() / g["n_points"].sum() - level,
                    "scaled_width": g["scaled_width"].mean(),
                    "points": int(g["n_points"].sum())})
    return pd.DataFrame(out)


@st.cache_data
def backtest_path(series_name: str, model_name: str, level: float, cutoff: str) -> pd.DataFrame:
    """Re-create one backtest forecast so the user can see where the errors came from."""
    raw, _, _ = load_data()
    [(_, train, test)] = list(make_splits(raw[series_name], [cutoff], HORIZON))
    frame = INTERVALS[model_name](train, HORIZON, level).loc[test.index]
    frame["drift"] = drift_forecast(train, HORIZON).loc[test.index]
    frame["actual"] = test
    return frame


def backtest_path_chart(frame, series_name, model_name, level_label, cutoff):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLORS["band"],
                             name=f"{level_label} prediction interval"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["actual"], mode="lines+markers", name="Actual",
                             line=dict(color=COLORS["actual"], width=2.5)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["mean"], mode="lines",
                             name=f"{model_name} forecast", line=dict(color=COLORS["forecast"], width=2.5)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["drift"], mode="lines",
                             name="Drift (reference)", line=dict(color=COLORS["drift"], dash="dash")))
    fig.update_layout(title=f"{series_name}: backtest from cutoff {cutoff}",
                      yaxis_title="Index (1982-1984 = 100)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60, b=20))
    return fig


def forecast_chart(history, filled_points, frame, series_name, model_name, level_label):
    fig = go.Figure()

    # Interval band: upper edge, then lower edge filled up to it
    fig.add_trace(go.Scatter(x=frame.index, y=frame["upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLORS["band"],
                             name=f"{level_label} prediction interval"))

    fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="Actual",
                             line=dict(color=COLORS["actual"], width=2.5)))
    if not filled_points.empty:
        fig.add_trace(go.Scatter(x=filled_points.index, y=filled_points, mode="markers",
                                 name="Interpolated (no source data)",
                                 marker=dict(symbol="x", size=11, color=COLORS["filled"])))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["mean"], mode="lines",
                             name=f"{model_name} forecast", line=dict(color=COLORS["forecast"], width=2.5)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["drift"], mode="lines",
                             name="Drift (reference)", line=dict(color=COLORS["drift"], dash="dash")))

    fig.update_layout(title=f"{series_name}: {model_name} forecast with {level_label} interval",
                      yaxis_title="Index (1982-1984 = 100)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60, b=20))
    return fig


# Sidebar

raw, filled, filled_mask = load_data()
m2, m3 = load_results()

st.sidebar.header("Settings")
series_name = st.sidebar.selectbox("Series", list(raw.columns))
model_name = st.sidebar.radio("Model", list(INTERVALS))
level_label = st.sidebar.radio("Interval", list(LEVELS))
level = LEVELS[level_label]

st.title("CPI Cost Forecast")
st.caption("12-month forecasts of US CPI categories (BLS, not seasonally adjusted), "
           "used as a public proxy for cost indices.")

tab_forecast, tab_backtest, tab_about = st.tabs(["Forecast", "Backtest", "About"])


# Tab 1: Forecast

with tab_forecast:
    frame = forecast(series_name, model_name, level)
    coverage, n_points = backtest_coverage(m3, series_name, model_name, level)
    mase = backtest_mase(m2, series_name, model_name)

    first, last = frame.index[0], frame.index[-1]
    st.subheader(f"Forecast for {first:%b %Y} to {last:%b %Y}")

    # How far to trust the range: the backtest answer, shown next to the chart
    message = (f"In backtests, the {model_name} {level_label} interval for {series_name} "
               f"covered **{coverage:.0%}** of actual values ({n_points} months).")
    if np.isnan(coverage):
        st.info("No backtest coverage is available for this selection.")
    elif coverage < level - 0.10:
        st.warning(message + f" That is well below the stated {level_label}, "
                   "so treat this range as narrower than the real uncertainty.")
    else:
        st.info(message)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Backtest coverage ({level_label})", f"{coverage:.0%}")
    col2.metric("Backtest mean MASE", f"{mase['mean']:.2f}")
    col3.metric("Backtest worst MASE", f"{mase['max']:.2f}")

    history = raw[series_name].iloc[-HISTORY_MONTHS:]
    gap_dates = filled_mask[series_name].iloc[-HISTORY_MONTHS:]
    filled_points = filled[series_name].iloc[-HISTORY_MONTHS:][gap_dates]
    st.plotly_chart(forecast_chart(history, filled_points, frame, series_name, model_name, level_label),
                    width="stretch")

    if not filled_points.empty:
        months = ", ".join(d.strftime("%b %Y") for d in filled_points.index)
        st.caption(f"{months} has no value in the source data. It was filled by linear "
                   "interpolation for fitting only, and is shown as an orange X marker.")

    table = frame.rename(columns={"mean": "forecast", "drift": "drift_reference"}).round(2)
    table.index = table.index.strftime("%Y-%m")
    table.index.name = "month"
    st.dataframe(table, width="stretch")
    # Callable data: the CSV is only generated when the button is clicked, not on every rerun
    st.download_button("Download forecast (CSV)", data=lambda: table.to_csv().encode("utf-8"),
                       file_name=f"forecast_{series_name}_{model_name}_{int(level * 100)}.csv",
                       mime="text/csv")


with tab_backtest:
    st.subheader(f"How the models performed on past data: {series_name}")
    st.caption("Rolling-origin backtest with four cutoffs and a 12-month horizon. "
               "MASE below 1 means the error was smaller than an in-sample seasonal naive "
               "benchmark. A model 'wins' a cutoff when it beats the best of the three baselines "
               "at that cutoff (chosen with hindsight, so this is a strict test).")

    st.markdown("#### Point forecast accuracy")
    summary = mase_summary(m2, series_name)
    st.dataframe(
        summary,
        column_config={
            "mean_mase": st.column_config.NumberColumn("Mean MASE", format="%.2f"),
            "best_mase": st.column_config.NumberColumn("Best MASE", format="%.2f"),
            "worst_mase": st.column_config.NumberColumn("Worst MASE", format="%.2f"),
            "wins": st.column_config.TextColumn("Wins vs best baseline"),
        },
        width="stretch",
    )
    st.plotly_chart(mase_by_cutoff_chart(m2, series_name), width="stretch")

    st.markdown("#### Prediction interval coverage")
    coverage = coverage_summary(m3, series_name)
    coverage_view = coverage.assign(coverage=coverage["coverage"] * 100,
                                    gap_vs_stated=coverage["gap_vs_stated"] * 100)
    st.dataframe(
        coverage_view,
        column_config={
            "model": "Model", "interval": "Interval",
            "coverage": st.column_config.NumberColumn("Coverage", format="%.0f%%"),
            "gap_vs_stated": st.column_config.NumberColumn("Gap vs stated (pts)", format="%+.0f"),
            "scaled_width": st.column_config.NumberColumn("Scaled width", format="%.2f"),
            "points": st.column_config.NumberColumn("Points"),
        },
        hide_index=True, width="stretch",
    )
    st.caption("Coverage pools all evaluated months across cutoffs. Scaled width is the average "
               "interval width divided by the MASE scale, so it can be compared across series. "
               "Each value comes from only 47 correlated months, so small differences are not meaningful.")

    st.markdown("#### Inspect one backtest")
    cutoff = st.selectbox("Cutoff", sorted(m2["cutoff"].unique()), key="backtest_cutoff")
    path = backtest_path(series_name, model_name, level, cutoff)
    errors = (path["mean"] - path["actual"]).dropna()
    inside = ((path["actual"] >= path["lower"]) & (path["actual"] <= path["upper"]))[path["actual"].notna()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average error (forecast - actual)", f"{errors.mean():+.2f}",
              help="Positive means the forecast was too high on average; negative means too low.")
    c2.metric("Mean absolute error", f"{errors.abs().mean():.2f}")
    c3.metric(f"Months inside {level_label} interval", f"{int(inside.sum())} of {len(inside)}")
    st.plotly_chart(backtest_path_chart(path, series_name, model_name, level_label, cutoff),
                    width="stretch")

with tab_about:
    filled_months = sorted({d.strftime("%b %Y") for d in filled_mask.index[filled_mask.any(axis=1)]})
    series_rows = "\n".join(
        f"| {name} | `{series_id}` | [BLS series](https://data.bls.gov/timeseries/{series_id}) |"
        for series_id, name in cpi_series_dict.items()
    )

    source_info = load_source_info(DATA_PATH)
    retrieved = source_info["retrieved"] if source_info else "unknown"

    st.subheader("About this dashboard")
    st.markdown(f"""
This dashboard forecasts five US Consumer Price Index (CPI) categories 12 months ahead and shows
how well each model and prediction interval performed on past data. CPI is used as a public proxy
for cost indices; it is not company cost data.

#### Data
CPI for All Urban Consumers (CPI-U), U.S. city average, monthly, not seasonally adjusted,
from {raw.index.min():%b %Y} to {raw.index.max():%b %Y}.
Source: U.S. Bureau of Labor Statistics, retrieved through the BLS Public Data API on {retrieved}.

BLS.gov cannot vouch for the data or analyses derived from these data after the data have been
retrieved from BLS.gov.

| Category | Series ID | Source |
|---|---|---|
{series_rows}

#### Models
- **ETS(A,A,A):** exponential smoothing with additive error, trend and 12-month seasonality.
- **SARIMA (0,1,1)(0,1,1,12) with drift:** the "airline" model.
- **Drift (reference):** last value plus the average monthly change over the training period.
  It is shown because it had the lowest backtest error of all models for Food and Medical care.

Both ETS and SARIMA are fitted on all available data and forecast the next {HORIZON} months.
Model specifications were fixed in advance and are the same as in the backtest.

#### How to read the forecast
- The shaded band is the model's prediction interval. It assumes the model is correct, so it is
  often too narrow. The **backtest coverage** shown above each chart is the share of past months
  where the actual value fell inside the band.
- A yellow warning appears when backtest coverage is more than 10 percentage points below the
  stated level. This threshold is a display rule, not a statistical test.
- **Average error** in the Backtest tab shows direction: positive means the forecast was too high.

#### Backtest method
Rolling origin with four cutoffs, each forecasting the next {HORIZON} months from data up to the cutoff.
Accuracy is measured with MAE and MASE (scaled by the in-sample seasonal naive error).
Months with no actual value are excluded.

#### Limitations
- **Missing month:** {", ".join(filled_months) if filled_months else "none"} has no value in the source data.
  For forecasting only, it is filled by linear interpolation, which ignores seasonality.
  Backtest results use the original data without this fill.
- **Small sample:** four cutoffs and 47 evaluated months per series; months within one forecast are correlated,
  so small differences between models are not meaningful.
- **Univariate models:** no external drivers (for example oil prices), so sudden shocks cannot be anticipated.
- **Intervals ignore parameter uncertainty,** which is one reason coverage is below the stated level.
- **Proxy data:** CPI categories are not the costs of a specific company.
""")
