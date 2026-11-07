import json
import os
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
YEARS_PER_QUERY = 20  # BLS API v2 limit for registered users

# CPI-U, U.S. city average, not seasonally adjusted (BLS series IDs)
cpi_series_dict = {
    "CUUR0000SA0": "All_Items",
    "CUUR0000SAF1": "Food",
    "CUUR0000SA0E": "Energy",
    "CUUR0000SAA": "Apparel",
    "CUUR0000SAM": "Medical_Care",
}


def parse_bls_response(payload: dict, series_map: dict) -> pd.DataFrame:
    """Turn one BLS API JSON response into a DataFrame (Date index, one column per series).

    Only monthly periods M01-M12 are kept (M13 is an annual average). Values that are not
    numbers, such as '-' for a month with no data, become NaN.
    """
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: {payload.get('status')} {payload.get('message')}")

    columns = {}
    for item in payload["Results"]["series"]:
        name = series_map[item["seriesID"]]
        rows = {}
        for obs in item["data"]:
            period = obs["period"]
            if not (period.startswith("M") and period != "M13"):
                continue
            month_start = pd.Timestamp(year=int(obs["year"]), month=int(period[1:]), day=1)
            rows[month_start] = pd.to_numeric(obs["value"], errors="coerce")
        columns[name] = pd.Series(rows, dtype=float)

    missing = set(series_map.values()) - set(columns)
    if missing:
        raise RuntimeError(f"BLS API returned no data for: {sorted(missing)}")

    out = pd.DataFrame(columns)[list(series_map.values())].sort_index()
    out.index.name = "Date"
    return out


def fetch_and_cache(path, series=cpi_series_dict, start_year=2010, end_year=None):
    """Download the series from the BLS API (v2, needs BLS_API_KEY) and cache them as a CSV.

    Also writes '<path>.source.json' with the retrieval date, which the BLS API terms
    ask users to cite.
    """
    load_dotenv()
    api_key = os.getenv("BLS_API_KEY")
    if not api_key:
        raise RuntimeError("Set BLS_API_KEY in your .env file (register at https://data.bls.gov/registrationEngine/)")

    end_year = end_year or date.today().year
    frames = []
    for first in range(start_year, end_year + 1, YEARS_PER_QUERY):
        last = min(first + YEARS_PER_QUERY - 1, end_year)
        payload = {"seriesid": list(series), "startyear": str(first), "endyear": str(last),
                   "registrationkey": api_key}
        response = requests.post(BLS_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        frames.append(parse_bls_response(response.json(), series))

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_csv(path)

    source = {
        "source": "U.S. Bureau of Labor Statistics (BLS) Public Data API v2",
        "series": series,
        "retrieved": date.today().isoformat(),
    }
    with open(f"{path}.source.json", "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2)
    print(f"Data saved to {path} (retrieved {source['retrieved']})")


def load_source_info(path):
    """Return the retrieval metadata written by fetch_and_cache, or None if it is missing."""
    try:
        with open(f"{path}.source.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_cpi(path):
    """Return a continuous monthly DataFrame.

    Months missing in the source stay in the index as NaN rows.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    raw.index.name = "Date"
    if raw.index.has_duplicates:
        raise ValueError("Duplicate dates found in the CPI data")
    # asfreq inserts any absent month as an all-NaN row and sets freq='MS'
    out = raw.sort_index().asfreq("MS")
    assert out.index.is_monotonic_increasing
    return out


def report_missing(frame):
    """Print which months are NaN. Nothing is dropped or filled here."""
    nan_rows = frame[frame.isna().any(axis=1)]
    print(f"Range: {frame.index.min():%Y-%m} to {frame.index.max():%Y-%m} "
          f"({len(frame)} months)")
    print("NaN count per column:")
    print(frame.isna().sum())
    if nan_rows.empty:
        print("No missing months.")
    else:
        print("Months with missing values:", list(nan_rows.index.strftime("%Y-%m")))


def fill_single_month_gaps(frame: pd.DataFrame):
    """Linearly interpolate isolated one-month gaps, for forecasting only.

    Returns (filled_frame, filled_mask), where filled_mask is True at every value
    that was filled. Raises ValueError if a gap is longer than one month or sits at
    the start or end of the data, because those cannot be filled from both sides.
    """
    if frame.index.freqstr != "MS":
        raise ValueError("Index must be continuous monthly data with freq='MS' (use load_cpi)")

    missing = frame.isna()
    if not missing.to_numpy().any():
        return frame.copy(), missing

    edge = missing.iloc[[0, -1]]
    if edge.to_numpy().any():
        cols = list(edge.columns[edge.any()])
        raise ValueError(f"Missing value at the first or last month in: {cols}")

    # A month that is missing while the previous month is also missing means a gap of 2+ months
    consecutive = missing & missing.shift(1, fill_value=False)
    if consecutive.to_numpy().any():
        months = sorted({d.strftime("%Y-%m") for d in consecutive.index[consecutive.any(axis=1)]})
        raise ValueError(f"Gap longer than one month ending at: {months}")

    # 'linear' treats every row as one step, which is correct because the index has
    # exactly one row per month. method='time' would weight by days (30 vs 31).
    filled = frame.interpolate(method="linear", limit_area="inside")
    assert not filled.isna().to_numpy().any()
    return filled, missing
