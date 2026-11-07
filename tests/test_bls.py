import numpy as np
import pandas as pd
import pytest

from data_utils import load_cpi, parse_bls_response

SERIES = {"CUUR0000SA0": "All_Items", "CUUR0000SAM": "Medical_Care"}


def _obs(year, period, value):
    return {"year": str(year), "period": period, "periodName": "", "value": value, "footnotes": [{}]}


def _payload(series_data, status="REQUEST_SUCCEEDED"):
    return {"status": status, "message": [],
            "Results": {"series": [{"seriesID": sid, "data": data} for sid, data in series_data.items()]}}


def test_parses_monthly_values_in_date_order():
    # BLS returns newest first; the result must be sorted oldest first
    payload = _payload({
        "CUUR0000SA0": [_obs(2025, "M02", "320.5"), _obs(2025, "M01", "319.1")],
        "CUUR0000SAM": [_obs(2025, "M02", "580.2"), _obs(2025, "M01", "579.0")],
    })
    out = parse_bls_response(payload, SERIES)
    assert list(out.columns) == ["All_Items", "Medical_Care"]
    assert list(out.index) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]
    assert out.loc["2025-02-01", "All_Items"] == pytest.approx(320.5)


def test_annual_average_period_is_dropped():
    payload = _payload({
        "CUUR0000SA0": [_obs(2025, "M13", "321.0"), _obs(2025, "M01", "319.1")],
        "CUUR0000SAM": [_obs(2025, "M01", "579.0")],
    })
    assert len(parse_bls_response(payload, SERIES)) == 1


def test_non_numeric_value_becomes_nan():
    payload = _payload({
        "CUUR0000SA0": [_obs(2025, "M10", "-"), _obs(2025, "M09", "324.0")],
        "CUUR0000SAM": [_obs(2025, "M10", "-"), _obs(2025, "M09", "582.0")],
    })
    out = parse_bls_response(payload, SERIES)
    assert np.isnan(out.loc["2025-10-01", "All_Items"])


def test_absent_month_becomes_nan_after_load(tmp_path):
    # A month missing from the response entirely must reappear as NaN when loaded
    payload = _payload({
        "CUUR0000SA0": [_obs(2025, "M11", "325.0"), _obs(2025, "M09", "324.0")],
        "CUUR0000SAM": [_obs(2025, "M11", "583.0"), _obs(2025, "M09", "582.0")],
    })
    path = tmp_path / "cpi.csv"
    parse_bls_response(payload, SERIES).to_csv(path)
    loaded = load_cpi(path)
    assert len(loaded) == 3
    assert loaded.loc["2025-10-01"].isna().all()


def test_failed_request_raises():
    with pytest.raises(RuntimeError, match="failed"):
        parse_bls_response(_payload({}, status="REQUEST_NOT_PROCESSED"), SERIES)


def test_missing_series_raises():
    payload = _payload({"CUUR0000SA0": [_obs(2025, "M01", "319.1")]})
    with pytest.raises(RuntimeError, match="Medical_Care"):
        parse_bls_response(payload, SERIES)
