"""Tests for ml_fw.data.loaders.omni()."""

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from ml_fw.data import omni


def make_synthetic_omni(year, n_rows=24):
    """Create a small synthetic OMNI2 DataFrame (n_rows hourly records for a year)."""
    data = {
        "year": [year] * n_rows,
        "doy": list(range(1, n_rows + 1)),
        "hour": list(range(n_rows)) % 24,
        "bartels_rotation": np.full(n_rows, 2500.0),
        "imf_sc_id": np.full(n_rows, 71.0),
        "plasma_sc_id": np.full(n_rows, 71.0),
        "n_imf_obs": np.full(n_rows, 60.0),
        "n_plasma_obs": np.full(n_rows, 60.0),
        "b_mag_avg": np.random.uniform(4, 6, n_rows),
        "b_vec_mag": np.random.uniform(3, 8, n_rows),
        "b_lat_gse": np.random.uniform(-30, 30, n_rows),
        "b_lon_gse": np.random.uniform(-180, 180, n_rows),
        "bx_gse_gsm": np.random.uniform(-10, 10, n_rows),
        "by_gse": np.random.uniform(-10, 10, n_rows),
        "bz_gse": np.random.uniform(-15, 5, n_rows),
        "by_gsm": np.random.uniform(-10, 10, n_rows),
        "bz_gsm": np.random.uniform(-30, 20, n_rows),
        "sigma_b_mag": np.full(n_rows, 0.5),
        "sigma_b_vec": np.full(n_rows, 0.3),
        "sigma_bx": np.full(n_rows, 0.2),
        "sigma_by": np.full(n_rows, 0.2),
        "sigma_bz": np.full(n_rows, 0.2),
        "proton_temp": np.random.uniform(10000, 100000, n_rows),
        "proton_density": np.random.uniform(1, 10, n_rows),
        "plasma_speed": np.random.uniform(300, 500, n_rows),
        "plasma_flow_lon": np.random.uniform(-180, 180, n_rows),
        "plasma_flow_lat": np.random.uniform(-30, 30, n_rows),
        "na_np_ratio": np.random.uniform(0.01, 0.1, n_rows),
        "flow_pressure": np.random.uniform(0.1, 5, n_rows),
        "sigma_proton_temp": np.full(n_rows, 5000.0),
        "sigma_proton_density": np.full(n_rows, 0.5),
        "sigma_plasma_speed": np.full(n_rows, 10.0),
        "sigma_flow_lon": np.full(n_rows, 5.0),
        "sigma_flow_lat": np.full(n_rows, 5.0),
        "sigma_na_np_ratio": np.full(n_rows, 0.01),
        "electric_field": np.random.uniform(0, 10, n_rows),
        "plasma_beta": np.random.uniform(0.01, 1, n_rows),
        "alfven_mach": np.random.uniform(1, 5, n_rows),
        "kp": np.random.uniform(0, 9, n_rows),
        "r_sunspot": np.full(n_rows, 50.0),
        "dst": np.random.uniform(-100, 20, n_rows),
        "ae": np.random.uniform(0, 500, n_rows),
        "prot_flux_1mev": np.random.uniform(100, 1000, n_rows),
        "prot_flux_2mev": np.random.uniform(10, 100, n_rows),
        "prot_flux_4mev": np.random.uniform(1, 50, n_rows),
        "prot_flux_10mev": np.random.uniform(0.1, 10, n_rows),
        "prot_flux_30mev": np.random.uniform(0.01, 1, n_rows),
        "prot_flux_60mev": np.random.uniform(0.001, 0.1, n_rows),
        "flag": np.full(n_rows, 0.0),
        "ap": np.random.uniform(0, 200, n_rows),
        "f107": np.random.uniform(60, 200, n_rows),
        "pc_n": np.random.uniform(0, 5, n_rows),
        "al": np.random.uniform(-500, 0, n_rows),
        "au": np.random.uniform(0, 500, n_rows),
        "m_sonic_mach": np.random.uniform(1, 10, n_rows),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# omni()
# ---------------------------------------------------------------------------

@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_single_year(mock_read_csv):
    """Test loading a single year of data."""
    df_2015 = make_synthetic_omni(2015, n_rows=10)
    mock_read_csv.return_value = df_2015

    result = omni("2015-01-01", "2015-01-10")

    # Check URL was called with correct year
    assert mock_read_csv.call_count == 1
    call_args = mock_read_csv.call_args[0][0]
    assert "omni2_2015.dat" in call_args

    # Check result shape and index
    assert result.shape[0] == 10
    assert result.shape[1] == 55  # All 55 OMNI2 fields
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.freq == "h"  # Hourly

    # Check time columns dropped
    assert "year" not in result.columns
    assert "doy" not in result.columns
    assert "hour" not in result.columns


@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_multi_year(mock_read_csv):
    """Test loading data spanning multiple years."""
    df_2015 = make_synthetic_omni(2015, n_rows=12)
    df_2016 = make_synthetic_omni(2016, n_rows=12)
    mock_read_csv.side_effect = [df_2015, df_2016]

    result = omni("2015-06-01", "2016-06-30")

    # Check both years were fetched
    assert mock_read_csv.call_count == 2
    urls = [call[0][0] for call in mock_read_csv.call_args_list]
    assert any("omni2_2015.dat" in url for url in urls)
    assert any("omni2_2016.dat" in url for url in urls)

    # Check concatenation
    assert result.shape[0] == 24
    assert result.index[0].year == 2015
    assert result.index[-1].year == 2016


@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_fill_values_replaced(mock_read_csv):
    """Test that documented fill values are replaced with NaN."""
    # Create a synthetic DataFrame with known fill values
    df = make_synthetic_omni(2015, n_rows=5)
    # Insert known fill values at specific locations
    df.loc[0, "bz_gsm"] = 999.9  # Fill value for bz_gsm
    df.loc[1, "ae"] = 9999.0     # Fill value for ae
    df.loc[2, "proton_temp"] = 9999999.0  # Fill value for proton_temp
    df.loc[3, "kp"] = 99.0       # Fill value for kp
    df.loc[4, "flag"] = 0.0      # Fill value for flag

    mock_read_csv.return_value = df

    result = omni("2015-01-01", "2015-01-05")

    # Check that fill values are NaN
    assert pd.isna(result.iloc[0]["bz_gsm"])
    assert pd.isna(result.iloc[1]["ae"])
    assert pd.isna(result.iloc[2]["proton_temp"])
    assert pd.isna(result.iloc[3]["kp"])
    assert pd.isna(result.iloc[4]["flag"])


@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_slicing_by_date(mock_read_csv):
    """Test that date-range slicing works correctly."""
    df = make_synthetic_omni(2015, n_rows=24)
    mock_read_csv.return_value = df

    result = omni("2015-01-02", "2015-01-15")

    # Result should only contain rows within the requested range
    assert result.index[0].date() == pd.Timestamp("2015-01-02").date()
    assert result.index[-1].date() == pd.Timestamp("2015-01-15").date()


@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_datetime_index_hourly(mock_read_csv):
    """Test that returned index is hourly DatetimeIndex."""
    df = make_synthetic_omni(2015, n_rows=48)
    mock_read_csv.return_value = df

    result = omni("2015-01-01", "2015-02-16")

    assert isinstance(result.index, pd.DatetimeIndex)
    # Check that successive hours increment by 1 hour
    diffs = result.index[1:] - result.index[:-1]
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_omni_start_after_end_raises():
    """Test that start > end raises ValueError."""
    with pytest.raises(ValueError, match="start .* must be <= end"):
        omni("2015-12-31", "2015-01-01")


def test_omni_invalid_url_raises():
    """Test that network error is wrapped in ValueError."""
    with patch("ml_fw.data.loaders.pd.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = Exception("Connection refused")
        with pytest.raises(ValueError, match="Failed to load"):
            omni("2099-01-01", "2099-01-02")


@patch("ml_fw.data.loaders.pd.read_csv")
def test_omni_column_count(mock_read_csv):
    """Test that all 55 OMNI2 fields are present in output."""
    df = make_synthetic_omni(2015, n_rows=5)
    mock_read_csv.return_value = df

    result = omni("2015-01-01", "2015-01-05")

    # Should have exactly 55 columns (time fields stripped)
    assert len(result.columns) == 55
    # Verify no time columns remain
    assert not any(col in result.columns for col in ["year", "doy", "hour"])
