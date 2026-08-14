"""Stream OMNI2 low-resolution hourly space-weather data from NASA CDAWeb.

Data source: https://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/

All 55 OMNI2 fields are included. Missing/fill values (per-column) are
converted to NaN for analysis-ready DataFrames.
"""

from typing import Optional
import numpy as np
import pandas as pd

# 55 OMNI2 field names (pythonic short names), in exact order from omni2.text
_COLUMNS = [
    "year", "doy", "hour",  # 1–3: time fields
    "bartels_rotation", "imf_sc_id", "plasma_sc_id",  # 4–6: identifiers
    "n_imf_obs", "n_plasma_obs",  # 7–8: observation counts
    "b_mag_avg", "b_vec_mag",  # 9–10: field magnitude
    "b_lat_gse", "b_lon_gse",  # 11–12: field direction (GSE)
    "bx_gse_gsm", "by_gse", "bz_gse", "by_gsm", "bz_gsm",  # 13–17: components
    "sigma_b_mag", "sigma_b_vec", "sigma_bx", "sigma_by", "sigma_bz",  # 18–22: stds
    "proton_temp", "proton_density", "plasma_speed",  # 23–25: plasma params
    "plasma_flow_lon", "plasma_flow_lat",  # 26–27: flow direction
    "na_np_ratio", "flow_pressure",  # 28–29: derived plasma
    "sigma_proton_temp", "sigma_proton_density", "sigma_plasma_speed",  # 30–32: std devs
    "sigma_flow_lon", "sigma_flow_lat", "sigma_na_np_ratio",  # 33–35: more stds
    "electric_field", "plasma_beta", "alfven_mach",  # 36–38: derived fields
    "kp", "r_sunspot", "dst", "ae", "prot_flux_1mev",  # 39–43: indices & flux
    "prot_flux_2mev", "prot_flux_4mev", "prot_flux_10mev",  # 44–46: more flux
    "prot_flux_30mev", "prot_flux_60mev",  # 47–48: more flux
    "flag", "ap", "f107", "pc_n", "al", "au", "m_sonic_mach"  # 49–55: aux fields
]

# Fill/missing values (NaN substitutes) for each column, in same order. None = no replacement.
_FILL_VALUES = [
    None, None, None,  # year, doy, hour: no fill value
    9999.0, 99.0, 99.0,  # bartels, imf_id, plasma_id
    999.0, 999.0,  # n_imf, n_plasma
    999.9, 999.9,  # b_mag_avg, b_vec_mag
    999.9, 999.9,  # b_lat, b_lon
    999.9, 999.9, 999.9, 999.9, 999.9,  # bx, by_gse, bz_gse, by_gsm, bz_gsm
    999.9, 999.9, 999.9, 999.9, 999.9,  # sigma fields
    9999999.0, 999.9, 9999.0,  # proton_temp, density, speed
    999.9, 999.9,  # flow angles
    9.999, 99.99,  # ratio, pressure
    9999999.0, 999.9, 9999.0,  # sigma temps/density/speed
    999.9, 999.9, 9.999,  # sigma angles/ratio
    999.99, 999.99, 999.9,  # electric_field, beta, alfven_mach
    99.0, 999.0, 99999.0, 9999.0, 999999.99,  # kp, r, dst, ae, flux1
    99999.99, 99999.99, 99999.99,  # flux2–4
    99999.99, 99999.99,  # flux10, flux30
    0.0, 999.0, 999.9, 999.9, 99999.0, 99999.0, 99.9  # flag, ap, f107, pc, al, au, m_sonic
]

_BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"


def omni(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Load OMNI2 low-res hourly data for a date range from NASA CDAWeb.

    Data is streamed (not downloaded to disk) via pandas.read_csv and
    returned with a DatetimeIndex, all 55 OMNI2 fields included, and fill
    values replaced with NaN.

    Parameters
    ----------
    start : str or pd.Timestamp
        Start date (inclusive), e.g. '2015-01-01' or pd.Timestamp(...).
    end : str or pd.Timestamp
        End date (inclusive), e.g. '2015-12-31' or pd.Timestamp(...).

    Returns
    -------
    pd.DataFrame
        Shape (n_hours, 55), columns are OMNI2 fields, index is DatetimeIndex
        at hourly frequency. Fill values are NaN.

    Raises
    ------
    ValueError
        If start > end or if data cannot be fetched from the server.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts}) must be <= end ({end_ts})")

    # Collect all years that span the date range
    years = range(start_ts.year, end_ts.year + 1)
    dfs = []

    for year in years:
        url = _BASE_URL.format(year=year)
        try:
            # Stream the file directly; pandas.read_csv handles HTTP transparently
            df_year = pd.read_csv(url, sep=r"\s+", header=None, names=_COLUMNS)
        except Exception as e:
            raise ValueError(f"Failed to load {url}: {e}") from e

        dfs.append(df_year)

    # Concatenate all years
    df = pd.concat(dfs, ignore_index=True)

    # Build DatetimeIndex from year, doy, hour
    # doy is 1-indexed day of year; year*1000+doy forms YYYYDDD
    dt_index = pd.to_datetime(
        df["year"] * 1000 + df["doy"], format="%Y%j"
    ) + pd.to_timedelta(df["hour"], unit="h")
    df.index = dt_index
    df.index.name = None

    # Drop the time-component columns
    df = df.drop(columns=["year", "doy", "hour"])

    # Slice to requested date range (inclusive)
    df = df.loc[start_ts:end_ts]

    # Replace fill values with NaN
    # (skip the first 3 elements of _FILL_VALUES which were for year/doy/hour)
    for col, fill_val in zip(df.columns, _FILL_VALUES[3:]):
        if fill_val is not None:
            df[col] = df[col].replace(fill_val, np.nan)

    # Convert all columns to float64 for consistency
    df = df.astype("float64")

    return df
