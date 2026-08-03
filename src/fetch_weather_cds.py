"""Bulk ERA5 daily weather via Copernicus CDS — the fast path for backfill.

Where fetch_weather.py trickles through Open-Meteo's free-tier quota
(~2–3 days of quota per YEAR of data), this script pulls the same ERA5
reanalysis directly from the Copernicus Climate Data Store as a handful of
area-wide NetCDF downloads — the whole Thailand bbox, all years, in one
queued session. Output is the SAME ``data/weather/weather_cache.parquet``
schema/units as fetch_weather.py, so data_loader / features.py are untouched:

    lat_grid, lon_grid, date, temp_max(°C), temp_min(°C),
    precip_sum(mm), wind_max(km/h), et0(mm)

Real-data notes (project hard rule):
  • Every field comes from the `derived-era5-single-levels-daily-statistics`
    CDS dataset (ERA5 post-processed daily aggregates), aggregated on
    UTC+07:00 days to match the Open-Meteo rows (timezone=Asia/Bangkok).
  • ET0 is not an ERA5 output anywhere — Open-Meteo *computes* it server-side
    with FAO-56 Penman-Monteith from ERA5 fields. We do exactly the same,
    from real ERA5 daily statistics (Tmax/Tmin, dewpoint, wind, solar
    radiation) + real ERA5 orography for station pressure. Standard published
    formula, no fitted or invented values.
  • 0.1° cells receive their nearest 0.25° ERA5 point's value — nearest-
    neighbour at the source's native resolution, identical to what the
    Open-Meteo API returns for those coordinates. No interpolation.
  • Existing cache rows (e.g. from fetch_weather.py) are never overwritten.

Setup (one-time):
  1. Register (free): https://cds.climate.copernicus.eu
  2. Accept the ERA5 licence once on the dataset page:
     https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics
  3. Put your key in ~/.cdsapirc  (shown on https://cds.climate.copernicus.eu/profile):
         url: https://cds.climate.copernicus.eu/api
         key: <your-personal-access-token>
  4. pip install cdsapi xarray netcdf4

Usage:
    cd src && python fetch_weather_cds.py [--years 2012-2024] [--et0-selftest]

Downloads are cached per (variable, statistic, year) under
``data/weather/cds/`` — rerunning skips finished files (idempotent), so a
killed run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetch_weather_cds")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_weather import (  # noqa: E402  (shared config — single source of truth)
    WEATHER_CACHE_PATH,
    ERA5_NATIVE_RES_DEG,
    _era5_parent,
    discover_active_cells,
)
from io_utils import read_table, write_table  # noqa: E402

# lon_min, lat_min, lon_max, lat_max — same bbox as the FIRMS pipeline.
BBOX = tuple(float(x) for x in os.getenv("FIRMS_BBOX", "96,4,107,22").split(","))
# CDS area order: North, West, South, East.
CDS_AREA = [BBOX[3], BBOX[0], BBOX[1], BBOX[2]]

CDS_DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
# Match Open-Meteo rows, which aggregate days on Asia/Bangkok wall clock.
CDS_TIME_ZONE = "utc+07:00"

CDS_DIR = os.path.join(os.path.dirname(WEATHER_CACHE_PATH), "cds")

# (variable, daily_statistic) → internal name. Everything below is a real
# ERA5 field from ONE dataset (single licence); units converted explicitly in
# _year_frame(). Surface pressure gives FAO-56's psychrometric constant
# directly — no separate orography download / second-dataset licence needed.
REQUESTS: List[Tuple[str, str, str]] = [
    ("2m_temperature",                     "daily_maximum", "t2m_max"),
    ("2m_temperature",                     "daily_minimum", "t2m_min"),
    ("total_precipitation",                "daily_sum",     "tp_sum"),
    ("10m_wind_speed",                     "daily_maximum", "ws10_max"),
    # ET0 (FAO-56) inputs:
    ("2m_dewpoint_temperature",            "daily_mean",    "d2m_mean"),
    ("10m_wind_speed",                     "daily_mean",    "ws10_mean"),
    ("surface_solar_radiation_downwards",  "daily_sum",     "ssrd_sum"),
    ("surface_pressure",                   "daily_mean",    "sp_mean"),
]


# ─────────────────────────────────────────────────────────────────────────────
# FAO-56 Penman-Monteith reference evapotranspiration (Allen et al., 1998)
# ─────────────────────────────────────────────────────────────────────────────

def _svp_kpa(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (kPa) at temperature °C — FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def et0_fao56(
    tmax_c: np.ndarray,
    tmin_c: np.ndarray,
    tdew_c: np.ndarray,
    wind10_ms: np.ndarray,
    rs_mj: np.ndarray,          # daily solar radiation, MJ m-2 day-1
    lat_deg: np.ndarray,
    doy: np.ndarray,
    p_kpa: np.ndarray,          # real ERA5 daily-mean surface pressure, kPa
) -> np.ndarray:
    """Daily reference evapotranspiration (mm/day), FAO-56 eqs. 6–42.

    All inputs are real ERA5 daily statistics; the formula is the published
    FAO-56 standard (the same one Open-Meteo applies to ERA5 for its
    ``et0_fao_evapotranspiration`` output).
    """
    tmean = (tmax_c + tmin_c) / 2.0

    # Psychrometric constant (eq. 8) from REAL surface pressure. Elevation for
    # the small clear-sky-radiation term (eq. 37) is inverted from that same
    # real pressure via eq. 7 — no separate orography dataset needed.
    gamma = 0.000665 * p_kpa
    elev_m = (293.0 / 0.0065) * (1.0 - (p_kpa / 101.3) ** (1.0 / 5.26))

    # Slope of the SVP curve at Tmean (eq. 13)
    delta = 4098.0 * _svp_kpa(tmean) / (tmean + 237.3) ** 2

    # Vapour pressures (eqs. 12, 14)
    es = (_svp_kpa(tmax_c) + _svp_kpa(tmin_c)) / 2.0
    ea = _svp_kpa(tdew_c)

    # 10 m → 2 m wind (eq. 47)
    u2 = wind10_ms * (4.87 / math.log(67.8 * 10.0 - 5.42))

    # Extraterrestrial radiation Ra (eqs. 21–25)
    phi = np.radians(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)
    )

    # Net radiation (eqs. 37–40), G ≈ 0 for daily steps (eq. 42)
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = (1.0 - 0.23) * rs_mj
    rel = np.clip(np.divide(rs_mj, rso, out=np.zeros_like(rs_mj), where=rso > 0), 0.0, 1.0)
    sigma = 4.903e-9  # MJ K-4 m-2 day-1
    rnl = sigma * ((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2.0 \
        * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0.0, None))) * (1.35 * rel - 0.35)
    rn = rns - rnl

    et0 = (0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) \
        / (delta + gamma * (1.0 + 0.34 * u2))
    return np.clip(et0, 0.0, None)


def et0_selftest() -> None:
    """Physical-plausibility check over Thailand-like conditions."""
    # Hot dry-season day, northern Thailand (~977 hPa at 300 m): 4–7 mm/day.
    hot = et0_fao56(
        tmax_c=np.array([36.0]), tmin_c=np.array([22.0]), tdew_c=np.array([16.0]),
        wind10_ms=np.array([2.5]), rs_mj=np.array([22.0]),
        lat_deg=np.array([18.8]), doy=np.array([90]), p_kpa=np.array([97.7]),
    )[0]
    # Overcast rainy-season day, near sea level (~1013 hPa): 2–4 mm/day.
    wet = et0_fao56(
        tmax_c=np.array([30.0]), tmin_c=np.array([24.0]), tdew_c=np.array([23.0]),
        wind10_ms=np.array([1.5]), rs_mj=np.array([12.0]),
        lat_deg=np.array([13.7]), doy=np.array([220]), p_kpa=np.array([101.3]),
    )[0]
    assert 4.0 <= hot <= 7.5, f"dry-season ET0 implausible: {hot:.2f} mm/day"
    assert 1.5 <= wet <= 4.5, f"rainy-season ET0 implausible: {wet:.2f} mm/day"
    assert hot > wet, "dry-season ET0 should exceed rainy-season ET0"
    log.info("ET0 self-test OK: dry-season=%.2f mm/day, rainy-season=%.2f mm/day", hot, wet)


# ─────────────────────────────────────────────────────────────────────────────
# CDS downloads (idempotent per variable-statistic-year)
# ─────────────────────────────────────────────────────────────────────────────

def _cds_client():
    try:
        import cdsapi
    except ImportError:
        raise SystemExit(
            "cdsapi not installed — run: pip install cdsapi xarray netcdf4"
        )
    try:
        return cdsapi.Client()
    except Exception as exc:
        raise SystemExit(
            f"CDS credentials problem ({exc}).\n"
            "Create a free account at https://cds.climate.copernicus.eu, accept the\n"
            "ERA5 licence on the dataset page, then write ~/.cdsapirc:\n"
            "    url: https://cds.climate.copernicus.eu/api\n"
            "    key: <personal access token from your profile page>"
        )


def _download_year(client, variable: str, statistic: str, year: int, dest: str) -> str:
    """One (variable, statistic, year) NetCDF over the bbox. Skips if cached."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    log.info("CDS retrieve: %s / %s / %d → %s", variable, statistic, year, dest)
    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": statistic,
        "time_zone": CDS_TIME_ZONE,
        "frequency": "1_hourly",
        "area": CDS_AREA,
    }
    tmp = dest + ".part"
    client.retrieve(CDS_DAILY_DATASET, request, tmp)
    os.replace(tmp, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# NetCDF → cache-schema DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _open_da(path: str):
    """Open the single data variable of a CDS NetCDF as a DataArray."""
    import xarray as xr

    ds = xr.open_dataset(path)
    data_vars = [v for v in ds.data_vars if v.lower() not in ("number", "expver")]
    if len(data_vars) != 1:
        raise ValueError(f"{path}: expected one data variable, got {list(ds.data_vars)}")
    da = ds[data_vars[0]]
    # Normalise dimension names across CDS product generations.
    rename = {}
    for cand, target in (("valid_time", "time"), ("date", "time")):
        if cand in da.dims:
            rename[cand] = target
    if rename:
        da = da.rename(rename)
    return da


def _year_frame(year: int, files: Dict[str, str]) -> pd.DataFrame:
    """Merge one year's NetCDFs into a per-(ERA5 point, date) frame with the
    cache's column names and units."""
    frames = {}
    for name, path in files.items():
        da = _open_da(path)
        df = da.to_dataframe(name=name).reset_index()
        df["date"] = pd.to_datetime(df["time"]).dt.date
        frames[name] = df[["latitude", "longitude", "date", name]]

    out = None
    for name, df in frames.items():
        out = df if out is None else out.merge(df, on=["latitude", "longitude", "date"], how="outer")

    # Unit conversions — targets match Open-Meteo daily defaults exactly.
    out["temp_max"] = out["t2m_max"] - 273.15                 # K → °C
    out["temp_min"] = out["t2m_min"] - 273.15                 # K → °C
    out["precip_sum"] = out["tp_sum"] * 1000.0                # m → mm
    out["wind_max"] = out["ws10_max"] * 3.6                   # m/s → km/h
    doy = pd.to_datetime(out["date"]).dt.dayofyear.to_numpy()
    out["et0"] = et0_fao56(
        tmax_c=out["temp_max"].to_numpy(float),
        tmin_c=out["temp_min"].to_numpy(float),
        tdew_c=(out["d2m_mean"] - 273.15).to_numpy(float),
        wind10_ms=out["ws10_mean"].to_numpy(float),
        rs_mj=(out["ssrd_sum"] / 1e6).to_numpy(float),        # J m-2 → MJ m-2
        lat_deg=out["latitude"].to_numpy(float),
        doy=doy,
        p_kpa=(out["sp_mean"] / 1000.0).to_numpy(float),      # Pa → kPa
    )
    cols = ["latitude", "longitude", "date",
            "temp_max", "temp_min", "precip_sum", "wind_max", "et0"]
    out = out[cols].dropna(subset=["temp_max", "temp_min"])
    log.info("%d: built %d ERA5 point-days", year, len(out))
    return out


def _fan_out_to_cells(points: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Assign each active 0.1° cell its nearest ERA5 point's rows."""
    pts = points.copy()
    pts["latitude"] = pts["latitude"].astype(float).round(6)
    pts["longitude"] = pts["longitude"].astype(float).round(6)

    cell_rows = []
    for _, row in cells.iterrows():
        la, lo = float(row["lat_grid"]), float(row["lon_grid"])
        pla, plo = _era5_parent(la, lo)
        cell_rows.append((la, lo, pla, plo))
    mapping = pd.DataFrame(cell_rows, columns=["lat_grid", "lon_grid", "latitude", "longitude"])

    merged = mapping.merge(pts, on=["latitude", "longitude"], how="inner")
    n_missing = mapping[["latitude", "longitude"]].drop_duplicates().merge(
        pts[["latitude", "longitude"]].drop_duplicates(),
        on=["latitude", "longitude"], how="left", indicator=True,
    )["_merge"].eq("left_only").sum()
    if n_missing:
        log.warning("%d ERA5 parent point(s) had no CDS data (edge of bbox)", n_missing)
    keep = ["lat_grid", "lon_grid", "date",
            "temp_max", "temp_min", "precip_sum", "wind_max", "et0"]
    out = merged[keep].copy()
    for c in keep[3:]:
        out[c] = out[c].astype("float32")
    return out


def _merge_into_cache(fresh: pd.DataFrame, cache_path: str) -> int:
    """Append rows whose (cell, date) key is absent. Existing rows win."""
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        cache = read_table(cache_path)
        cache["date"] = pd.to_datetime(cache["date"]).dt.date
    else:
        cache = pd.DataFrame()
    before = len(cache)
    combined = pd.concat([cache, fresh], ignore_index=True) if before else fresh
    combined = combined.drop_duplicates(subset=["lat_grid", "lon_grid", "date"], keep="first")
    combined = combined.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    write_table(combined, cache_path)
    return len(combined) - before


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_years(spec: str) -> List[int]:
    years: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def main(years: List[int]) -> None:
    et0_selftest()
    os.makedirs(CDS_DIR, exist_ok=True)
    client = _cds_client()

    cells, _, _ = discover_active_cells()
    log.info("Fan-out target: %d active FIRMS cells → nearest ERA5 0.25° point",
             len(cells))

    this_year = datetime.now(timezone.utc).year
    for year in years:
        if year > this_year:
            continue
        files: Dict[str, str] = {}
        for variable, statistic, name in REQUESTS:
            dest = os.path.join(CDS_DIR, f"{name}_{year}.nc")
            files[name] = _download_year(client, variable, statistic, year, dest)
        frame = _year_frame(year, files)
        fresh = _fan_out_to_cells(frame, cells)
        added = _merge_into_cache(fresh, WEATHER_CACHE_PATH)
        log.info("%d: +%d new cache rows (existing rows preserved) → %s",
                 year, added, WEATHER_CACHE_PATH)

    log.info("CDS backfill done for years: %s", years)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--years", default="2012-2024",
                   help="Years to backfill, e.g. 2012-2024 or 2019,2021-2023")
    p.add_argument("--et0-selftest", action="store_true",
                   help="Run the ET0 plausibility self-test and exit")
    args = p.parse_args()
    if args.et0_selftest:
        et0_selftest()
        sys.exit(0)
    main(_parse_years(args.years))
