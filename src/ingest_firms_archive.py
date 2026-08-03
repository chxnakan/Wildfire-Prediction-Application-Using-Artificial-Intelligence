"""Ingest NASA FIRMS bulk-archive downloads into data/raw/.

FIRMS archive requests (firms.modaps.eosdis.nasa.gov/download) arrive as
DL_FIRE_<SAT>-C2_<id>/ folders holding fire_archive_*.csv (science-quality)
and fire_nrt_*.csv (near-real-time) — often GLOBAL coverage and several GB.
This tool converts them into what train.py's data_loader expects:

  1. stream each CSV in chunks (bounded memory, works on 50M+ row files)
  2. keep only detections inside the Thailand model BBOX (FIRMS_BBOX)
  3. keep only the columns the pipeline uses
  4. merge archive + NRT per satellite, dedupe on (lat, lon, acq_date,
     acq_time) — the NRT file overlaps the archive tail
  5. write one compressed parquet per satellite into data/raw/, with the
     satellite token (J1V/J2V/SV-C2) preserved in the filename so
     data_loader._infer_dataset_from_filename assigns the right dataset

The original CSVs are left untouched (delete them manually once satisfied).

It can also pull NASA's per-country yearly summaries (no login, ~10 MB/yr,
already country-filtered) — the way around the archive tool's 1-year-per-
request limit. VIIRS SNPP yearly files exist for 2012–2024:

Usage:
    cd src && python ingest_firms_archive.py            # scan data/DL_FIRE_*
    cd src && python ingest_firms_archive.py --src /path/to/DL_FIRE_xxx ...
    cd src && python ingest_firms_archive.py --yearly 2018-2024   # country files
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest_firms_archive")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same bbox as fetch_firms.py / FIRMS_BBOX. lon_min, lat_min, lon_max, lat_max.
BBOX = tuple(float(x) for x in os.getenv("FIRMS_BBOX", "96,4,107,22").split(","))

RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
if not os.path.isabs(RAW_DIR):
    RAW_DIR = os.path.join(BASE_DIR, RAW_DIR)

CHUNK_ROWS = 2_000_000

# Columns the pipeline consumes (clean_hotspots + feature engineering).
# `type` (archive-only fire classification) is intentionally dropped.
KEEP_COLS = [
    "latitude", "longitude", "acq_date", "acq_time",
    # Bulk archives name brightness `brightness`/`bright_t31`; the country
    # yearly files use the native VIIRS names `bright_ti4`/`bright_ti5`.
    # data_loader.clean_hotspots derives `bright_main` from either.
    "frp", "brightness", "bright_t31", "bright_ti4", "bright_ti5", "confidence",
    "scan", "track", "satellite", "daynight",
]

# Satellite token in the bundle name → output filename. Keeping the token in
# the filename is REQUIRED: data_loader infers the dataset label from it.
SAT_TOKEN_RE = re.compile(r"(J1V|J2V|SV)-C2", re.IGNORECASE)


def _filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = BBOX
    lat = pd.to_numeric(chunk["latitude"], errors="coerce")
    lon = pd.to_numeric(chunk["longitude"], errors="coerce")
    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    out = chunk.loc[mask, [c for c in KEEP_COLS if c in chunk.columns]].copy()
    return out


def ingest_bundle(bundle_dir: str) -> Optional[str]:
    """One DL_FIRE_* folder → one bbox-filtered parquet in data/raw/."""
    m = SAT_TOKEN_RE.search(os.path.basename(bundle_dir))
    if not m:
        log.warning("Skipping %s — no J1V/J2V/SV satellite token in name", bundle_dir)
        return None
    token = m.group(0).upper()

    csvs = sorted(glob.glob(os.path.join(bundle_dir, "fire_*.csv")))
    if not csvs:
        log.warning("Skipping %s — no fire_*.csv inside", bundle_dir)
        return None

    frames: List[pd.DataFrame] = []
    for csv_path in csvs:
        total_in = kept = 0
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS, low_memory=False):
            total_in += len(chunk)
            sub = _filter_chunk(chunk)
            kept += len(sub)
            if len(sub):
                frames.append(sub)
        log.info("  %s: %d rows → %d in Thailand bbox",
                 os.path.basename(csv_path), total_in, kept)

    if not frames:
        log.warning("No in-bbox rows for %s", bundle_dir)
        return None

    df = pd.concat(frames, ignore_index=True)
    # archive + NRT overlap at the archive/NRT boundary — same physical
    # detection appears in both files. Natural key dedupe.
    before = len(df)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    df = df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    # Include the FIRMS request id in the filename so a second archive
    # request for the same satellite (e.g. a different date range) does not
    # clobber an earlier ingest. The satellite token must stay in the name
    # for data_loader's dataset inference.
    id_m = re.search(r"_(\d+)$", os.path.basename(bundle_dir.rstrip("/")))
    suffix = f"_{id_m.group(1)}" if id_m else ""
    out_path = os.path.join(RAW_DIR, f"fire_combined_{token}{suffix}.parquet")
    df.to_parquet(out_path, index=False, compression="zstd")
    log.info("%s: %d rows (deduped %d) → %s  [%s → %s]",
             token, len(df), before - len(df), out_path,
             df["acq_date"].min(), df["acq_date"].max())
    return out_path


# NASA per-country yearly summaries (open, no key). SNPP goes back to 2012;
# the current year is not published until the year closes.
COUNTRY_YEARLY_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/"
    "{year}/viirs-snpp_{year}_{country}.csv"
)
COUNTRY_YEARLY_DIR = os.path.join(BASE_DIR, "data", "country_yearly")


def ingest_country_yearly(years: List[int], country: str = "Thailand") -> List[str]:
    """Download + convert NASA's per-country yearly VIIRS SNPP CSVs.

    Each year becomes data/raw/fire_yearly_SV-C2_<year>.parquet — the SV-C2
    token keeps data_loader's dataset inference correct (VIIRS_SNPP_NRT).
    Idempotent: an already-downloaded CSV is reused, an existing parquet is
    rewritten from it.
    """
    import requests

    os.makedirs(COUNTRY_YEARLY_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    written = []
    for year in years:
        url = COUNTRY_YEARLY_URL.format(year=year, country=country)
        csv_path = os.path.join(COUNTRY_YEARLY_DIR, os.path.basename(url))

        if not os.path.exists(csv_path):
            log.info("Downloading %s ...", url)
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code == 404:
                log.warning("%d: not published yet (HTTP 404) — skipped", year)
                continue
            resp.raise_for_status()
            tmp = csv_path + ".part"
            with open(tmp, "wb") as fh:
                for block in resp.iter_content(chunk_size=1 << 20):
                    fh.write(block)
            os.replace(tmp, csv_path)

        frames: List[pd.DataFrame] = []
        total_in = kept = 0
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS, low_memory=False):
            total_in += len(chunk)
            sub = _filter_chunk(chunk)
            kept += len(sub)
            if len(sub):
                frames.append(sub)
        if not frames:
            log.warning("%d: no in-bbox rows — skipped", year)
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
        df = df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)

        out_path = os.path.join(RAW_DIR, f"fire_yearly_SV-C2_{year}.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        log.info("%d: %d rows → %d in bbox → %s  [%s → %s]",
                 year, total_in, len(df), out_path,
                 df["acq_date"].min(), df["acq_date"].max())
        written.append(out_path)
    return written


def _parse_year_args(tokens: List[str]) -> List[int]:
    """Accept '2018-2024' ranges and bare years, in any mix."""
    years: List[int] = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def main(sources: Optional[List[str]] = None) -> List[str]:
    bundles = sources or sorted(glob.glob(os.path.join(BASE_DIR, "data", "DL_FIRE_*")))
    if not bundles:
        raise SystemExit("No DL_FIRE_* bundles found under data/ — nothing to ingest.")
    log.info("Ingesting %d bundle(s) → %s (bbox %s)", len(bundles), RAW_DIR, BBOX)
    written = []
    for b in bundles:
        out = ingest_bundle(b)
        if out:
            written.append(out)
    log.info("Done — %d parquet file(s) in %s", len(written), RAW_DIR)
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", nargs="*", help="DL_FIRE_* folders (default: scan data/)")
    p.add_argument("--yearly", nargs="*",
                   help="Country-yearly SNPP years, e.g. --yearly 2018-2024")
    args = p.parse_args()
    if args.yearly:
        ingest_country_yearly(_parse_year_args(args.yearly))
    else:
        main(args.src)
