"""Daily, past-only features and complete three-day satellite-detection labels."""

from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 3
HISTORY_DAYS = 7
FEATURES = ["fire_3d", "frp_3d", "frp_max", "fire_days_7d",
            "fire_yesterday", "frp_trend", "bright_mean", "confidence_mean"]
SATELLITES = {"N": "SNPP", "N20": "NOAA20", "N21": "NOAA21",
              "VIIRS_SNPP_NRT": "SNPP", "VIIRS_NOAA20_NRT": "NOAA20",
              "VIIRS_NOAA21_NRT": "NOAA21"}
TARGET_POLICY = "exclude_known_nonvegetation_v1"
QUALITY_FEATURES = ["low_confidence_fraction_7d", "night_fraction_7d",
                    "daynight_known_fraction_7d", "season_sin", "season_cos"]


def target_observations(observations, policy=None):
    """Type is retrospective outcome annotation, never an ignition-cause label.

    Untyped NRT observations remain eligible. Do not infer industrial sources
    from repetitions or project an archive classification into earlier features.
    """
    if policy is None or policy == "all_hotspots":
        return observations
    if policy != TARGET_POLICY:
        raise ValueError(f"Unknown target policy: {policy}")
    if "hotspot_type" not in observations:
        raise ValueError("Hotspot types unavailable; regenerate the observation archive")
    return observations[~observations.hotspot_type.isin([1, 2, 3])]


def load_observations(paths, grid=0.1):
    if grid <= 0:
        raise ValueError("GRID_SIZE must be positive")
    frames, input_rows = [], 0
    for path in paths:
        raw = pd.read_csv(path, dtype={"acq_time": str})
        input_rows += len(raw)
        times = raw["acq_time"].str.replace(r"\.0$", "", regex=True).str.zfill(4)
        frame = pd.DataFrame({
            "observed_at": pd.to_datetime(raw["acq_date"] + " " + times,
                format="%Y-%m-%d %H%M", errors="coerce"),
            "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
        })
        source = raw["satellite"] if "satellite" in raw else raw.get("dataset")
        if source is None:
            raise ValueError(f"Satellite identity missing in {path}")
        frame["satellite"] = source.replace(SATELLITES)
        frame["hotspot_type"] = pd.to_numeric(raw.get("type", pd.Series(np.nan, index=raw.index)), errors="coerce")
        frame["hotspot_type"] = frame.hotspot_type.where(frame.hotspot_type.isin([0, 1, 2, 3]))
        frame["daynight"] = raw.get("daynight", pd.Series("U", index=raw.index)).astype(str).str.upper()
        frame["daynight"] = frame.daynight.where(frame.daynight.isin(["D", "N"]), "U")
        for name, candidates in {"brightness": ["bright_ti4", "brightness", "bright"],
                                 "frp": ["frp"]}.items():
            frame[name] = np.nan
            for column in candidates:
                if column in raw:
                    frame[name] = frame[name].fillna(pd.to_numeric(raw[column], errors="coerce"))
        conf = raw["confidence"].astype(str).str.lower().replace({"l": "0", "n": "50", "h": "100"})
        frame["confidence"] = pd.to_numeric(conf, errors="coerce")
        frame = frame.dropna(subset=["observed_at", "latitude", "longitude", "satellite"])
        frame = frame[frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180)]
        frames.append(frame)
    if not frames:
        raise ValueError("No observation CSVs found")
    observations = pd.concat(frames, ignore_index=True)
    valid_rows = len(observations)
    # Same platform, timestamp and pixel repeated across archive/NRT files is one detection.
    keys = ["satellite", "observed_at", "latitude", "longitude"]
    # Prefer a typed archive record over its untyped NRT duplicate. Conflicting
    # known types remain unknown rather than silently choosing a source.
    type_counts = observations.groupby(keys).hotspot_type.transform("nunique")
    conflicts = int((type_counts > 1).sum())
    observations.loc[type_counts > 1, "hotspot_type"] = np.nan
    observations = observations.sort_values("hotspot_type", na_position="last", kind="stable").drop_duplicates(keys)
    observations["date"] = observations.observed_at.dt.normalize()
    observations["lat_cell"] = np.rint(observations.latitude / grid).astype("int32")
    observations["lon_cell"] = np.rint(observations.longitude / grid).astype("int32")
    return observations, {"input_rows": input_rows, "invalid_rows": input_rows - valid_rows,
                          "duplicates_removed": valid_rows - len(observations),
                          "unique_detections": len(observations), "conflicting_type_rows": conflicts,
                          "hotspot_types": {str(int(k)): int(v) for k, v in observations.hotspot_type.value_counts().items()},
                          "unknown_type_detections": int(observations.hotspot_type.isna().sum())}


def daily_panel(observations, observation_end=None, target_policy=None, candidate_cells=None):
    """A row at date D uses D-7..D-1 only and targets D+1..D+3.

    A grid enters the population the day after its first recorded detection.
    Zero means no record, not verified absence of fire. Days with no records
    anywhere in the supplied region are treated as unknown coverage.
    """
    if observations.empty:
        raise ValueError("No valid observations")
    end = pd.Timestamp(observation_end or observations.date.max()).normalize()
    obs = observations[observations.date <= end].copy()
    obs["target_detection"] = obs.index.isin(target_observations(obs, target_policy).index).astype(int)
    obs["low_confidence"] = obs.confidence.eq(0).astype(int)
    daynight = obs.get("daynight", pd.Series("U", index=obs.index))
    obs["night"] = daynight.eq("N").astype(int)
    obs["daynight_known"] = daynight.isin(["D", "N"]).astype(int)
    start = obs.date.min()
    dates = pd.date_range(start, end + pd.Timedelta(days=1))
    available = pd.Series(dates.isin(obs.date.unique()), index=dates).astype("int8")
    history_known = available.shift(1).rolling(HISTORY_DAYS, min_periods=HISTORY_DAYS).sum().eq(HISTORY_DAYS)
    future_known = pd.concat([available.shift(-day) for day in range(1, HORIZON + 1)], axis=1).eq(1).all(axis=1)
    daily = obs.groupby(["lat_cell", "lon_cell", "date"]).agg(
        fire_count=("observed_at", "size"), frp_sum=("frp", "sum"),
        frp_peak=("frp", "max"), brightness=("brightness", "mean"),
        confidence=("confidence", "mean"), target_count=("target_detection", "sum"),
        low_count=("low_confidence", "sum"), night_count=("night", "sum"), known_count=("daynight_known", "sum"))
    parts = []
    groups = dict(tuple(daily.groupby(level=[0, 1], sort=True)))
    cells = list(groups) if candidate_cells is None else list(candidate_cells[["lat_cell", "lon_cell"]].drop_duplicates().itertuples(index=False, name=None))
    for lat, lon in cells:
        group = groups.get((lat, lon))
        grid_start = group.index.get_level_values("date").min() if candidate_cells is None else start
        source = group.droplevel([0, 1]) if group is not None else pd.DataFrame(columns=daily.columns, dtype=float)
        values = source.reindex(pd.date_range(grid_start, dates[-1])).fillna(0)
        previous = values.shift(1).fillna(0)
        fires = values.target_count.gt(0)
        frame = pd.DataFrame(index=values.index)
        frame["fire_3d"] = previous.fire_count.rolling(3, min_periods=1).sum()
        frame["frp_3d"] = previous.frp_sum.rolling(3, min_periods=1).sum()
        frame["frp_max"] = previous.frp_peak.rolling(3, min_periods=1).max()
        frame["fire_days_7d"] = previous.fire_count.gt(0).rolling(7, min_periods=1).sum()
        frame["fire_yesterday"] = previous.fire_count
        frame["frp_trend"] = previous.frp_sum.diff().fillna(0)
        frame["bright_mean"] = previous.brightness.rolling(3, min_periods=1).mean()
        frame["confidence_mean"] = previous.confidence.rolling(3, min_periods=1).mean()
        total = previous.fire_count.rolling(7, min_periods=1).sum()
        for feature, column in [("low_confidence_fraction_7d", "low_count"), ("night_fraction_7d", "night_count"),
                                ("daynight_known_fraction_7d", "known_count")]:
            frame[feature] = previous[column].rolling(7, min_periods=1).sum().div(total.where(total > 0)).fillna(0)
        angle = 2 * np.pi * (frame.index.strftime("%j").astype(int) - 1) / 365.2425
        frame["season_sin"], frame["season_cos"] = np.sin(angle), np.cos(angle)
        frame = frame.astype("float32")
        days = np.full(len(values), -1, dtype="int8")
        for day in range(HORIZON, 0, -1):
            days[fires.shift(-day, fill_value=False).to_numpy()] = day
        frame["days_until_detection"] = days
        frame["target"] = (days > 0).astype("int8")
        frame["history_complete"] = history_known.reindex(frame.index).to_numpy()
        frame["label_complete"] = future_known.reindex(frame.index).to_numpy()
        frame["fire_count"] = values.fire_count.astype("int32")
        frame["lat_cell"], frame["lon_cell"] = np.int32(lat), np.int32(lon)
        frame.index.name = "date"
        # Supplied country cells include places with no previous detection.
        # Legacy mode still enters a cell only after its first observation.
        parts.append(frame.iloc[1:].reset_index())
    panel = pd.concat(parts, ignore_index=True)
    return panel, {"observation_start": str(start.date()), "observation_end": str(end.date()),
                   "regional_days_without_records": int((available.iloc[:-1] == 0).sum()),
                   "candidate_grids": len(parts), "daily_rows": len(panel)}


def chronological_split(panel, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between zero and one")
    eligible = panel[panel.history_complete & panel.label_complete]
    dates = np.sort(eligible.date.unique())
    if len(dates) < 30:
        raise ValueError("Need at least 30 eligible dates for chronological evaluation")
    test_start = pd.Timestamp(dates[int(len(dates) * (1 - test_size))])
    train = eligible[eligible.date + pd.Timedelta(days=HORIZON) < test_start]
    test = eligible[eligible.date >= test_start]
    if train.empty or test.empty or train.target.nunique() < 2 or not train.target.any():
        raise ValueError("Insufficient positive/negative training data after temporal split")
    return train, test, test_start


def purged_folds(frame, folds=4):
    """Contiguous date blocks for cross-fitting, with a label purge and embargo.

    Fitting rows whose three-day label window reaches into the held-out block,
    or whose seven-day feature history starts inside it, are removed.
    Yields boolean masks aligned with ``frame``.
    """
    if folds < 2:
        raise ValueError("CALIBRATION_FOLDS must be at least two")
    dates = np.sort(frame.date.unique())
    if len(dates) < folds * 10:
        raise ValueError("Too few dates for purged cross-fitting")
    label_gap = pd.Timedelta(days=HORIZON)
    history_gap = pd.Timedelta(days=HISTORY_DAYS)
    values = frame.date.to_numpy()
    for block in np.array_split(dates, folds):
        start, end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        held = (values >= block[0]) & (values <= block[-1])
        fit = (values < np.datetime64(start - label_gap)) | (values > np.datetime64(end + history_gap))
        yield fit, held


def observation_paths(root, raw_dir=None, firms_path=None):
    root = Path(root)
    paths = sorted(Path(raw_dir or root / "data/raw").glob("*.csv"))
    firms = Path(firms_path or root / "data/firms/firms_all.csv")
    if firms.exists():
        paths.append(firms)
    return paths
