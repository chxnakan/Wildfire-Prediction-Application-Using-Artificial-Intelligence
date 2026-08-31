#!/bin/sh
# =============================================================================
# Container entrypoint
# =============================================================================
# 1. Seed a complete promoted artifact set on a new persistent volume
#    (mounted at /srv on Railway). Existing complete bundles are preserved;
#    partial bundles fail loudly rather than mixing model versions.
# 2. Exec whatever was passed in (the cron service overrides CMD; the web
#    service uses the default uvicorn invocation).
# =============================================================================
set -e

seed_missing_file() {
  SRC="$1"
  DST="$2"
  if [ ! -f "$SRC" ]; then
    echo "[start.sh] bootstrap source missing — cannot seed $DST" >&2
    return 1
  fi
  if [ -e "$DST" ]; then
    if [ -f "$DST" ]; then
      return 0
    fi
    echo "[start.sh] expected a file but found another entry at $DST" >&2
    return 1
  fi
  mkdir -p "$(dirname "$DST")"
  # The existence check prevents normal restarts from replacing promoted or
  # refreshed artifacts. `cp -n` also avoids clobbering if two startup
  # processes race to seed the same newly mounted volume.
  cp -n "$SRC" "$DST"
  if [ ! -f "$DST" ]; then
    echo "[start.sh] failed to seed $DST" >&2
    return 1
  fi
  echo "[start.sh] seeded missing $DST"
}

# OUTPUT_DIR / DATA_DIR are set in the Dockerfile (default /srv/outputs and
# /srv/data). BOOTSTRAP_DIR is overridable only to make this bounded seeding
# behavior testable outside the container.
BOOTSTRAP_ROOT="${BOOTSTRAP_DIR:-/app/bootstrap}"
OUTPUT_ROOT="${OUTPUT_DIR:-/srv/outputs}"
DATA_ROOT="${DATA_DIR:-/srv/data}"

OUTPUT_MODEL="$OUTPUT_ROOT/models/lgbm_fire_date_model.pkl"
OUTPUT_FEATURES="$OUTPUT_ROOT/features/full_features.parquet"
OUTPUT_METADATA="$OUTPUT_ROOT/metadata/dataset_info.json"
OUTPUT_RISKMAP="$OUTPUT_ROOT/riskmap/fire_dates_all.geojson"

# These four files form one promoted artifact set. Filling an individual gap
# beside an existing file could pair thresholds/features with another model
# while still looking plausible to an operator.
OUTPUT_BUNDLE_EXISTING=0
for artifact in \
  "$OUTPUT_MODEL" "$OUTPUT_FEATURES" "$OUTPUT_METADATA" "$OUTPUT_RISKMAP"
do
  if [ -e "$artifact" ] && [ ! -f "$artifact" ]; then
    echo "[start.sh] invalid output bundle entry (not a file): $artifact" >&2
    exit 1
  fi
  if [ -f "$artifact" ]; then
    OUTPUT_BUNDLE_EXISTING=$((OUTPUT_BUNDLE_EXISTING + 1))
  fi
done

if [ "$OUTPUT_BUNDLE_EXISTING" -eq 0 ]; then
  seed_missing_file "$BOOTSTRAP_ROOT/outputs/models/lgbm_fire_date_model.pkl" \
    "$OUTPUT_MODEL"
  seed_missing_file "$BOOTSTRAP_ROOT/outputs/features/full_features.parquet" \
    "$OUTPUT_FEATURES"
  seed_missing_file "$BOOTSTRAP_ROOT/outputs/metadata/dataset_info.json" \
    "$OUTPUT_METADATA"
  seed_missing_file "$BOOTSTRAP_ROOT/outputs/riskmap/fire_dates_all.geojson" \
    "$OUTPUT_RISKMAP"
elif [ "$OUTPUT_BUNDLE_EXISTING" -ne 4 ]; then
  echo "[start.sh] partial output bundle ($OUTPUT_BUNDLE_EXISTING/4 files); refusing to mix bootstrap and volume artifact versions" >&2
  echo "[start.sh] restore or remove the entire promoted output bundle before restarting" >&2
  exit 1
else
  echo "[start.sh] complete promoted output bundle present — preserving it"
fi

seed_missing_file "$BOOTSTRAP_ROOT/data/boundaries/thailand.geojson" \
  "$DATA_ROOT/boundaries/thailand.geojson"
seed_missing_file "$BOOTSTRAP_ROOT/data/static/tree_cover_per_cell.parquet" \
  "$DATA_ROOT/static/tree_cover_per_cell.parquet"

# Runtime fetchers expect these cache directories even though a clean source
# checkout intentionally does not embed local raw/cache contents.
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/firms" "$DATA_ROOT/weather"

# If a startCommand was provided (cron service), exec it. Otherwise run the
# web service's default uvicorn invocation.
if [ $# -gt 0 ]; then
  exec "$@"
fi

cd /app/src
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
