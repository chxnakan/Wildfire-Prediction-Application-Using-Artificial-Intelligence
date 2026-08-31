#!/usr/bin/env bash
# =====================================================================
# 🔥 FireWatch Thailand — one-shot runner
#
# Source-backed operation. Missing feeds stay visibly missing; derived
# aggregates and reanalysis fields are labelled rather than passed off as raw.
#
# Default behaviour:
#   1. Train on whatever real FIRMS data is already in data/raw + data/firms/
#   2. Refresh outputs/riskmap/fire_dates_all.geojson (real model output)
#   3. Serve the FastAPI (which also serves the built React SPA at /) on
#                                      http://localhost:8000/
#
# Optional flags fetch the latest real hotspots and/or real ERA5 weather
# before training. Skip training entirely with --no-train.
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FETCH_FIRMS=0
FETCH_WEATHER=0
FETCH_GISTDA=0
DO_TRAIN=1
PREDICT_ONLY=0
OPEN_BROWSER=0
DEMO_MODE=0
WEATHER_ARGS=()
TRAIN_ARGS=()
API_PORT=8000

usage() {
  cat <<EOF
Usage: $0 [flags] [-- <extra args forwarded to train.py>]

Real-data pipeline orchestrator. Runs train.py (which triggers risk_map.run()
inside it) and then serves both the dashboard and the FastAPI.

Flags:
  --fresh                Fetch latest NASA FIRMS VIIRS NRT hotspots first
                         (requires FIRMS_API_KEY in .env). Falls back to the
                         cached data if the fetch fails.
  --weather              Run fetch_weather.py to pull real ERA5 daily
                         aggregates from Open-Meteo (no API key) before
                         training so weather features are included.
  --gistda               Run fetch_gistda_hotspots.py (GISTDA ArcGIS REST,
                         no key) → data/gistda/gistda_hotspots.parquet before
                         training. Not yet merged into model features — cache only.
  --weather-arg <ARG>    Pass an extra arg to fetch_weather.py (repeatable).
                         e.g. --weather-arg --limit-cells --weather-arg 50
  --no-train             Skip training; just start the servers using whatever
                         is already in outputs/.
  --demo                 Presentation-safe mode: serve the promoted artifacts
                         without FIRMS refresh, prediction, or training. Keeps
                         only the live GISTDA scheduler job enabled.
  --predict-only         Run train.py --predict-only: rebuild full_features from
                         latest FIRMS (+ weather cache if present), refresh GeoJSON,
                         leave the existing model .pkl unchanged. Overrides --no-train
                         for the train step. Pair with --fresh / --weather for daily ops.
  --open                 Open the dashboard in the default browser when ready.
  --quick                Shortcut: forward --quick to train.py (~8–15 min run,
                         LightGBM only). Useful for iteration.
  --fast                 Forward --fast to train.py: LightGBM only, 35×3 CV
                         (faster than default two-booster 100×5, good overnight-lite).
  --api-port  N          Port for FastAPI (also serves the SPA, default 8000).
  -h, --help             Show this message.

Anything after a literal "--" is forwarded verbatim to train.py, e.g.:
  $0 --fresh -- --n-iter 30 --only lightgbm,xgboost

Real data sources:
  • NASA FIRMS VIIRS NRT  (always)
  • Open-Meteo ERA5        (only if --weather is used or cache exists)
  • GISTDA hotspots        (optional --gistda; cache only until wired to training)
EOF
}

# ── Parse flags ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)        FETCH_FIRMS=1; shift ;;
    --weather)      FETCH_WEATHER=1; shift ;;
    --gistda)       FETCH_GISTDA=1; shift ;;
    --weather-arg)  WEATHER_ARGS+=("$2"); shift 2 ;;
    --no-train)     DO_TRAIN=0; shift ;;
    --demo)         DEMO_MODE=1; shift ;;
    --predict-only) PREDICT_ONLY=1; shift ;;
    --open)         OPEN_BROWSER=1; shift ;;
    --quick)        TRAIN_ARGS+=("--quick"); shift ;;
    --fast)         TRAIN_ARGS+=("--fast"); shift ;;
    --api-port)     API_PORT="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; TRAIN_ARGS+=("$@"); break ;;
    *)              echo "Unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Presentation mode is intentionally immutable with respect to model inputs and
# artifacts. Fail loudly if it is combined with a pipeline flag rather than
# silently doing a partial refresh that an operator did not expect.
if [[ $DEMO_MODE -eq 1 ]]; then
  if [[ $FETCH_FIRMS -eq 1 || $FETCH_WEATHER -eq 1 || $FETCH_GISTDA -eq 1 \
        || $PREDICT_ONLY -eq 1 || ${#WEATHER_ARGS[@]} -gt 0 \
        || ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    cat <<'EOF' >&2
❌ --demo cannot be combined with data-fetch, prediction, or training flags.
   Run those jobs before the presentation, then start with: ./run.sh --demo
EOF
    exit 2
  fi
  DO_TRAIN=0
  # Shell variables override .env because python-dotenv does not replace an
  # already-exported value. Keep the scheduler alive solely for GISTDA.
  export SCHEDULER_ENABLED=true
  export SCHEDULER_SYNC_ENABLED=false
  export SCHEDULER_RETRAIN_ENABLED=false
  export GISTDA_ENABLED=true
  echo "→ DEMO mode: model artifacts locked; only live GISTDA polling is enabled."
fi

# ── Activate virtualenv if present ──────────────────────────────────────
if [[ -d ".venv" ]]; then
  echo "→ Activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Sanity-check Python deps without installing anything (no surprise side effects)
if ! python -c "import fastapi, uvicorn, pandas, sklearn, joblib" >/dev/null 2>&1; then
  cat <<'EOF' >&2
❌ Missing Python deps. Run once:
     python -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt
EOF
  exit 1
fi

# ── Pre-flight: how stale is the FIRMS cache? ────────────────────────────
# If the most recent hotspot date in data/firms/firms_all.{parquet,csv} is
# more than STALE_DAYS old, auto-promote --fresh so we pull new data before
# training. This keeps daily runs from silently using week-old hotspots.
STALE_DAYS="${STALE_DAYS:-2}"
STALENESS=$(python <<PY 2>/dev/null || echo "unknown"
import os, sys
sys.path.insert(0, "src")
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from io_utils import resolve_existing
load_dotenv(".env")
data_dir = os.getenv("DATA_DIR", "./data")
p = resolve_existing(os.getenv(
    "FIRMS_PATH", os.path.join(data_dir, "firms", "firms_all.parquet")
))
if not p:
    print("missing"); sys.exit(0)
df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
if "acq_datetime" not in df.columns or df.empty:
    print("missing"); sys.exit(0)
latest = pd.to_datetime(df["acq_datetime"]).max().date()
lag = (date.today() - latest).days
print(f"{latest}|{lag}")
PY
)

if [[ "$STALENESS" == "missing" ]]; then
  if [[ $DEMO_MODE -eq 1 ]]; then
    echo "  ⚠️  No FIRMS cache found; DEMO mode will not fetch or change model inputs."
  else
    echo "→ No FIRMS cache found — promoting --fresh so we fetch initial data."
    FETCH_FIRMS=1
  fi
elif [[ "$STALENESS" == "unknown" ]]; then
  echo "  ⚠️  Could not determine FIRMS cache freshness; continuing without auto-refresh."
else
  LATEST_DATE="${STALENESS%|*}"
  LAG_DAYS="${STALENESS#*|}"
  if [[ "${LAG_DAYS}" -le 1 ]]; then
    echo "→ FIRMS cache: latest complete UTC day=${LATEST_DATE} (current for complete-day scoring)"
  else
    echo "→ FIRMS cache: latest=${LATEST_DATE} (${LAG_DAYS} day(s) behind today)"
  fi
  if [[ "${LAG_DAYS}" -gt "${STALE_DAYS}" && $FETCH_FIRMS -eq 0 ]]; then
    if [[ $DEMO_MODE -eq 1 ]]; then
      echo "  ℹ️  DEMO mode keeps the promoted snapshot; no FIRMS refresh is started."
    else
      echo "  ⚠️  Data is >${STALE_DAYS} days stale → promoting --fresh."
      echo "      Set STALE_DAYS=999 to disable, or run with --fresh explicitly."
      FETCH_FIRMS=1
    fi
  fi
fi

# ── Stage 1: optional FIRMS fetch (real NASA data) ──────────────────────
if [[ $FETCH_FIRMS -eq 1 ]]; then
  echo "→ Fetching latest FIRMS VIIRS NRT hotspots …"
  if ! (cd src && python fetch_firms.py); then
    echo "  ⚠️  FIRMS fetch failed (likely missing/exhausted FIRMS_API_KEY)."
    echo "      Continuing with whatever real data is already cached."
  fi
fi

# ── Stage 2: optional ERA5 weather fetch (real Open-Meteo data) ─────────
if [[ $FETCH_WEATHER -eq 1 ]]; then
  echo "→ Fetching real ERA5 daily weather (Open-Meteo Archive) …"
  (cd src && python fetch_weather.py "${WEATHER_ARGS[@]}")
fi

if [[ $FETCH_GISTDA -eq 1 ]]; then
  echo "→ Fetching GISTDA hotspots (no API key) …"
  if ! (cd src && python fetch_gistda_hotspots.py); then
    echo "  ⚠️  GISTDA fetch failed — continuing (see docs/DATA_APIS_TH.md)."
  fi
fi

# ── Stage 3: full train OR predict-only refresh (features + risk map) ───
if [[ $PREDICT_ONLY -eq 1 ]]; then
  echo "→ predict-only: refresh features + risk map (no model tuning) …"
  # Memory guard: feature engineering on the full 4.4M-row densified frame
  # peaks ~18 GB in pandas, which OOM-kills predict-only on a 22 GB laptop
  # with a browser/IDE open. Default the predict window to the last 180 days
  # (~1.6 M rows, peak ~7 GB) unless the user explicitly overrode it.
  EXTRA_TRAIN_ARGS=()
  if ! printf '%s\n' "${TRAIN_ARGS[@]}" | grep -q -- "--max-history-days"; then
    if [[ -z "${MAX_TRAIN_HISTORY_DAYS:-}" || "${MAX_TRAIN_HISTORY_DAYS:-0}" == "0" ]]; then
      EXTRA_TRAIN_ARGS+=("--max-history-days" "180")
      echo "   (memory guard: --max-history-days 180  · override with --max-history-days N)"
    fi
  fi
  (cd src && python train.py --predict-only "${EXTRA_TRAIN_ARGS[@]}" "${TRAIN_ARGS[@]}")
elif [[ $DO_TRAIN -eq 1 ]]; then
  echo "→ Training on real FIRMS data (+ ERA5 weather if cached) …"
  (cd src && python train.py "${TRAIN_ARGS[@]}")
fi

# ── Sanity check artefacts before serving ───────────────────────────────
# Resolve OUTPUT_DIR exactly as the Python runtime does. Previously the
# pipeline could write to an env-configured directory while this preflight
# silently checked ./outputs and then served a different/stale snapshot.
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  # An explicitly exported value wins over .env, matching python-dotenv.
  FIREWATCH_OUTPUT_DIR="$OUTPUT_DIR"
else
  FIREWATCH_OUTPUT_DIR=$(python <<'PY'
# FIREWATCH_OUTPUT_DIR_LOOKUP (also used by the hermetic shell regression test)
import os
from dotenv import load_dotenv

load_dotenv(".env")
print(os.getenv("OUTPUT_DIR", "./outputs"))
PY
)
fi
if [[ "$FIREWATCH_OUTPUT_DIR" != /* ]]; then
  FIREWATCH_OUTPUT_DIR="$SCRIPT_DIR/${FIREWATCH_OUTPUT_DIR#./}"
fi
FIREWATCH_MODEL_PATH="$FIREWATCH_OUTPUT_DIR/models/lgbm_fire_date_model.pkl"
FIREWATCH_FEATURE_PARQUET="$FIREWATCH_OUTPUT_DIR/features/full_features.parquet"
FIREWATCH_FEATURE_CSV="$FIREWATCH_OUTPUT_DIR/features/full_features.csv"
FIREWATCH_METADATA_PATH="$FIREWATCH_OUTPUT_DIR/metadata/dataset_info.json"
FIREWATCH_GEOJSON_PATH="$FIREWATCH_OUTPUT_DIR/riskmap/fire_dates_all.geojson"

if [[ ! -f "$FIREWATCH_MODEL_PATH" ]]; then
  cat <<'EOF' >&2
❌ No trained model in the configured OUTPUT_DIR.
   Run without --no-train, or place a trained artefact there first.
EOF
  exit 1
fi

# Model inference also requires the feature frame. The old check only required
# the pickle, so --demo/--no-train could print "Ready" while every prediction
# endpoint was actually degraded (HTTP 503).
if [[ ! -f "$FIREWATCH_FEATURE_PARQUET" && ! -f "$FIREWATCH_FEATURE_CSV" ]]; then
  cat <<EOF >&2
❌ No serving feature frame in the configured OUTPUT_DIR.
   Expected: $FIREWATCH_FEATURE_PARQUET (or .csv)
EOF
  exit 1
fi

if [[ ! -f "$FIREWATCH_GEOJSON_PATH" ]]; then
  if [[ $DEMO_MODE -eq 1 ]]; then
    cat <<EOF >&2
❌ DEMO mode requires a complete promoted snapshot, but GeoJSON is missing:
   $FIREWATCH_GEOJSON_PATH
   Presentation mode will not generate or modify model artifacts.
EOF
    exit 1
  fi
  echo "→ GeoJSON missing; running risk_map.py to generate it from real data …"
  (cd src && python risk_map.py)
  if [[ ! -f "$FIREWATCH_GEOJSON_PATH" ]]; then
    echo "❌ risk_map.py completed without creating $FIREWATCH_GEOJSON_PATH" >&2
    exit 1
  fi
fi

# /health/ready requires metadata. A normal operational run may intentionally
# boot degraded while automation repairs it, but a presentation must fail
# before claiming that an incomplete snapshot is ready.
if [[ ! -f "$FIREWATCH_METADATA_PATH" ]]; then
  if [[ $DEMO_MODE -eq 1 ]]; then
    echo "❌ DEMO mode requires metadata at $FIREWATCH_METADATA_PATH" >&2
    exit 1
  fi
  echo "  ⚠️  Metadata missing at $FIREWATCH_METADATA_PATH; API readiness will be degraded."
fi

# ── Stage 4: serve the FastAPI (which serves the built React SPA at /) ─────
# The legacy /frontend/ static-server step is gone; the React app lives in
# /web and is built into /web/dist, which FastAPI mounts at root via
# src/api.py's StaticFiles fallback. One process serves the dashboard +
# the API on the same origin.
API_URL="http://localhost:${API_PORT}/"

cleanup() {
  echo
  echo "→ Shutting down API …"
}
trap cleanup EXIT INT TERM

WEB_BUILD_NEEDED=0
if [[ ! -f "web/dist/index.html" ]]; then
  WEB_BUILD_NEEDED=1
  echo "→ React dashboard build is missing."
elif find \
    web/src \
    web/index.html \
    web/package.json \
    web/package-lock.json \
    web/tsconfig.json \
    web/vite.config.ts \
    -type f -newer web/dist/index.html -print -quit | grep -q .; then
  WEB_BUILD_NEEDED=1
  echo "→ React source changed since the last build."
fi

if [[ $WEB_BUILD_NEEDED -eq 1 ]]; then
  echo "→ Building the current React dashboard (web/) …"
  if [[ ! -d "web/node_modules" ]]; then
    (cd web && npm ci --no-audit --no-fund)
  fi
  (cd web && npm run build)
fi

if [[ $OPEN_BROWSER -eq 1 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    (sleep 1 && xdg-open "${API_URL}") &
  elif command -v open >/dev/null 2>&1; then
    (sleep 1 && open "${API_URL}") &
  fi
fi

cat <<EOF

✅ Ready.
   Dashboard:  ${API_URL}
   API:        ${API_URL}
   API docs:   ${API_URL}docs

   Displayed evidence comes from persisted, named data sources:
     • NASA FIRMS VIIRS NRT hotspots
     • Open-Meteo / ERA5 reanalysis (if cached)
     • Documented temporal/spatial feature aggregates
   Missing live feeds are not replaced with fabricated observations.

   Press Ctrl+C to stop both servers.

EOF

# Foreground FastAPI — Ctrl+C tears down the HTTP server via the trap.
(cd src && exec python -m uvicorn api:app --host 0.0.0.0 --port "${API_PORT}")
