"""Score prospective forecasts after each snapshot's own outcome window matures."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
from src.forecasting import score_snapshots
from src.pipeline import load_observations, observation_paths


def main():
    load_dotenv(ROOT / ".env")
    output = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs")))
    observations, _ = load_observations(observation_paths(ROOT, os.getenv("RAW_DIR"), os.getenv("FIRMS_PATH")))
    end = min(observations.date.max(), pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1))
    report = score_snapshots(output / "forecasts", observations, end)
    (output / "evaluation").mkdir(exist_ok=True)
    (output / "evaluation/prospective.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
