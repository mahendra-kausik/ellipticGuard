"""DVC stage entrypoint (Layer 9): per-time-step feature/target drift vs the train reference.

Run from the repo root: `python pipelines/monitor_drift.py`
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import FEATURE_COLS
from src.data.split import make_temporal_split
from src.monitoring.drift import drift_by_time_step, save_report_html

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params, monitor_params = params["split"], params["monitor"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    split = make_temporal_split(
        nodes_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )

    print("=== Layer 9 drift-by-time-step (reference = train, steps 1-29) ===")
    drift_df = drift_by_time_step(nodes_df, split, FEATURE_COLS)
    drift_path = PROCESSED_DIR / "drift_by_time_step.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"Wrote {drift_path}")
    print(drift_df.to_string(index=False))

    report_step = monitor_params["report_step"]
    report_path = PROCESSED_DIR / "drift_report.html"
    save_report_html(nodes_df, split, FEATURE_COLS, report_step, report_path)
    print(f"Wrote {report_path} (step {report_step})")


if __name__ == "__main__":
    main()
