"""DVC stage entrypoint (Layer 10): replay driver + drift-or-performance flag +
champion/challenger promotion gate.

Explicit simulation, not a live feed (PROJECT_PLAN.md §1): streams the fixed
test-range time steps in order. The champion is the Layer 6 train-only model
(steps 1-29, via `build_train_only_champion`) — the same model whose T43
collapse Layer 6 already documented (per-step illicit-F1 0.855 mean pre-T43 to
0.028 post-T43). Reusing it ties this replay's story directly to that prior
evidence instead of inventing a new artificial baseline. See D-026.

Run from the repo root: `python pipelines/replay_retraining.py`
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import FEATURE_COLS
from src.data.split import make_temporal_split
from src.models.advanced import PROVIDED_FEATURE_COLS, build_xy_features
from src.models.baseline import evaluate
from src.models.evaluate import build_train_only_champion
from src.monitoring.drift import drift_by_time_step
from src.retraining.replay import run_replay

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
METRICS_DIR = REPO_ROOT / "metrics"
PARAMS_PATH = REPO_ROOT / "params.yaml"


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params = params["split"]
    retrain_params = params["retrain"]
    random_state = params["advanced"]["random_state"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    split = make_temporal_split(
        nodes_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )

    # Champion's XGB hyperparams — reuse the Layer 5 tuned "provided" champion's
    # params rather than re-tuning (this stage isn't a tuning stage).
    advanced_metrics = json.loads((PROCESSED_DIR / "advanced_metrics.json").read_text())
    champion_params = advanced_metrics["provided"]["best_params"]

    champion = build_train_only_champion(split.train, PROVIDED_FEATURE_COLS, champion_params, random_state)
    champion_test_f1 = evaluate(champion, *build_xy_features(split.test, PROVIDED_FEATURE_COLS))["illicit_f1"]
    print(f"Replay champion (Layer 6, steps 1-{split_params['val_start'] - 1}) test illicit-F1: {champion_test_f1:.3f}")

    # Target-drift score per test-range step (reference = split.train, matches Layer 9).
    drift_df = drift_by_time_step(nodes_df, split, FEATURE_COLS)
    drift_by_step = dict(zip(drift_df["time_step"], drift_df["target_drift_score"]))

    print(f"=== Layer 10 replay: steps {split_params['test_start']}-{split_params['test_end']} ===")
    log = run_replay(
        nodes_df,
        PROVIDED_FEATURE_COLS,
        start_step=split_params["test_start"],
        end_step=split_params["test_end"],
        champion=champion,
        drift_by_step=drift_by_step,
        f1_floor=retrain_params["f1_floor"],
        drift_ceiling=retrain_params["drift_ceiling"],
        params=champion_params,
        random_state=random_state,
        promote_margin=retrain_params["promote_margin"],
    )
    log_path = PROCESSED_DIR / "replay_log.csv"
    log.to_csv(log_path, index=False)
    print(f"Wrote {log_path}")
    print(log.to_string(index=False))

    first_flag_step = log.loc[log["flag_fired"], "time_step"].min()
    promote_rows = log[log["action"] == "promote"]
    hold_rows = log[log["action"] == "hold"]
    print(f"\nFirst flag-fire step: {first_flag_step}")
    print(f"Promote steps: {list(promote_rows['time_step'])}")
    print(f"Hold steps: {list(hold_rows['time_step'])}")

    # Committed-to-git (not DVC-tracked) quality-gate snapshot — lets CI check
    # for a champion-quality regression without needing the git-ignored data/.
    # Uses the REGISTERED (v2, train+val-fit) champion's test illicit-F1 from
    # Layer 5 — the real serving-quality number — not this stage's train-only
    # replay champion (whose F1 is logged above for replay context only; it's
    # intentionally lower since val is held out from its fit, per Layer 6/D-023).
    METRICS_DIR.mkdir(exist_ok=True)
    registered_champion_f1 = advanced_metrics["provided"]["test_metrics"]["illicit_f1"]
    quality_gate = {
        "champion_test_illicit_f1": registered_champion_f1,
        "replay_champion_test_illicit_f1": champion_test_f1,
        "first_flag_fire_step": None if pd.isna(first_flag_step) else int(first_flag_step),
        "n_promote_steps": int(len(promote_rows)),
        "n_hold_steps": int(len(hold_rows)),
    }
    gate_path = METRICS_DIR / "quality_gate.json"
    gate_path.write_text(json.dumps(quality_gate, indent=2))
    print(f"Wrote {gate_path}")


if __name__ == "__main__":
    main()
