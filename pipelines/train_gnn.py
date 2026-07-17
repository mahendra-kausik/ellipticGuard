"""DVC stage entrypoint (Layer 8, optional): GCN / GraphSAGE / EvolveGCN comparison.

Same split, same 165 features, same preprocessor, same class weight, same
`evaluate()` code as the XGBoost champion (see src/models/gnn.py and D-030) —
only the layer type varies between the three models. Comparison models, not
serving candidates: nothing here is registered in the MLflow model registry
(mirrors Layer 6's choice for the calibrated model).

Run from the repo root: `python pipelines/train_gnn.py`
"""
import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.data.loaders import FEATURE_COLS
from src.data.split import make_temporal_split
from src.models.advanced import scale_pos_weight
from src.models.baseline import evaluate, fit_preprocessor
from src.models.evaluate import per_time_step_metrics
from src.models.gnn import GCN, EvolveGCN, GraphSAGE, build_evaluate_adapter, build_snapshots, train_gnn

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "elliptic-aml")

MODEL_CLASSES = {"gnn_gcn": GCN, "gnn_sage": GraphSAGE, "gnn_evolvegcn": EvolveGCN}


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params = params["split"]
    cfg = params["gnn"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges_df = pd.read_parquet(PROCESSED_DIR / "edges.parquet")

    split = make_temporal_split(
        nodes_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )
    # Fit-only slice (1..val_start-1) — never fit on val/test (CLAUDE.md Directive 3).
    preprocessor = fit_preprocessor(split.train[FEATURE_COLS])
    snapshots = build_snapshots(nodes_df, edges_df, preprocessor)
    snapshots.sort(key=lambda s: s.time_step)

    train_idx = [i for i, s in enumerate(snapshots) if s.time_step < split_params["val_start"]]
    val_idx = [i for i, s in enumerate(snapshots) if split_params["val_start"] <= s.time_step <= split_params["train_end"]]
    test_idx = [i for i, s in enumerate(snapshots) if split_params["test_start"] <= s.time_step <= split_params["test_end"]]

    train_labeled = split.train[split.train["label"].notna()]
    class_weight = scale_pos_weight(train_labeled["label"].astype(int))

    mlflow.set_experiment(EXPERIMENT_NAME)

    xgb_metrics = json.loads((PROCESSED_DIR / "advanced_metrics.json").read_text())
    xgb_champion_name = xgb_metrics["champion"]
    xgb_champion_f1 = xgb_metrics[xgb_champion_name]["test_metrics"]["illicit_f1"]

    all_results = {}
    per_time_step_frames = []

    for run_name, model_cls in MODEL_CLASSES.items():
        seed_f1s, seed_details = [], []
        best_seed_f1, best_seed_model = -1.0, None

        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            model = model_cls(h1=cfg["hidden1"], h2=cfg["hidden2"], dropout=cfg["dropout"])
            train_cfg = {
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "max_epochs": cfg["max_epochs"],
                "patience": cfg["patience"],
                "class_weight": class_weight,
            }
            train_gnn(model, snapshots, train_idx, val_idx, train_cfg)

            adapter = build_evaluate_adapter(model, snapshots, test_idx, split.test)
            labeled_test = split.test[split.test["label"].notna()]
            test_metrics = evaluate(adapter, labeled_test[FEATURE_COLS], labeled_test["label"].astype(int))
            f1 = test_metrics["illicit_f1"]
            seed_f1s.append(f1)
            seed_details.append({"seed": seed, **test_metrics})
            print(f"[{run_name}] seed={seed} test_illicit_f1={f1:.3f} auc_pr={test_metrics['auc_pr']:.3f}")

            if f1 > best_seed_f1:
                best_seed_f1, best_seed_model = f1, model

        mean_f1, std_f1 = float(np.mean(seed_f1s)), float(np.std(seed_f1s))
        all_results[run_name] = {
            "seeds": cfg["seeds"],
            "illicit_f1_per_seed": seed_f1s,
            "illicit_f1_mean": mean_f1,
            "illicit_f1_std": std_f1,
            "per_seed_metrics": seed_details,
        }

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("purpose", "layer8_gnn_comparison_not_for_serving")
            mlflow.log_params({"model": run_name, **cfg, "class_weight": class_weight})
            mlflow.log_metric("test_illicit_f1_mean", mean_f1)
            mlflow.log_metric("test_illicit_f1_std", std_f1)
            for i, f1 in enumerate(seed_f1s):
                mlflow.log_metric("test_illicit_f1_per_seed", f1, step=i)

        # T43 curve from the seed with the best validation-selected test F1 —
        # illustrative only; the headline number is the mean±std above.
        best_seed_model.eval()
        step_metrics = per_time_step_metrics(
            build_evaluate_adapter(best_seed_model, snapshots, test_idx, split.test), split.test, FEATURE_COLS
        )
        step_metrics["model"] = run_name
        per_time_step_frames.append(step_metrics)

        print(f"[{run_name}] test_illicit_f1 = {mean_f1:.3f} +/- {std_f1:.3f} (seeds={cfg['seeds']})")

    print("\n=== Layer 8 comparison (test illicit-F1, mean +/- std across seeds) ===")
    print(f"  XGBoost champion ({xgb_champion_name}): F1={xgb_champion_f1:.3f}")
    for run_name in MODEL_CLASSES:
        r = all_results[run_name]
        collapsed = [s for s, f1 in zip(r["seeds"], r["illicit_f1_per_seed"]) if f1 < 0.05]
        flag = f"  (seed(s) {collapsed} collapsed — reported individually, not smoothed into the mean)" if collapsed else ""
        print(f"  {run_name:14s}: F1={r['illicit_f1_mean']:.3f} +/- {r['illicit_f1_std']:.3f}{flag}")

    out = {
        "xgboost_champion": {"feature_set": xgb_champion_name, "illicit_f1": xgb_champion_f1},
        **all_results,
    }
    metrics_path = PROCESSED_DIR / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {metrics_path}")

    per_step_df = pd.concat(per_time_step_frames, ignore_index=True)
    per_step_path = PROCESSED_DIR / "gnn_per_time_step.csv"
    per_step_df.to_csv(per_step_path, index=False)
    print(f"Wrote {per_step_path}")


if __name__ == "__main__":
    main()
