"""DVC stage entrypoint (Layer 5): XGBoost champion + provided-vs-+graph experiments.

Run from the repo root: `python pipelines/train_advanced.py`

Tuning happens on val (steps 30-34) only. Once params are chosen, each feature
set's model is refit on train+val (steps 1-34) and evaluated once on test
(steps 35-49) — the gate comparison. See CLAUDE.md Directive 3 and the Layer 5
plan for why this order is leakage-safe.
"""
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.data.split import make_temporal_split
from src.models.advanced import (
    ALL_FEATURE_COLS,
    PROVIDED_FEATURE_COLS,
    build_xy_features,
    merge_graph_features,
    scale_pos_weight,
    train_xgb,
    tune_on_val,
)
from src.models.baseline import evaluate

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "elliptic-aml")
REGISTRY_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "elliptic-illicit")

FEATURE_SETS = {"provided", "provided_plus_graph"}


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params = params["split"]
    adv_params = params["advanced"]
    random_state = adv_params["random_state"]
    grid = adv_params["grid"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    graph_df = pd.read_parquet(PROCESSED_DIR / "graph_features.parquet")
    merged_df = merge_graph_features(nodes_df, graph_df)

    split = make_temporal_split(
        merged_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )
    # train+val = steps 1..train_end, for the final champion refit only.
    train_val_df = merged_df[merged_df["time_step"] <= split_params["train_end"]]

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}
    feature_col_map = {"provided": PROVIDED_FEATURE_COLS, "provided_plus_graph": ALL_FEATURE_COLS}

    for name, feature_cols in feature_col_map.items():
        X_train, y_train = build_xy_features(split.train, feature_cols)
        X_val, y_val = build_xy_features(split.val, feature_cols)
        X_trainval, y_trainval = build_xy_features(train_val_df, feature_cols)
        X_test, y_test = build_xy_features(split.test, feature_cols)

        spw = scale_pos_weight(y_train)
        best_params, val_f1 = tune_on_val(X_train, y_train, X_val, y_val, grid, spw, random_state)

        # Refit champion-candidate on train+val with the chosen params (D-022).
        spw_trainval = scale_pos_weight(y_trainval)
        model = train_xgb(X_trainval, y_trainval, best_params, spw_trainval, random_state)
        test_metrics = evaluate(model, X_test, y_test)

        with mlflow.start_run(run_name=f"xgb_{name}") as run:
            mlflow.set_tag("feature_set", name)
            mlflow.log_params({"model": "xgboost", "feature_set": name, **best_params})
            mlflow.log_metric("val_illicit_f1", val_f1)
            mlflow.log_metrics({k: v for k, v in test_metrics.items() if k != "confusion_matrix"})
            model_info = mlflow.xgboost.log_model(model, "model")

        results[name] = {
            "best_params": best_params,
            "val_illicit_f1": val_f1,
            "test_metrics": test_metrics,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }
        print(f"[{name}] best_params={best_params} val_f1={val_f1:.3f} test={test_metrics}")

    # Champion = higher test illicit-F1 across feature sets.
    champion_name = max(results, key=lambda n: results[n]["test_metrics"]["illicit_f1"])
    champion = results[champion_name]
    registered = mlflow.register_model(champion["model_uri"], REGISTRY_NAME)
    client = MlflowClient()
    client.set_model_version_tag(REGISTRY_NAME, registered.version, "serving_candidate", "true")
    client.set_model_version_tag(REGISTRY_NAME, registered.version, "feature_set", champion_name)
    print(f"Registered {REGISTRY_NAME} v{registered.version} (champion={champion_name}, serving_candidate=true)")

    # Baseline (Layer 3) numbers, for the side-by-side gate comparison.
    baseline_metrics = json.loads((PROCESSED_DIR / "baseline_metrics.json").read_text())
    rf_v1 = baseline_metrics["random_forest"]

    print("\n=== Layer 5 comparison (test illicit-F1 / AUC-PR) ===")
    print(f"  RF v1 (provided, Layer 3):        F1={rf_v1['illicit_f1']:.3f}  AUC-PR={rf_v1['auc_pr']:.3f}")
    for name in feature_col_map:
        m = results[name]["test_metrics"]
        print(f"  XGB {name:22s}: F1={m['illicit_f1']:.3f}  AUC-PR={m['auc_pr']:.3f}")
    print(f"  Champion: {champion_name} (v{registered.version})")

    out = {
        "baseline_rf_v1": rf_v1,
        **results,
        "champion": champion_name,
        "champion_version": registered.version,
    }
    metrics_path = PROCESSED_DIR / "advanced_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
