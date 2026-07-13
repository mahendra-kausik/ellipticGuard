"""DVC stage entrypoint (Layer 3): train LR + RF baselines, log to MLflow, register RF as v1.

Run from the repo root: `python pipelines/train_baseline.py`
"""
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.data.split import make_temporal_split
from src.models.baseline import (
    build_xy,
    evaluate,
    fit_preprocessor,
    train_logistic_regression,
    train_random_forest,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"

# MLflow 3.x has put the plain filesystem store ('./mlruns') into maintenance mode —
# it raises unless MLFLOW_ALLOW_FILE_STORE is set. A local sqlite file is still a free,
# local, no-server backend, so that's the default here instead (see DECISIONS.md D-020).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "elliptic-aml")
REGISTRY_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "elliptic-illicit")


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params = params["split"]
    model_params = params["baseline"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    split = make_temporal_split(
        nodes_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )

    X_train_raw, y_train = build_xy(split.train)
    X_test_raw, y_test = build_xy(split.test)
    print(f"labeled train: n={len(y_train)}, illicit={int(y_train.sum())} ({y_train.mean():.1%})")
    print(f"labeled test:  n={len(y_test)}, illicit={int(y_test.sum())} ({y_test.mean():.1%})")

    prep = fit_preprocessor(X_train_raw)
    X_train = prep.transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}
    with mlflow.start_run(run_name="logistic_regression"):
        lr = train_logistic_regression(
            X_train, y_train, model_params["random_state"], model_params["lr_max_iter"]
        )
        mlflow.log_params({"model": "logistic_regression", "max_iter": model_params["lr_max_iter"]})
        metrics = evaluate(lr, X_test, y_test)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        mlflow.sklearn.log_model(lr, "model")
        results["logistic_regression"] = metrics
        print(f"LR  test metrics: {metrics}")

    with mlflow.start_run(run_name="random_forest") as rf_run:
        rf = train_random_forest(
            X_train,
            y_train,
            model_params["random_state"],
            model_params["rf_n_estimators"],
            model_params["rf_max_depth"],
        )
        mlflow.log_params(
            {
                "model": "random_forest",
                "n_estimators": model_params["rf_n_estimators"],
                "max_depth": model_params["rf_max_depth"],
            }
        )
        metrics = evaluate(rf, X_test, y_test)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        model_info = mlflow.sklearn.log_model(rf, "model")
        results["random_forest"] = metrics
        print(f"RF  test metrics: {metrics}")

        registered = mlflow.register_model(model_info.model_uri, REGISTRY_NAME)
        print(f"Registered {REGISTRY_NAME} v{registered.version} from run {rf_run.info.run_id}")

    metrics_path = PROCESSED_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
