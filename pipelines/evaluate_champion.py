"""DVC stage entrypoint (Layer 6): per-time-step F1 curve, calibration, lean SHAP.

Run from the repo root: `python pipelines/evaluate_champion.py`

Uses the champion feature set + tuned params from Layer 5's
`advanced_metrics.json`, but refits on train-only (steps 1-29) so val
(30-34) stays a clean calibration holdout. Test (35-49) is touched only for
final scoring (per-step F1 curve, Brier). The calibrated model is logged to
MLflow but NOT registered — Layer 7 decides what to serve (see D-023).
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
from src.models.advanced import PROVIDED_FEATURE_COLS, ALL_FEATURE_COLS, build_xy_features
from src.models.baseline import evaluate
from src.models.evaluate import (
    brier,
    build_train_only_champion,
    calibrate_on_val,
    per_time_step_metrics,
    top_shap_features,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "elliptic-aml")

T43_ANNOTATION_STEP = 43

FEATURE_COL_MAP = {"provided": PROVIDED_FEATURE_COLS, "provided_plus_graph": ALL_FEATURE_COLS}


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    split_params = params["split"]
    adv_params = params["advanced"]
    eval_params = params["evaluate"]
    random_state = adv_params["random_state"]

    adv_metrics = json.loads((PROCESSED_DIR / "advanced_metrics.json").read_text())
    champion_name = adv_metrics["champion"]
    best_params = adv_metrics[champion_name]["best_params"]
    feature_cols = FEATURE_COL_MAP[champion_name]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    if champion_name == "provided_plus_graph":
        graph_df = pd.read_parquet(PROCESSED_DIR / "graph_features.parquet")
        from src.models.advanced import merge_graph_features
        nodes_df = merge_graph_features(nodes_df, graph_df)

    split = make_temporal_split(
        nodes_df,
        train_end=split_params["train_end"],
        val_start=split_params["val_start"],
        test_start=split_params["test_start"],
        test_end=split_params["test_end"],
    )

    # 1. Train-only base model (steps 1-29) — keeps val a clean calibration holdout.
    base_model = build_train_only_champion(split.train, feature_cols, best_params, random_state)

    # 2. Per-time-step F1 curve on test (35-49).
    per_step_df = per_time_step_metrics(base_model, split.test, feature_cols)
    per_step_path = PROCESSED_DIR / "per_time_step_f1.csv"
    per_step_df.to_csv(per_step_path, index=False)

    pre43 = per_step_df[per_step_df["time_step"] < T43_ANNOTATION_STEP]["illicit_f1"]
    post43 = per_step_df[per_step_df["time_step"] >= T43_ANNOTATION_STEP]["illicit_f1"]
    print("=== Per-time-step illicit-F1 (test range) ===")
    print(per_step_df.to_string(index=False))
    if not pre43.empty and not post43.empty:
        print(f"\nT43 annotation: mean F1 before T43 = {pre43.mean():.3f}, "
              f"mean F1 T43-onward = {post43.mean():.3f} "
              f"(delta = {post43.mean() - pre43.mean():+.3f})")

    # 3. Calibration on val (30-34, prefit) + test Brier for uncal vs each method.
    X_val, y_val = build_xy_features(split.val, feature_cols)
    X_test, y_test = build_xy_features(split.test, feature_cols)

    brier_scores = {"uncalibrated": brier(base_model, X_test, y_test)}
    calibrated_models = {}
    for method in eval_params["calibration_methods"]:
        cal_model = calibrate_on_val(base_model, X_val, y_val, method)
        calibrated_models[method] = cal_model
        brier_scores[method] = brier(cal_model, X_test, y_test)

    best_method = min(
        (m for m in eval_params["calibration_methods"]), key=lambda m: brier_scores[m]
    )
    best_calibrated = calibrated_models[best_method]
    print(f"\n=== Calibration (test Brier score, lower is better) ===")
    for name, score in brier_scores.items():
        print(f"  {name:14s}: {score:.4f}")
    print(f"  Chosen calibration method: {best_method}")

    # 4. Lean SHAP (guarded — never fails the stage).
    shap_sample = X_test.sample(
        n=min(eval_params["shap_sample"], len(X_test)), random_state=random_state
    )
    shap_df = top_shap_features(base_model, shap_sample, eval_params["shap_top_n"])
    if shap_df is not None:
        shap_path = PROCESSED_DIR / "shap_top_features.csv"
        shap_df.to_csv(shap_path, index=False)
        print(f"\n=== Top {eval_params['shap_top_n']} SHAP features (mean |value|) ===")
        print(shap_df.to_string(index=False))
    else:
        print("\nSHAP skipped (see warning above).")

    # 5. Log to MLflow — produce + log only, do NOT register (D-023).
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.set_tag("feature_set", champion_name)
        mlflow.set_tag("purpose", "layer6_evaluation_not_for_serving")
        mlflow.log_metrics({f"brier_{k}": v for k, v in brier_scores.items()})
        mlflow.log_param("chosen_calibration_method", best_method)
        mlflow.xgboost.log_model(base_model, "base_model_train_only")
        # skops (mlflow's default sklearn serializer) refuses to round-trip a
        # CalibratedClassifierCV wrapping an XGBClassifier (untrusted-type
        # allowlist); pickle is fine here since this is our own trusted
        # artifact, not an untrusted upload.
        mlflow.sklearn.log_model(
            best_calibrated, "calibrated_model", serialization_format="pickle"
        )
        mlflow.log_artifact(str(per_step_path))
        if shap_df is not None:
            mlflow.log_artifact(str(shap_path))

    out = {
        "champion_name": champion_name,
        "per_time_step_f1_path": str(per_step_path),
        "brier_scores": brier_scores,
        "chosen_calibration_method": best_method,
        "t43_annotation": {
            "mean_f1_before_t43": float(pre43.mean()) if not pre43.empty else None,
            "mean_f1_t43_onward": float(post43.mean()) if not post43.empty else None,
        },
        "shap_top_features_path": str(shap_path) if shap_df is not None else None,
    }
    eval_path = PROCESSED_DIR / "evaluation.json"
    eval_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {per_step_path}")
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
