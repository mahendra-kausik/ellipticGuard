"""Champion evaluation (Layer 6): per-time-step F1 curve, calibration, lean SHAP.

The base model here is refit on train ONLY (steps 1-29), not train+val like the
registered v2 champion — this keeps the val slice (30-34) genuinely held out to
use as a calibration set (CLAUDE.md Directive 3: never fit a calibration model
on anything but the train partition; val is the sanctioned tuning/calibration
holdout, test stays untouched until final scoring). See D-023.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import brier_score_loss

from src.models.baseline import evaluate
from src.models.advanced import train_xgb, scale_pos_weight


def per_time_step_metrics(model, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Evaluate `model` on each labeled test time step separately.

    Returns one row per time step present in test_df: time_step, n_labeled,
    n_illicit, illicit_precision, illicit_recall, illicit_f1, auc_pr.
    """
    rows = []
    for step, step_df in test_df.groupby("time_step"):
        labeled = step_df[step_df["label"].notna()]
        if labeled.empty:
            continue
        X = labeled[feature_cols]
        y = labeled["label"].astype(int)
        m = evaluate(model, X, y)
        rows.append({
            "time_step": int(step),
            "n_labeled": len(labeled),
            "n_illicit": int((y == 1).sum()),
            "illicit_precision": m["illicit_precision"],
            "illicit_recall": m["illicit_recall"],
            "illicit_f1": m["illicit_f1"],
            "auc_pr": m["auc_pr"],
        })
    return pd.DataFrame(rows).sort_values("time_step").reset_index(drop=True)


def calibrate_on_val(base_model, X_val, y_val, method: str) -> CalibratedClassifierCV:
    """Fit a calibrator on the val holdout against an already-trained (frozen) base model.

    `FrozenEstimator` is what makes this leakage-safe (sklearn >=1.6 replacement
    for the deprecated `cv="prefit"`): it wraps the base model — already trained
    on train only — so `CalibratedClassifierCV.fit` only fits the calibration
    mapping, on val only, and never refits/touches the base model itself.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    calibrated.fit(X_val, y_val)
    return calibrated


def brier(model, X, y) -> float:
    proba = model.predict_proba(X)[:, 1]
    return float(brier_score_loss(y, proba))


def top_shap_features(model, X_sample: pd.DataFrame, n: int) -> pd.DataFrame | None:
    """Global mean(|SHAP value|) feature ranking via TreeExplainer.

    Guarded: any import/compute failure is swallowed (a warning is emitted) and
    None is returned, so a SHAP environment issue never fails the pipeline/gate.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranking = (
            pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return ranking
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.warn(f"SHAP computation skipped: {exc}")
        return None


def build_train_only_champion(train_df: pd.DataFrame, feature_cols: list[str], best_params: dict, random_state: int):
    """Refit the champion's tuned params on train (steps 1-29) only — see module docstring."""
    from src.models.advanced import build_xy_features

    X_train, y_train = build_xy_features(train_df, feature_cols)
    spw = scale_pos_weight(y_train)
    model = train_xgb(X_train, y_train, best_params, spw, random_state)
    return model
