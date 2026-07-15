"""Layer 6 gate tests: per-time-step F1 curve shape, and calibration is
leakage-safe (fit on val only, via cv="prefit", never touching test).
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

from src.data.loaders import FEATURE_COLS
from src.models.advanced import PROVIDED_FEATURE_COLS, build_xy_features, scale_pos_weight, train_xgb
from src.models.evaluate import brier, calibrate_on_val, per_time_step_metrics


def _fake_split_df(steps: range, n_per_step: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for step in steps:
        for i in range(n_per_step):
            label = 1 if i < 5 else (0 if i < 25 else np.nan)
            row = {"txId": step * 10000 + i, "time_step": step, "label": label}
            row.update({col: rng.normal() for col in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def _fake_model():
    train_df = _fake_split_df(range(1, 30))
    X_train, y_train = build_xy_features(train_df, PROVIDED_FEATURE_COLS)
    spw = scale_pos_weight(y_train)
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 20}
    return train_xgb(X_train, y_train, params, spw, random_state=42)


def test_per_time_step_metrics_one_row_per_test_step():
    model = _fake_model()
    test_df = _fake_split_df(range(35, 50))  # 15 steps, matches PROJECT_PLAN test range

    result = per_time_step_metrics(model, test_df, PROVIDED_FEATURE_COLS)

    assert len(result) == 15
    assert set(result["time_step"]) == set(range(35, 50))
    assert result["illicit_f1"].between(0, 1).all()
    assert result["auc_pr"].between(0, 1).all()


def test_calibration_is_prefit_on_val_and_produces_valid_probabilities():
    model = _fake_model()
    val_df = _fake_split_df(range(30, 35))
    test_df = _fake_split_df(range(35, 50))
    X_val, y_val = build_xy_features(val_df, PROVIDED_FEATURE_COLS)
    X_test, y_test = build_xy_features(test_df, PROVIDED_FEATURE_COLS)

    calibrated = calibrate_on_val(model, X_val, y_val, method="sigmoid")

    # Leakage guard: calibrator must wrap a frozen (already-trained-on-train-only)
    # base model — FrozenEstimator is what guarantees val/test never influence
    # the base model's fit, only the calibration mapping.
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert isinstance(calibrated.estimator, FrozenEstimator)

    proba = calibrated.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    score = brier(calibrated, X_test, y_test)
    assert np.isfinite(score)
    assert 0 <= score <= 1
