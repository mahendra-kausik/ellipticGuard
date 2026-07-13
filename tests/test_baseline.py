"""Layer 3 gate tests: baseline models use labeled rows only, and the preprocessor
is fit strictly on train (leakage-safety), never shifting when val/test distributions differ.
"""
import numpy as np
import pandas as pd

from src.data.loaders import FEATURE_COLS
from src.models.baseline import build_xy, fit_preprocessor, train_random_forest


def _fake_nodes(n_per_step: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for step in range(1, 6):
        for i in range(n_per_step):
            label = 1 if i == 0 else (0 if i < 15 else np.nan)  # some illicit, licit, unknown
            row = {"time_step": step, "label": label}
            row.update({col: rng.normal() for col in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_xy_excludes_unknown_labels():
    nodes_df = _fake_nodes()
    X, y = build_xy(nodes_df)
    assert y.isna().sum() == 0
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == (nodes_df["label"].notna()).sum()


def test_preprocessor_fit_only_on_train_not_shifted_by_other_partitions():
    train_df = _fake_nodes(n_per_step=20)
    X_train, _ = build_xy(train_df)

    # A distribution-shifted "val" set the preprocessor must NEVER see.
    shifted_df = _fake_nodes(n_per_step=20)
    shifted_df[FEATURE_COLS] = shifted_df[FEATURE_COLS] + 1000.0
    X_shifted, _ = build_xy(shifted_df)

    prep = fit_preprocessor(X_train)

    # Fitting only on train means the scaler's mean matches train's mean, not a
    # blend with the shifted partition.
    train_mean = X_train[FEATURE_COLS[0]].median()  # imputer uses median strategy
    assert abs(prep.imputer.statistics_[X_train.columns.get_loc(FEATURE_COLS[0])] - train_mean) < 1e-6

    # Transforming the shifted set with train-fit params should NOT look standardized
    # (proves the scaler wasn't refit on it).
    X_shifted_transformed = prep.transform(X_shifted)
    assert X_shifted_transformed.mean() > 5, "scaler appears fit on shifted data, not train-only"


def test_train_random_forest_and_evaluate_shapes():
    nodes_df = _fake_nodes()
    X, y = build_xy(nodes_df)
    prep = fit_preprocessor(X)
    X_t = prep.transform(X)
    model = train_random_forest(X_t, y, random_state=42, n_estimators=10, max_depth=3)
    preds = model.predict(X_t)
    assert len(preds) == len(y)
    assert set(np.unique(preds)) <= {0, 1}
