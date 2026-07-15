"""Advanced model (Layer 5): XGBoost on provided + graph features.

Imbalance handled via `scale_pos_weight` (vs RF's `class_weight` in Layer 3 — see
DECISIONS.md). Hyperparameters are tuned on the `val` slice only (steps 30-34);
`test` is touched once, at final evaluation (CLAUDE.md Directive 3). XGBoost
handles NaN natively, so there is no imputer/scaler to fit.
"""
import itertools

import pandas as pd
from xgboost import XGBClassifier

from src.features.graph import GRAPH_FEATURE_COLS
from src.models.baseline import MODEL_FEATURE_COLS, evaluate

PROVIDED_FEATURE_COLS = MODEL_FEATURE_COLS  # 166 provided features (Layer 3)
ALL_FEATURE_COLS = MODEL_FEATURE_COLS + GRAPH_FEATURE_COLS  # 166 + 7 = 173


def merge_graph_features(nodes_df: pd.DataFrame, graph_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join graph topology features onto nodes by txId.

    graph_df carries its own `time_step` (identical to nodes_df's, since graph
    features are computed per-step) — drop it to avoid a duplicate/suffixed column.
    """
    graph_cols = ["txId"] + GRAPH_FEATURE_COLS
    merged = nodes_df.merge(graph_df[graph_cols], on="txId", how="left", validate="one_to_one")
    assert len(merged) == len(nodes_df), "merge changed row count — integrity broken"
    return merged


def build_xy_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to labeled rows (D-002) and select the given feature columns."""
    labeled = df[df["label"].notna()]
    return labeled[feature_cols], labeled["label"].astype(int)


def scale_pos_weight(y_train: pd.Series) -> float:
    """neg/pos count ratio, the standard XGBoost imbalance-handling knob."""
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return neg / pos


def train_xgb(X_train, y_train, params: dict, spw: float, random_state: int) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        **params,
    )
    model.fit(X_train, y_train)
    return model


def tune_on_val(X_train, y_train, X_val, y_val, grid: dict, spw: float, random_state: int) -> tuple[dict, float]:
    """Small manual grid search, scored on val illicit-F1. Returns (best_params, best_val_f1)."""
    keys = list(grid.keys())
    best_params, best_f1 = None, -1.0
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        model = train_xgb(X_train, y_train, params, spw, random_state)
        val_f1 = evaluate(model, X_val, y_val)["illicit_f1"]
        if val_f1 > best_f1:
            best_params, best_f1 = params, val_f1
    return best_params, best_f1
