"""Runtime-validated loaders for the raw Elliptic CSVs (Layer 1).

Header presence and feature count differ across mirrors of this dataset, so we
confirm both at load time instead of hardcoding assumptions (PROJECT_PLAN.md §2).
"""
from pathlib import Path

import pandas as pd

NUM_FEATURES = 165  # feat_0..feat_164; time_step is the paper's 166th ("first local") feature
FEATURE_COLS = [f"feat_{i}" for i in range(NUM_FEATURES)]
# Defined here (not in src/models/baseline.py) so the serving path can use it without
# importing sklearn — src/models/baseline.py re-exports it for training-side callers (D-034).
MODEL_FEATURE_COLS = ["time_step"] + FEATURE_COLS  # 166 features, per PROJECT_PLAN.md §2


def load_features(path: Path) -> pd.DataFrame:
    """elliptic_txs_features.csv has NO header: txId, time_step, 165 feature cols."""
    df = pd.read_csv(path, header=None)
    assert df.shape[1] == 167, f"expected 167 raw columns (txId, time_step, 165 features), got {df.shape[1]}"
    df.columns = ["txId", "time_step"] + FEATURE_COLS
    assert df["txId"].is_unique, "duplicate txId in features file"
    assert df["time_step"].between(1, 49).all(), "time_step outside expected 1-49 range"
    return df


def load_classes(path: Path) -> pd.DataFrame:
    """elliptic_txs_classes.csv HAS a header: txId, class."""
    df = pd.read_csv(path)
    assert list(df.columns) == ["txId", "class"], f"unexpected classes header: {list(df.columns)}"
    assert df["txId"].is_unique, "duplicate txId in classes file"
    return df


def load_edgelist(path: Path) -> pd.DataFrame:
    """elliptic_txs_edgelist.csv HAS a header: txId1, txId2."""
    df = pd.read_csv(path)
    assert list(df.columns) == ["txId1", "txId2"], f"unexpected edgelist header: {list(df.columns)}"
    return df


def assemble_node_table(features_df: pd.DataFrame, classes_df: pd.DataFrame) -> pd.DataFrame:
    """Join features + classes on txId and remap labels.

    illicit (1) -> 1, licit (2) -> 0, unknown -> NaN (excluded from supervised
    train/eval downstream, per D-002; kept here so the row is still available
    for graph-structure use).
    """
    nodes_df = features_df.merge(classes_df, on="txId", how="left", validate="one_to_one")
    assert len(nodes_df) == len(features_df), "merge changed row count — integrity broken"
    nodes_df["label"] = nodes_df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})
    return nodes_df


def check_integrity(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """Sanity counts used both by the assemble pipeline and the Layer 1 gate test."""
    return {
        "n_nodes": len(nodes_df),
        "n_edges": len(edges_df),
        "time_step_min": int(nodes_df["time_step"].min()),
        "time_step_max": int(nodes_df["time_step"].max()),
        "n_feature_cols": len(FEATURE_COLS),
        "null_counts_features": int(nodes_df[FEATURE_COLS].isna().sum().sum()),
        "illicit": int((nodes_df["label"] == 1).sum()),
        "licit": int((nodes_df["label"] == 0).sum()),
        "unknown": int(nodes_df["label"].isna().sum()),
    }
