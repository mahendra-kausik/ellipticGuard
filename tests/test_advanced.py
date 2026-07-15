"""Layer 5 gate tests: graph-feature merge integrity, imbalance-weight math, and
feature-set shapes (provided-only vs +graph) stay leakage-safe (labeled-only rows).
"""
import numpy as np
import pandas as pd

from src.data.loaders import FEATURE_COLS
from src.features.graph import GRAPH_FEATURE_COLS
from src.models.advanced import (
    ALL_FEATURE_COLS,
    PROVIDED_FEATURE_COLS,
    build_xy_features,
    merge_graph_features,
    scale_pos_weight,
)


def _fake_nodes(n_per_step: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for step in range(1, 6):
        for i in range(n_per_step):
            label = 1 if i == 0 else (0 if i < 15 else np.nan)
            row = {"txId": step * 1000 + i, "time_step": step, "label": label}
            row.update({col: rng.normal() for col in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def _fake_graph_features(nodes_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    graph_df = nodes_df[["txId", "time_step"]].copy()
    for col in GRAPH_FEATURE_COLS:
        graph_df[col] = rng.normal(size=len(graph_df))
    return graph_df


def test_merge_graph_features_aligns_by_txid_no_row_loss():
    nodes_df = _fake_nodes()
    graph_df = _fake_graph_features(nodes_df)

    merged = merge_graph_features(nodes_df, graph_df)

    assert len(merged) == len(nodes_df)
    for col in GRAPH_FEATURE_COLS:
        assert col in merged.columns
    # spot-check one node's graph feature came from its own row, not a mismatched one
    sample_txid = nodes_df["txId"].iloc[5]
    expected = graph_df.loc[graph_df["txId"] == sample_txid, GRAPH_FEATURE_COLS[0]].iloc[0]
    actual = merged.loc[merged["txId"] == sample_txid, GRAPH_FEATURE_COLS[0]].iloc[0]
    assert expected == actual


def test_scale_pos_weight_is_neg_over_pos():
    y = pd.Series([1, 0, 0, 0, 1, 0, 0, 0])  # 2 positive, 6 negative
    assert scale_pos_weight(y) == 3.0


def test_build_xy_features_shapes_for_both_feature_sets():
    nodes_df = _fake_nodes()
    graph_df = _fake_graph_features(nodes_df)
    merged = merge_graph_features(nodes_df, graph_df)

    X_provided, y_provided = build_xy_features(merged, PROVIDED_FEATURE_COLS)
    X_all, y_all = build_xy_features(merged, ALL_FEATURE_COLS)

    assert X_provided.shape[1] == 166
    assert X_all.shape[1] == 166 + len(GRAPH_FEATURE_COLS)
    assert len(X_provided) == len(X_all) == len(y_provided) == len(y_all)
    assert len(X_provided) == merged["label"].notna().sum()
    assert set(y_all.unique()) <= {0, 1}
