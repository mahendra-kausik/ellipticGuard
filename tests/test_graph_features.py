import pandas as pd
import pytest

from src.features.graph import compute_graph_features


def test_hand_computed_tiny_subgraph():
    """A ⟶ B ⟶ C, A ⟶ C, plus an isolated node D, all in time step 1.

    Hand-computed (undirected view for clustering/component/neighbor-degree):
      A: out={B,C}, in={}          -> in=0, out=2, neighbors={B,C}
      B: out={C}, in={A}           -> in=1, out=1, neighbors={A,C}
      C: out={}, in={A,B}          -> in=2, out=0, neighbors={A,B}
      D: isolated                  -> in=0, out=0, neighbors={}
    Undirected triangle A-B-C is fully connected (3 edges among 3 nodes) ->
    clustering coefficient = 1.0 for A, B, C. D has no neighbors -> clustering 0.
    Component sizes: {A,B,C} -> 3 each; {D} -> 1.
    """
    nodes_df = pd.DataFrame(
        {"txId": ["A", "B", "C", "D"], "time_step": [1, 1, 1, 1]}
    )
    edges_df = pd.DataFrame(
        {"txId1": ["A", "B", "A"], "txId2": ["B", "C", "C"]}
    )

    feats = compute_graph_features(nodes_df, edges_df).set_index("txId")

    assert feats.loc["A", "in_degree"] == 0
    assert feats.loc["A", "out_degree"] == 2
    assert feats.loc["A", "unique_neighbors"] == 2
    assert feats.loc["A", "clustering_coef"] == 1.0
    assert feats.loc["A", "component_size"] == 3

    assert feats.loc["B", "in_degree"] == 1
    assert feats.loc["B", "out_degree"] == 1
    assert feats.loc["B", "clustering_coef"] == 1.0

    assert feats.loc["C", "in_degree"] == 2
    assert feats.loc["C", "out_degree"] == 0
    assert feats.loc["C", "clustering_coef"] == 1.0

    assert feats.loc["D", "in_degree"] == 0
    assert feats.loc["D", "out_degree"] == 0
    assert feats.loc["D", "unique_neighbors"] == 0
    assert feats.loc["D", "clustering_coef"] == 0.0
    assert feats.loc["D", "component_size"] == 1

    assert feats["pagerank"].sum() == pytest.approx(1.0)


def test_features_are_per_time_step_only():
    """A node's features must not change when unrelated edges are added to a
    different time step — the causal-by-construction guarantee (PROJECT_PLAN.md
    Layer 4 gate)."""
    nodes_df = pd.DataFrame(
        {"txId": ["A", "B", "X", "Y"], "time_step": [1, 1, 2, 2]}
    )
    edges_step1_only = pd.DataFrame({"txId1": ["A"], "txId2": ["B"]})
    edges_with_step2_noise = pd.concat(
        [edges_step1_only, pd.DataFrame({"txId1": ["X"], "txId2": ["Y"]})],
        ignore_index=True,
    )

    feats_a = compute_graph_features(nodes_df, edges_step1_only).set_index("txId")
    feats_b = compute_graph_features(nodes_df, edges_with_step2_noise).set_index("txId")

    for col in ["in_degree", "out_degree", "unique_neighbors", "clustering_coef", "component_size"]:
        assert feats_a.loc["A", col] == feats_b.loc["A", col]
        assert feats_a.loc["B", col] == feats_b.loc["B", col]
