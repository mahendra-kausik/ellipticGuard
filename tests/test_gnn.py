"""Layer 8 (optional) — GNN module tests. Data-free, synthetic fixtures only.

`importorskip` keeps lean CI (`pytest -m "not needs_data"`) green without
installing torch — these tests need no real data (so `needs_data` would be the
wrong marker), they just need torch, which CI doesn't install (D-030).
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.loaders import FEATURE_COLS  # noqa: E402
from src.models.baseline import fit_preprocessor  # noqa: E402
from src.models.gnn import (  # noqa: E402
    GCN,
    EvolveGCN,
    GraphSAGE,
    build_snapshots,
    masked_cross_entropy,
    normalize_adj,
    row_normalize_adj,
)


def _synthetic_nodes(tx_ids, time_steps, labels):
    """165 zero-valued feature columns — only structure/labels matter for these tests."""
    n = len(tx_ids)
    df = pd.DataFrame({"txId": tx_ids, "time_step": time_steps})
    for col in FEATURE_COLS:
        df[col] = 0.0
    df["label"] = labels
    return df


def test_normalize_adj_hand_computed_3_node_graph():
    """Path graph 0-1-2 (edge_index encodes 0->1, 1->2). Â = D^-1/2 (A+I) D^-1/2
    over the *undirected* graph: degrees (incl. self-loop) are 2, 3, 2 for
    nodes 0, 1, 2 -> deg_inv_sqrt = 1/sqrt2, 1/sqrt3, 1/sqrt2 (hand-worked)."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    adj = normalize_adj(edge_index, n=3).to_dense()

    inv_sqrt2, inv_sqrt3 = 1 / np.sqrt(2), 1 / np.sqrt(3)
    expected = torch.tensor(
        [
            [inv_sqrt2 * inv_sqrt2, inv_sqrt2 * inv_sqrt3, 0.0],
            [inv_sqrt3 * inv_sqrt2, inv_sqrt3 * inv_sqrt3, inv_sqrt3 * inv_sqrt2],
            [0.0, inv_sqrt2 * inv_sqrt3, inv_sqrt2 * inv_sqrt2],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(adj, expected, atol=1e-6)


def test_row_normalize_adj_mean_aggregation():
    """Same path graph: row-normalized (no self-loop) D^-1 A over the
    symmetrized/deduped edge set. Node 1 has 2 undirected neighbours (0, 2) ->
    each row entry 0.5; nodes 0 and 2 have 1 neighbour each -> entry 1.0."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    adj = row_normalize_adj(edge_index, n=3).to_dense()

    expected = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    assert torch.allclose(adj, expected, atol=1e-6)


def test_snapshots_are_causal_per_time_step():
    """A node's snapshot (features + adjacency) must not change when unrelated
    edges are added to a different time step — mirrors
    test_graph_features.py::test_features_are_per_time_step_only."""
    nodes_df = _synthetic_nodes(
        ["A", "B", "X", "Y"], [1, 1, 2, 2], [1, 0, 1, 0]
    )
    edges_step1_only = pd.DataFrame({"txId1": ["A"], "txId2": ["B"]})
    edges_with_step2_noise = pd.concat(
        [edges_step1_only, pd.DataFrame({"txId1": ["X"], "txId2": ["Y"]})], ignore_index=True
    )
    preprocessor = fit_preprocessor(nodes_df[FEATURE_COLS])

    snaps_a = {s.time_step: s for s in build_snapshots(nodes_df, edges_step1_only, preprocessor)}
    snaps_b = {s.time_step: s for s in build_snapshots(nodes_df, edges_with_step2_noise, preprocessor)}

    assert torch.equal(snaps_a[1].edge_index, snaps_b[1].edge_index)
    assert torch.equal(snaps_a[1].x, snaps_b[1].x)


def test_masked_cross_entropy_ignores_unknown_nodes():
    """Flipping an unknown (label_mask False) node's stored label must not
    change the loss — only labeled rows may contribute."""
    logits = torch.randn(4, 2)
    y = torch.tensor([1, 0, 1, 0])
    label_mask = torch.tensor([True, True, False, True])

    loss_before = masked_cross_entropy(logits, y, label_mask)
    y_flipped = y.clone()
    y_flipped[2] = 1 - y_flipped[2]  # flip the unknown node's placeholder label
    loss_after = masked_cross_entropy(logits, y_flipped, label_mask)

    assert torch.equal(loss_before, loss_after)


@pytest.mark.parametrize("model_cls", [GCN, GraphSAGE, EvolveGCN])
def test_model_forward_pass_shape(model_cls):
    """Each model forward-passes a synthetic 5-node snapshot to shape (5, 2)."""
    n = 5
    x = torch.randn(n, 165)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    adj_fn = row_normalize_adj if model_cls.ADJ_TYPE == "row" else normalize_adj
    adj = adj_fn(edge_index, n)

    model = model_cls()
    out = model(x, adj)

    assert out.shape == (n, 2)
