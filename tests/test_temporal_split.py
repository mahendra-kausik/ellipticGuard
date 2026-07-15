from pathlib import Path

import pandas as pd
import pytest

from src.data.split import make_temporal_split, split_edges

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load():
    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges_df = pd.read_parquet(PROCESSED_DIR / "edges.parquet")
    return nodes_df, edges_df


@pytest.mark.needs_data
def test_no_cross_time_step_edges():
    """(c) Edges never cross time steps — the premise that makes per-step graph
    features causal by construction (PROJECT_PLAN.md §2)."""
    nodes_df, edges_df = _load()
    txid_to_step = nodes_df.set_index("txId")["time_step"]
    step1 = edges_df["txId1"].map(txid_to_step)
    step2 = edges_df["txId2"].map(txid_to_step)
    assert (step1 == step2).all(), "found edges spanning more than one time step"


@pytest.mark.needs_data
def test_temporal_split_partitions_disjoint_and_ordered():
    """(a) no txId in more than one partition; (b) test steps > train/val steps,
    and val is carved from the tail of the training range, never the future."""
    nodes_df, edges_df = _load()
    split = make_temporal_split(nodes_df, train_end=34, val_start=30, test_start=35, test_end=49)

    train_ids = set(split.train["txId"])
    val_ids = set(split.val["txId"])
    test_ids = set(split.test["txId"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(nodes_df)

    assert split.train["time_step"].max() < split.val["time_step"].min()
    assert split.val["time_step"].max() < split.test["time_step"].min()
    assert split.train["time_step"].min() == 1
    assert split.val["time_step"].max() == 34
    assert split.test["time_step"].max() == 49

    edge_split = split_edges(edges_df, nodes_df, split)
    assert len(edge_split.train) + len(edge_split.val) + len(edge_split.test) == len(edges_df)
