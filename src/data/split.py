"""Temporal split harness (Layer 2) — the ONLY sanctioned way to partition data downstream.

train = time steps 1..val_start-1, used for fitting scalers/imputers/models.
val   = time steps val_start..train_end, held out from fitting for tuning/threshold
        selection (carved from the *tail* of the training range, never the future).
test  = time steps test_start..test_end, untouched until final evaluation.

Never random — see CLAUDE.md Directive 3 / DECISIONS.md D-001.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_temporal_split(
    nodes_df: pd.DataFrame,
    train_end: int = 34,
    val_start: int = 30,
    test_start: int = 35,
    test_end: int = 49,
) -> TemporalSplit:
    assert val_start <= train_end < test_start <= test_end, "split boundaries out of order"

    train = nodes_df[nodes_df["time_step"] < val_start].reset_index(drop=True)
    val = nodes_df[
        (nodes_df["time_step"] >= val_start) & (nodes_df["time_step"] <= train_end)
    ].reset_index(drop=True)
    test = nodes_df[
        (nodes_df["time_step"] >= test_start) & (nodes_df["time_step"] <= test_end)
    ].reset_index(drop=True)
    return TemporalSplit(train=train, val=val, test=test)


def split_edges(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, split: TemporalSplit) -> TemporalSplit:
    """Assign each edge to train/val/test via its `txId1` endpoint's time step.

    Edges never cross time steps in this dataset (verified by a test), so either
    endpoint's step would agree — `txId1` is used as the lookup key.
    """
    txid_to_step = nodes_df.set_index("txId")["time_step"]
    edge_step = edges_df["txId1"].map(txid_to_step)

    train_steps = set(split.train["time_step"].unique())
    val_steps = set(split.val["time_step"].unique())
    test_steps = set(split.test["time_step"].unique())

    return TemporalSplit(
        train=edges_df[edge_step.isin(train_steps)].reset_index(drop=True),
        val=edges_df[edge_step.isin(val_steps)].reset_index(drop=True),
        test=edges_df[edge_step.isin(test_steps)].reset_index(drop=True),
    )
