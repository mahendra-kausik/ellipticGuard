"""Layer 9 gate tests: drift metric responds to real shift, and /metrics captures
request latency p50/p95 from the running API.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipelines.promote_model import main as promote
from src.data.split import TemporalSplit
from src.monitoring.drift import drift_by_time_step
from src.serving.app import _percentile, app


def _synthetic_nodes(shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({
        "time_step": 1,
        "feat_a": rng.normal(0, 1, n),
        "label": rng.integers(0, 2, n),
    })
    later = pd.DataFrame({
        "time_step": 2,
        "feat_a": rng.normal(shift, 1, n),
        "label": rng.integers(0, 2, n),
    })
    return pd.concat([train, later], ignore_index=True)


def test_drift_rises_with_shift():
    shifted_nodes = _synthetic_nodes(shift=5.0)
    unshifted_nodes = _synthetic_nodes(shift=0.0)
    feature_cols = ["feat_a"]

    shifted_split = TemporalSplit(
        train=shifted_nodes[shifted_nodes["time_step"] == 1], val=pd.DataFrame(), test=pd.DataFrame()
    )
    unshifted_split = TemporalSplit(
        train=unshifted_nodes[unshifted_nodes["time_step"] == 1], val=pd.DataFrame(), test=pd.DataFrame()
    )

    shifted_drift = drift_by_time_step(shifted_nodes, shifted_split, feature_cols)
    unshifted_drift = drift_by_time_step(unshifted_nodes, unshifted_split, feature_cols)

    assert shifted_drift.loc[0, "share_drifted"] > unshifted_drift.loc[0, "share_drifted"]


def test_percentile_known_values():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 50) == 30.0
    assert _percentile(values, 95) == 50.0
    assert _percentile([], 50) == 0.0


def test_metrics_endpoint_captures_latency():
    promote()  # idempotent — ensures @production alias exists before the app loads it
    client = TestClient(app)
    for _ in range(5):
        client.get("/health")

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert 0.0 <= body["p50_ms"] <= body["p95_ms"]
