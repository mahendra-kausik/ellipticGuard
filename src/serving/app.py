"""FastAPI serving app (Layer 7).

Loads the Production model (`models:/<name>@production`, set by
`pipelines/promote_model.py`) from the MLflow registry — never a hardcoded path.
Serves the champion's raw probability: Layer 6 (D-023) found the registered v2 has
no leakage-free holdout left to calibrate on, and is already near-calibrated
(test Brier 0.0268 -> 0.0264 with sigmoid calibration on a different base model) —
so a raw, honestly-labeled probability is the served score. See DECISIONS.md D-024.
"""
import logging
import os
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, field_validator

from src.models.baseline import MODEL_FEATURE_COLS

logger = logging.getLogger("elliptic_guard.serving")

# ponytail: in-process deque — single-worker demo latency buffer, not a shared metrics
# store. Swap for a Prometheus exporter if serving moves to multiple workers/replicas.
_LATENCY_MS: deque[float] = deque(maxlen=1000)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (stdlib only, no numpy dependency for this)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(pct / 100 * len(ordered)), len(ordered) - 1)
    return ordered[idx]

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
REGISTRY_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "elliptic-illicit")
THRESHOLD = float(os.environ.get("PREDICT_THRESHOLD", 0.5))


@lru_cache(maxsize=1)
def get_model():
    return mlflow.xgboost.load_model(f"models:/{REGISTRY_NAME}@production")


@lru_cache(maxsize=1)
def get_model_version() -> str:
    return str(MlflowClient().get_model_version_by_alias(REGISTRY_NAME, "production").version)


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        n = len(MODEL_FEATURE_COLS)
        if len(v) != n:
            raise ValueError(f"expected {n} features (order: {MODEL_FEATURE_COLS}), got {len(v)}")
        return v


class PredictResponse(BaseModel):
    illicit_probability: float
    prediction: int
    threshold: float
    model_name: str
    model_version: str


app = FastAPI(title="EllipticGuard")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _LATENCY_MS.append(elapsed_ms)
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.get("/metrics")
def metrics():
    values = list(_LATENCY_MS)
    return {
        "count": len(values),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
    }


@app.get("/health")
def health():
    version = get_model_version()
    return {"status": "ok", "model_version": version}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = get_model()
    X = pd.DataFrame([request.features], columns=MODEL_FEATURE_COLS)
    proba = float(model.predict_proba(X)[0, 1])
    return PredictResponse(
        illicit_probability=proba,
        prediction=int(proba >= THRESHOLD),
        threshold=THRESHOLD,
        model_name=REGISTRY_NAME,
        model_version=get_model_version(),
    )
