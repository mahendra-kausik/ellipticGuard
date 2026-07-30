"""FastAPI serving app (Layer 7).

Loads the Production model (`models:/<name>@production`, set by
`pipelines/promote_model.py`) from the MLflow registry — never a hardcoded path.
In the deployed container there is no registry to query, so MODEL_URI points at weights
exported from that same alias at build time (D-028).
Serves the champion's raw probability: Layer 6 (D-023) found the registered v2 has
no leakage-free holdout left to calibrate on, and is already near-calibrated
(test Brier 0.0268 -> 0.0264 with sigmoid calibration on a different base model) —
so a raw, honestly-labeled probability is the served score. See DECISIONS.md D-024.

The deployed image installs `requirements-serve.txt`, not the full `requirements.txt` —
mlflow (and everything it drags in: sqlalchemy, alembic, pyarrow, flask) is a training/
tracking dependency the container never needs, since MODEL_URI is a local directory
there. It's imported lazily, only on the local-dev registry path. See DECISIONS.md D-034.
"""
import logging
import os
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, field_validator

from src.data.loaders import MODEL_FEATURE_COLS

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

REGISTRY_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "elliptic-illicit")
THRESHOLD = float(os.environ.get("PREDICT_THRESHOLD", 0.5))

# The registry is the default source of truth. MODEL_URI/MODEL_VERSION exist only for the
# deployed container, which has no mlflow.db to query — `pipelines/export_model.py` resolves
# the alias at build time and bakes the weights in. See DECISIONS.md D-028.
MODEL_URI = os.environ.get("MODEL_URI") or f"models:/{REGISTRY_NAME}@production"


def _load_from_registry():
    """Only reached on the local-dev path (MODEL_URI is a registry URI, not a directory).
    mlflow is imported here, not at module level, so the slim serving image — which never
    installs it — can import this module at all. See DECISIONS.md D-034."""
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    )
    return mlflow.xgboost, MlflowClient


@lru_cache(maxsize=1)
def get_model():
    path = Path(MODEL_URI)
    if path.is_dir():  # exported weights (container) — no mlflow needed (D-034)
        model = xgboost.XGBClassifier()
        model.load_model(str(path / "model.ubj"))
        return model
    mlflow_xgboost, _ = _load_from_registry()
    return mlflow_xgboost.load_model(MODEL_URI)


@lru_cache(maxsize=1)
def get_model_version() -> str:
    pinned = os.environ.get("MODEL_VERSION")
    if pinned:
        return pinned
    _, MlflowClient = _load_from_registry()
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
