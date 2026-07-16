"""Layer 7 gate tests: /health returns OK, /predict scores a known illicit and a
known licit example sanely via TestClient, and the request-schema guard rejects a
malformed feature vector at the trust boundary.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipelines.promote_model import main as promote
from src.models.baseline import MODEL_FEATURE_COLS
from src.serving.app import app

pytestmark = pytest.mark.needs_data

REPO_ROOT_NODES = "data/processed/nodes.parquet"


@pytest.fixture(scope="module")
def client():
    promote()  # idempotent — ensures @production alias exists before the app loads it
    return TestClient(app)


def _labeled_row(label: int) -> list[float]:
    nodes = pd.read_parquet(REPO_ROOT_NODES)
    test_rows = nodes[(nodes["time_step"] >= 35) & (nodes["label"] == label)]
    row = test_rows.iloc[0]
    return row[MODEL_FEATURE_COLS].astype(float).tolist()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_illicit_and_licit(client):
    illicit_features = _labeled_row(1)
    licit_features = _labeled_row(0)

    illicit_resp = client.post("/predict", json={"features": illicit_features})
    licit_resp = client.post("/predict", json={"features": licit_features})

    assert illicit_resp.status_code == 200
    assert licit_resp.status_code == 200

    illicit_proba = illicit_resp.json()["illicit_probability"]
    licit_proba = licit_resp.json()["illicit_probability"]
    assert 0.0 <= illicit_proba <= 1.0
    assert 0.0 <= licit_proba <= 1.0
    assert illicit_proba > licit_proba  # sanity: the wiring feeds the model correctly


def test_predict_wrong_length(client):
    resp = client.post("/predict", json={"features": [0.0] * (len(MODEL_FEATURE_COLS) - 1)})
    assert resp.status_code == 422


def test_serves_from_exported_weights_without_registry(monkeypatch):
    """Layer 11: the deployed container has no mlflow.db — prove MODEL_URI/MODEL_VERSION
    serve from exported weights alone, with every registry lookup made to explode."""
    import src.serving.app as app_module

    monkeypatch.setattr(app_module, "MODEL_URI", "api/model")
    monkeypatch.setenv("MODEL_VERSION", "2")
    monkeypatch.setattr(
        app_module, "MlflowClient", lambda *a, **kw: pytest.fail("registry was queried")
    )
    app_module.get_model.cache_clear()
    app_module.get_model_version.cache_clear()

    try:
        client = TestClient(app_module.app)
        assert client.get("/health").json() == {"status": "ok", "model_version": "2"}

        resp = client.post("/predict", json={"features": _labeled_row(1)})
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["illicit_probability"] <= 1.0
    finally:
        app_module.get_model.cache_clear()  # don't leak the exported model into other tests
        app_module.get_model_version.cache_clear()
