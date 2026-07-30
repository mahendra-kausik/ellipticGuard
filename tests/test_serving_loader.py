"""Layer 11 (D-034): the slimmed serving image loads model.ubj directly via
xgboost.XGBClassifier instead of mlflow.xgboost.load_model, to drop mlflow from the
serving path. Prove the two loaders agree on real weights — silent numeric divergence
between them is the one failure this change could plausibly introduce.

Data-free: reads the exported weights committed at api/model/ (see D-028), not the raw
Elliptic dataset — runs in CI, no `needs_data` marker.
"""
import numpy as np
import pandas as pd
import xgboost

from src.data.loaders import MODEL_FEATURE_COLS

MODEL_DIR = "api/model"


def test_direct_xgboost_load_matches_mlflow_load():
    import mlflow.xgboost  # dev/CI-only dependency; not present in the serving image itself

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((5, len(MODEL_FEATURE_COLS))), columns=MODEL_FEATURE_COLS
    )

    direct = xgboost.XGBClassifier()
    direct.load_model(f"{MODEL_DIR}/model.ubj")

    via_mlflow = mlflow.xgboost.load_model(MODEL_DIR)

    np.testing.assert_allclose(
        direct.predict_proba(X), via_mlflow.predict_proba(X), rtol=1e-6, atol=1e-8
    )
