"""DVC-adjacent utility (Layer 7): promote the serving-candidate model to @production.

MLflow 3.x stages (Staging/Production) are deprecated in favor of aliases (D-020's sqlite
backend supports them fine). Idempotent: safe to re-run any time a new champion is trained.

Run from the repo root: `python pipelines/promote_model.py`
"""
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
REGISTRY_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "elliptic-illicit")


def main() -> None:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions registered under '{REGISTRY_NAME}' — run train_advanced.py first.")

    candidates = [v for v in versions if v.tags.get("serving_candidate") == "true"]
    chosen = max(candidates or versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(REGISTRY_NAME, "production", chosen.version)
    print(f"{REGISTRY_NAME}@production -> v{chosen.version} (feature_set={chosen.tags.get('feature_set')})")


if __name__ == "__main__":
    main()
