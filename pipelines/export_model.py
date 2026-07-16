"""Deployment utility (Layer 11): export the @production model to a shippable directory.

The container cannot use the MLflow registry: `mlruns/` and `mlflow.db` are git-ignored
(HF Spaces builds from a git repo), and `mlflow.db` bakes absolute Windows artifact paths
into `model_versions.source` — so even committing the registry would leave a literal
`C:/Users/...` string that no Linux container can resolve. Instead we resolve the alias
*here*, on the machine that owns the registry, and ship only the ~1MB artifact.

The alias still decides *what* ships; the container just loads the path via MODEL_URI.
See DECISIONS.md D-028. Idempotent: re-run after every promotion.

Run from the repo root: `python pipelines/export_model.py`
"""
import os
import shutil
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
DEST = REPO_ROOT / "api" / "model"


def main() -> None:
    version = MlflowClient().get_model_version_by_alias(REGISTRY_NAME, "production").version

    if DEST.exists():
        shutil.rmtree(DEST)  # stale weights from a previous champion must not survive an export
    DEST.parent.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTRY_NAME}@production", dst_path=str(DEST)
    )

    # MLmodel ships with `artifact_path: file:C:/Users/.../mlruns/...` — the exact stale
    # absolute path this export exists to escape. Loading ignores it (model.ubj resolves
    # relative to the dir), but it would publish a local filesystem path to a public Space
    # and invites "so does the container use a Windows path?" — drop it.
    mlmodel = DEST / "MLmodel"
    kept = [ln for ln in mlmodel.read_text().splitlines() if not ln.startswith("artifact_path:")]
    mlmodel.write_text("\n".join(kept) + "\n")

    size_mb = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e6
    print(f"exported {REGISTRY_NAME}@production (v{version}) -> {DEST} ({size_mb:.1f} MB)")
    print(f"set MODEL_VERSION={version} in the root Dockerfile if it changed")


if __name__ == "__main__":
    main()
