# EllipticGuard serving image — CPU-only. Deployed to Cloud Run (D-034); HF Docker Spaces
# turned out to require a paid plan (D-029). Lives at the repo root because that was the
# HF constraint that shaped the layout; kept here since Cloud Run's --source build also
# expects the Dockerfile at the build context root. PROJECT_PLAN.md §5's api/ layout
# predates both constraints. See D-028.
FROM python:3.11-slim

WORKDIR /app

# requirements-serve.txt, not requirements.txt: mlflow/dvc/shap/matplotlib/evidently/pytest
# are training & CI dependencies the serving path never imports (D-034) — installing them
# here was most of the image's 3.43 GB.
COPY requirements-serve.txt .
# xgboost installed separately with --no-deps: its wheel hard-requires nvidia-nccl-cu12
# (~400 MB, GPU-only, unused here) alongside numpy/scipy, which requirements-serve.txt
# already installs above.
RUN pip install --no-cache-dir -r requirements-serve.txt \
 && pip install --no-cache-dir --no-deps "xgboost>=2.0.0"

COPY src/ src/
# Weights exported from the elliptic-illicit@production alias by pipelines/export_model.py.
# The registry itself can't come along: mlflow.db bakes absolute Windows artifact paths that
# don't resolve on Linux. A hosted MLflow tracking server is the production upgrade — then
# MODEL_URI/MODEL_VERSION drop away and the app queries the alias directly (D-028).
COPY api/model/ model/
ENV MODEL_URI=/app/model MODEL_VERSION=2

EXPOSE 7860
# Shell form so Cloud Run's injected $PORT is honored; falls back to 7860 for
# `docker run -p 7860:7860 ellipticguard` and the HF-era default.
CMD uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-7860}
