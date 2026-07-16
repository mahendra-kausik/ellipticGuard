# EllipticGuard serving image — CPU-only, free-tier (Hugging Face Spaces CPU-basic).
# Lives at the repo root because HF Docker Spaces require it there (the path isn't
# configurable); PROJECT_PLAN.md §5's api/ layout predates that constraint. See D-028.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# Weights exported from the elliptic-illicit@production alias by pipelines/export_model.py.
# The registry itself can't come along: mlflow.db bakes absolute Windows artifact paths that
# don't resolve on Linux. A hosted MLflow tracking server is the production upgrade — then
# MODEL_URI/MODEL_VERSION drop away and the app queries the alias directly (D-028).
COPY api/model/ model/
ENV MODEL_URI=/app/model MODEL_VERSION=2

EXPOSE 7860
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
