# EllipticGuard serving image — CPU-only, free-tier (Hugging Face Spaces CPU-basic).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# ponytail: ships the local sqlite registry + its mlruns/ artifact store as the container's
# model store; a hosted MLflow tracking server is the production upgrade (PROJECT_PLAN.md §8
# already flags this).
COPY mlflow.db .
COPY mlruns/ mlruns/

EXPOSE 7860
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
