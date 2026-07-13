# SESSION_LOG.md — EllipticGuard

Single source of truth for "where are we?" Update at the end of every session and every time a layer gate is reached. Newest entry on top.

---

## Current state
- **Active layer:** Layer 1 — Data ingestion & integrity (NOT STARTED)
- **Last gate passed:** Layer 0 — Scaffolding & environment
- **Next action:** Read `PROJECT_PLAN.md` §6 Layer 1. Build loaders that confirm headers/shapes at runtime, join features+classes, parse edgelist, remap labels, DVC-track raw + assembled data.
- **Blockers / open questions:** None currently blocking. Notebooks are now in `notebooks/` for reference at their designated layers.

### Owner action items — do BEFORE this layer starts (things Claude Code cannot do)
> Claude Code refreshes this list at the end of each layer for the next one. These are steps that need the owner's hands (accounts, secrets, uploads, external setup).
- [x] Create a GitHub repo and connect it — remote `origin` set to `https://github.com/mahendra-kausik/ellipticGuard.git`; owner must push (or confirm push) since this session did not verify remote push access/credentials.
- [x] Confirm Python 3.11+ is installed locally — Python 3.13.3 confirmed, venv created at `.venv/`.
- [x] Decide the DVC remote — Google Drive folder ID present in `.env` (`GDRIVE_FOLDER_ID`) and wired into `.dvc/config.local` (git-ignored) via `dvc remote add -d --local`.
- [ ] (Layer 1) Owner should verify `git push -u origin main` succeeds with their GitHub credentials (this session cannot authenticate interactively).
- [ ] (Layer 1) Elliptic CSVs are already in place at `data/raw/` — no download needed. Preprocessing notebook (`notebooks/01_Elliptic_Preprocessing.ipynb`) is available as reference.
- [ ] (Later) Create a free Hugging Face account + a Space (Layer 11) and a free MLflow-compatible setup if you choose a hosted tracking server.
- [x] Fill `.env` from `.env.example` with required keys/IDs — done; `.env` confirmed git-ignored, never staged.

---

## Layer status
| Layer | Name | Status | Gate met? | Notes |
|---|---|---|---|---|
| 0 | Scaffolding & environment | complete | yes | venv (.venv), git init + remote, DVC init + local gdrive remote, mlflow dir, pytest smoke green |
| 1 | Data ingestion & integrity | not started | — | needs raw CSVs; confirm headers + feature count at load |
| 2 | Temporal split harness & EDA | not started | — | CRITICAL — leakage-safety gate |
| 3 | Baseline models | not started | — | register RF as v1 |
| 4 | Graph features | not started | — | per-time-step, causal |
| 5 | Advanced model (XGBoost) | not started | — | register champion v2 |
| 6 | Evaluation, calibration, drift story | not started | — | T43 curve |
| 7 | Serving API (FastAPI) | not started | — | loads Production model |
| 8 | GNN comparison (OPTIONAL) | not started | — | Colab free; owner notebooks |
| 9 | Monitoring & observability | not started | — | Evidently / NannyML |
| 10 | Retraining loop & CI/CD (replay) | not started | — | flag fires at T43 |
| 11 | Deployment & docs | not started | — | HF Spaces live URL |

---

## Notebooks provided by owner
| Notebook | Provided? | Read at layer | Bugs found / fixed (→ DECISIONS.md) |
|---|---|---|---|
| preprocessing | yes (`notebooks/01_Elliptic_Preprocessing.ipynb`) | 1–2 | |
| static GCN | yes (`notebooks/02_Static_GCN.ipynb`) | 8 | |
| EvolveGCN | yes (`notebooks/03_EvolveGCN.ipynb`) | 8 | |
| SHAP | yes (`notebooks/04_Shap_Drift_Analysis.ipynb`) | 6 | |

---

## Changelog
<!-- Newest on top. One block per session/gate. -->

### 2026-07-13 — Layer 0
- **Layer worked on:** Layer 0 — Scaffolding & environment
- **What changed:** Created repo folder structure (`src/{data,features,models,monitoring,serving}`, `pipelines/`, `tests/`, `api/`, `notebooks/`, `.github/workflows/`, `data/raw/`); moved the 4 owner notebooks into `notebooks/` and the 3 Elliptic CSVs into `data/raw/`; created `.venv` and installed `requirements.txt` clean (no pin conflicts on Python 3.13.3); wrote `.gitignore`, stub `dvc.yaml`/`params.yaml`, `README.md`, `tests/test_smoke.py`; `git init` (branch `main`) + `git remote add origin`; `dvc init` + Google Drive remote added via `--local` config (ID sourced from `.env`, never committed); created `mlruns/`; added a commit-attribution note at the top of `CLAUDE.md` (no `Co-Authored-By: Claude` trailer).
- **Gate evidence:** `pytest -q` → `1 passed`; `dvc status` → clean, no tracked data/pipelines yet (expected pre-Layer-1); `python -c "import mlflow; print(mlflow.__version__)"` → `3.14.0`; `git check-ignore` confirmed `.env`, `data/`, `mlruns/`, `.dvc/config.local` all ignored before first commit.
- **Decisions logged:** D-013 (DVC remote via `--local` config), D-014 (`.venv` name, Python 3.13.3 used without pin issues).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 1 — data ingestion & integrity (loaders with runtime header/shape checks, feature+class join, edgelist parsing, label remap, DVC-track raw + assembled data).

### (template — copy for each session)
- **Date / session:**
- **Layer worked on:**
- **What changed:** (files added/edited, key functions)
- **Gate evidence:** (test output, metric, artifact path)
- **Decisions logged:** (D-numbers added)
- **Gate met?:** yes/no — if yes, approval requested from owner
- **Next action:**
