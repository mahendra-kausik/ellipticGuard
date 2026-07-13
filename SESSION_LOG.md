# SESSION_LOG.md — EllipticGuard

Single source of truth for "where are we?" Update at the end of every session and every time a layer gate is reached. Newest entry on top.

---

## Current state
- **Active layer:** Layer 1 — Data ingestion & integrity (COMPLETE, gate met)
- **Last gate passed:** Layer 1 — Data ingestion & integrity
- **Next action:** Read `PROJECT_PLAN.md` §6 Layer 2. Build `make_temporal_split()` (train 1–34, val carved from 30–34, test 35–49) + EDA + the no-cross-time-step-edges test.
- **Blockers / open questions:**
  - The preprocessing notebook (`notebooks/01_Elliptic_Preprocessing.ipynb`) uses a **different** split (train T1–34, val T35–36, test T37–49) than `PROJECT_PLAN.md` (train T1–34, val carved from the *tail* of train T30–34, test T35–49). Owner has confirmed this notebook was reused from a separate prior project on the same dataset, not authored for EllipticGuard — its split is stale reference material, not a bug to diagnose. Claude Code may edit the notebook freely. When Layer 2 starts, implement `make_temporal_split()` per `PROJECT_PLAN.md`'s partition (authoritative) and update/replace the notebook's split logic to match; no `DECISIONS.md` "corrected bug" entry needed since this isn't an owner error, just a straightforward implementation to build.

### Owner action items — things Claude Code cannot do
_(none right now — nothing blocking Layer 2)_

---

## Layer status
| Layer | Name | Status | Gate met? | Notes |
|---|---|---|---|---|
| 0 | Scaffolding & environment | complete | yes | venv (.venv), git init + remote, DVC init + local gdrive remote, mlflow dir, pytest smoke green |
| 1 | Data ingestion & integrity | complete | yes | loaders in `src/data/loaders.py`; DVC `assemble` stage; exact counts verified |
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

### 2026-07-13 — Layer 1
- **Layer worked on:** Layer 1 — Data ingestion & integrity
- **What changed:** `src/data/loaders.py` — `load_features`/`load_classes`/`load_edgelist` (each asserts header presence/absence and shape at runtime per PROJECT_PLAN.md §2), `assemble_node_table` (joins features+classes on `txId`, remaps `class` → `label` as illicit=1/licit=0/unknown=NaN, keeps unknown rows per D-016), `check_integrity` (row/label/time-step/null summary). `pipelines/assemble.py` — DVC stage entrypoint: loads raw CSVs, assembles, prints the sanity summary, writes `data/processed/{nodes,edges}.parquet`. `dvc.yaml` — added the `assemble` stage (raw CSVs + loader/pipeline code as deps, the two parquet files as outs). `tests/test_data_integrity.py` — asserts the exact gate counts.
- **Gate evidence:** `pytest -q` → `2 passed`. `dvc repro` → ran the `assemble` stage clean, printed: `n_nodes: 203769, n_edges: 234355, time_step_min: 1, time_step_max: 49, n_feature_cols: 165, null_counts_features: 0, illicit: 4545, licit: 42019, unknown: 157205` — matches PROJECT_PLAN.md §6 Layer 1 gate exactly. `dvc status` → "Data and pipelines are up to date." `dvc.lock` records md5 hashes for both raw inputs and processed outputs (DVC-tracked).
- **Decisions logged:** D-015 (165-vs-166 feature count reconciled as `time_step` + `feat_0..feat_164`), D-016 (unknown-labeled nodes kept in the assembled table, not dropped).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 2 — temporal split harness & EDA (`make_temporal_split()`, no-cross-time-step-edges test). Note the notebook-vs-plan split discrepancy recorded under "Blockers / open questions" above before implementing.

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
