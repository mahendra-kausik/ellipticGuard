# SESSION_LOG.md — EllipticGuard

Single source of truth for "where are we?" Update at the end of every session and every time a layer gate is reached. Newest entry on top.

---

## Current state
- **Active layer:** Layer 3 — Baseline models (COMPLETE, gate met)
- **Last gate passed:** Layer 3 — Baseline models
- **Next action:** Read `PROJECT_PLAN.md` §6 Layer 4. Build per-time-step graphs (networkx) and causal topology features (degree, PageRank, clustering coefficient, component size, 1-hop label-free aggregates); assemble an engineered feature table; DVC-track it.
- **Blockers / open questions:**
  - None currently open.

### Owner action items — things Claude Code cannot do
_(none right now — nothing blocking Layer 4)_

---

## Layer status
| Layer | Name | Status | Gate met? | Notes |
|---|---|---|---|---|
| 0 | Scaffolding & environment | complete | yes | venv (.venv), git init + remote, DVC init + local gdrive remote, mlflow dir, pytest smoke green |
| 1 | Data ingestion & integrity | complete | yes | loaders in `src/data/loaders.py`; DVC `assemble` stage; exact counts verified |
| 2 | Temporal split harness & EDA | complete | yes | CRITICAL — leakage-safety gate; `make_temporal_split()` + per-time-step EDA |
| 3 | Baseline models | complete | yes | LR + RF on 166 features; RF illicit-F1=0.752 on test; registered `elliptic-illicit` v1 |
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

### 2026-07-14 — Layer 3
- **Layer worked on:** Layer 3 — Baseline models
- **What changed:** `src/models/baseline.py` — `MODEL_FEATURE_COLS` (the provided 166 features: `time_step` + `feat_0..feat_164`), `build_xy()` (filters to labeled rows per D-002), `fit_preprocessor()` (median imputer + standard scaler, fit on `split.train` only), `train_logistic_regression()`/`train_random_forest()` (both `class_weight="balanced"` per D-008), `evaluate()` (illicit-class precision/recall/F1 + AUC-PR + confusion matrix — no accuracy, per D-003). `pipelines/train_baseline.py` — DVC stage entrypoint: builds the temporal split, fits the preprocessor on train only, trains both models, logs params/metrics/models to MLflow, registers the RF run as `elliptic-illicit` v1, writes `data/processed/baseline_metrics.json`. `params.yaml` — added `baseline.{random_state,lr_max_iter,rf_n_estimators,rf_max_depth}`. `dvc.yaml` — added the `train_baseline` stage. `tests/test_baseline.py` — three tests: labeled-only filtering, preprocessor-fit-only-on-train (leakage-safety — proves a distribution-shifted partition doesn't move the fitted scaler), and a basic train/predict shape check.
- **Gate evidence:** `pytest -q` → `7 passed` (4 prior + 3 new). `dvc repro train_baseline` → ran clean. Labeled train: n=26,381 (10.9% illicit); labeled test: n=16,670 (6.5% illicit). **LR test:** illicit-F1=0.245, AUC-PR=0.204, precision=0.140, recall=0.947. **RF test:** illicit-F1=0.752, AUC-PR=0.770, precision=0.849, recall=0.675 — comfortably above the ≥0.6 sanity bound and close to Weber et al.'s reference RF-AF figure (≈0.788), reported honestly as our own measured value, not adjusted to match. MLflow (`sqlite:///mlflow.db`, experiment `elliptic-aml`) shows both runs with logged params/metrics/models; `elliptic-illicit` v1 registered from the RF run (verified via `MlflowClient.search_model_versions`).
- **Decisions logged:** D-020 (MLflow 3.x's file store is in maintenance mode; switched the local backend default to `sqlite:///mlflow.db`, corrected `.env.example` and `.gitignore` accordingly).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 4 — per-time-step graph construction + causal topology features (in/out-degree, unique neighbors, PageRank, clustering coefficient, component size, 1-hop label-free aggregates), assembled into a DVC-tracked engineered feature table.

### 2026-07-13 — Layer 2
- **Layer worked on:** Layer 2 — Temporal split harness & EDA (CRITICAL leakage-safety gate)
- **What changed:** `src/data/split.py` — `TemporalSplit` dataclass + `make_temporal_split()` (train = steps 1–29 only, the fit-only slice; val = 30–34, held out from fitting; test = 35–49 — see D-019 for why `train` excludes the val tail) and `split_edges()` (assigns each edge to a partition via its `txId1` endpoint's time step). `src/data/eda.py` — `compute_time_step_stats()` (per-time-step node/illicit/licit/unknown counts + illicit rate over labeled nodes only). `pipelines/split_eda.py` — DVC stage entrypoint: loads `data/processed/{nodes,edges}.parquet`, runs the split + EDA, prints partition sizes/time-step ranges, writes `data/processed/eda_per_time_step.csv`. `params.yaml` — added `split.{train_end,val_start,test_start,test_end}` so the partition boundaries are versioned, not hardcoded. `dvc.yaml` — added the `split_eda` stage. `tests/test_temporal_split.py` — two tests: no-cross-time-step-edges (proves the causal-graph premise), and split disjointness/ordering (no txId in >1 partition, train < val < test time-step ranges, edge counts sum to the total).
- **Gate evidence:** `pytest -q` → `4 passed` (2 from Layer 1 + 2 new). `dvc repro` → ran `split_eda` clean: `train: n_nodes=120804, n_edges=140223, range=[1,29]`; `val: n_nodes=15461, n_edges=16620, range=[30,34]`; `test: n_nodes=67504, n_edges=77512, range=[35,49]` — sums to 203,769 nodes / 234,355 edges, matching Layer 1's totals exactly. Per-time-step EDA table printed and written to `data/processed/eda_per_time_step.csv`; a visible illicit-count/rate drop appears at T43 (24 illicit / 1.75% rate vs. a generally higher/noisier rate before it) — the known dataset collapse, to be formally surfaced in Layer 6.
- **Decisions logged:** D-019 (`TemporalSplit.train` means the fit-only 1–29 slice, not the full 1–34 range; split boundaries externalized to `params.yaml`).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 3 — baseline models (Logistic Regression + Random Forest on the 166 provided features, `split.train`-only scaler/imputer fit, MLflow logging, register RF as `elliptic-illicit` v1).

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
