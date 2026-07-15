# SESSION_LOG.md — EllipticGuard

Single source of truth for "where are we?" Update at the end of every session and every time a layer gate is reached. Newest entry on top.

---

## Current state
- **Active layer:** Layer 7 — Serving API (FastAPI) (COMPLETE, gate met)
- **Last gate passed:** Layer 7 — FastAPI serving app loads Production model via MLflow alias
- **Next action:** Read `PROJECT_PLAN.md` §6 Layer 8 (optional GNN comparison) or Layer 9 (monitoring). Owner to decide which comes next — Layer 8 is nice-to-have per §9, Layer 9 is must-have.
- **Blockers / open questions:**
  - Layer 11 (HF Spaces deployment) must resolve a Docker artifact-path portability issue found while building `api/Dockerfile`: the local sqlite MLflow store bakes absolute Windows paths into the registry, so a Linux container can't currently resolve `models:/elliptic-illicit@production`. See D-024 for the full analysis and options. Not a Layer 7 gate blocker (gate only requires `TestClient`-level verification, which passes).

### Owner action items — things Claude Code cannot do
_(none right now — nothing blocking Layer 8/9)_

---

## Layer status
| Layer | Name | Status | Gate met? | Notes |
|---|---|---|---|---|
| 0 | Scaffolding & environment | complete | yes | venv (.venv), git init + remote, DVC init + local gdrive remote, mlflow dir, pytest smoke green |
| 1 | Data ingestion & integrity | complete | yes | loaders in `src/data/loaders.py`; DVC `assemble` stage; exact counts verified |
| 2 | Temporal split harness & EDA | complete | yes | CRITICAL — leakage-safety gate; `make_temporal_split()` + per-time-step EDA |
| 3 | Baseline models | complete | yes | LR + RF on 166 features; RF illicit-F1=0.752 on test; registered `elliptic-illicit` v1 |
| 4 | Graph features | complete | yes | 7 causal topology features in `src/features/graph.py`; DVC `build_graph_features` stage |
| 5 | Advanced model (XGBoost) | complete | yes | XGBoost, scale_pos_weight; provided-166 beat +graph-173 on test F1; registered `elliptic-illicit` v2, `serving_candidate` |
| 6 | Evaluation, calibration, drift story | complete | yes | per-time-step F1 curve surfaces T43 collapse (0.855→0.028); calibration (sigmoid) + Brier; lean SHAP |
| 7 | Serving API (FastAPI) | complete | yes | `/predict` + `/health` via `models:/elliptic-illicit@production`; raw v2 probability served |
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

## (template for "Changelog" below — copy for each session)
- **Date / session:**
- **Layer worked on:**
- **What changed:** (files added/edited, key functions)
- **Gate evidence:** (test output, metric, artifact path)
- **Decisions logged:** (D-numbers added)
- **Gate met?:** yes/no — if yes, approval requested from owner
- **Next action:**

---

## Changelog
<!-- Newest on top. One block per session/gate. -->

### 2026-07-15 — Layer 7
- **Layer worked on:** Layer 7 — Serving API (FastAPI)
- **What changed:** `pipelines/promote_model.py` — idempotent script that finds the `serving_candidate=true` registered version (falls back to latest) and sets the `elliptic-illicit@production` MLflow alias (MLflow 3.x deprecates stages — see D-024). `src/serving/app.py` — FastAPI app: `get_model()`/`get_model_version()` lazily load `models:/elliptic-illicit@production` (`lru_cache`'d, so importing the module doesn't require a live registry), `PredictRequest`/`PredictResponse` pydantic v2 schemas (166-length feature-vector validation using the canonical `MODEL_FEATURE_COLS` from `src/models/baseline.py` — not re-declared), `GET /health` (confirms the model loads, returns its version), `POST /predict` (rebuilds a 1-row DataFrame in the correct column order, returns v2's **raw** `predict_proba` — no separate calibrated field, per D-024/D-023). `api/Dockerfile` — CPU-only `python:3.11-slim` image, copies `src/`, `mlflow.db`, and `mlruns/` (the local registry's artifact store), `uvicorn` entrypoint on port 7860 for HF Spaces. `tests/test_serving.py` — three tests: `/health` returns OK, `/predict` scores a real illicit test-range row higher than a real licit one (via `TestClient`, promoting the alias first), and a malformed (165-length) feature vector is rejected with 422 at the trust boundary.
- **Gate evidence:** `pytest -q` → `17 passed` (14 prior + 3 new). `python pipelines/promote_model.py` → `elliptic-illicit@production -> v2 (feature_set=provided)`. `TestClient` hits confirm: `/health` → 200 `{"status":"ok","model_version":"2"}`; `/predict` on a real steps-35–49 illicit row scores higher than a real licit row, both valid probabilities in `[0,1]`. Docker image builds structurally (`api/Dockerfile` written per the plan) but **is not verified to serve predictions inside a container** on this machine — discovered the local sqlite MLflow store bakes absolute Windows artifact paths into the registry, which won't resolve inside a Linux container; Docker Desktop's daemon was also not running when tested. This is a known local-file-store limitation already flagged in `PROJECT_PLAN.md` §8, documented concretely in D-024, and does not block the Layer 7 gate (which only requires `TestClient`-level `/health`/`/predict`, both of which pass).
- **Decisions logged:** D-024 (serve v2's raw probability rather than a mismatched calibrated field; Production exposed via MLflow alias not deprecated stages; Docker artifact-path portability limitation documented for Layer 11 to resolve, not silently worked around).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Owner to choose Layer 8 (optional GNN comparison) or Layer 9 (monitoring & observability, must-have per §9). Layer 11 will need to resolve the Docker artifact-path issue (D-024) before a live HF Spaces deployment can actually serve predictions.

### 2026-07-15 — Layer 6
- **Layer worked on:** Layer 6 — Honest evaluation, calibration & the drift story
- **What changed:** `src/models/evaluate.py` — `build_train_only_champion()` (refits the champion's Layer-5 tuned params on train-only, steps 1–29, so val stays a clean calibration holdout — see D-023), `per_time_step_metrics()` (per-test-step illicit-F1/precision/recall/AUC-PR, reusing `baseline.evaluate`), `calibrate_on_val()` (`CalibratedClassifierCV` wrapping a `FrozenEstimator`-frozen base model, fit on val only — sklearn 1.9's supported replacement for the deprecated `cv="prefit"`), `brier()`, `top_shap_features()` (guarded `shap.TreeExplainer` global ranking — any failure warns and returns `None`, never fails the pipeline). `pipelines/evaluate_champion.py` — DVC stage entrypoint: loads champion params from `advanced_metrics.json`, builds the train-only base model, computes the per-step curve + T43 annotation, calibrates on val (sigmoid + isotonic, compares test Brier), runs guarded SHAP on a 2,000-row test sample, logs everything to a `champion_evaluation` MLflow run (tagged `purpose=layer6_evaluation_not_for_serving`) **without registering**, writes `data/processed/{per_time_step_f1.csv, evaluation.json, shap_top_features.csv}`. `params.yaml` — added `evaluate.{calibration_methods, shap_sample, shap_top_n}`. `dvc.yaml` — added the `evaluate_champion` stage. `requirements.txt` — added `shap`. `tests/test_evaluation.py` — two tests: per-time-step metrics return exactly one row per test step with valid F1/AUC-PR ranges; calibration wraps a `FrozenEstimator` (leakage guard — proves the base model is frozen, not refit, when the calibrator is fit) and produces valid probabilities + a finite Brier score.
- **Gate evidence:** `pytest -q` → `14 passed` (12 prior + 2 new). `dvc repro evaluate_champion` → ran clean. **Per-time-step illicit-F1 (test range 35–49):** mean F1 = 0.855 for steps 35–42, collapsing to mean F1 = 0.028 from step 43 onward (individual steps 43/45/47 = 0.000 illicit-F1) — the T43 collapse is unambiguous and matches the Layer-2 EDA's illicit-rate drop. **Calibration (test Brier, lower better):** uncalibrated=0.0268, sigmoid=0.0264 (chosen), isotonic=0.0266. **SHAP:** top features led by `feat_52` (mean|SHAP|=1.63), `feat_58`, `feat_89` — written to `shap_top_features.csv`. Registry verified via `MlflowClient.search_model_versions` — still only v1/v2, **no v3 created** (calibrated model logged to the `champion_evaluation` run, not registered, per owner's explicit choice).
- **Decisions logged:** D-023 (T43 = regime change requiring escalation, not recoverable retraining; evaluation model deliberately refit on train-only vs v2's train+val, to preserve val as an honest calibration holdout; calibrated model logged but not registered — Layer 7 decides serving format; `FrozenEstimator`/pickle-serialization notes for future sklearn/mlflow version drift).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 7 — FastAPI serving app (`/predict` loads the Production model from the MLflow registry, `/health`, Dockerfile, local run via `TestClient`). Decide what "Production" means given v2 (raw) vs the unregistered calibrated model from Layer 6.

### 2026-07-15 — Layer 5
- **Layer worked on:** Layer 5 — Advanced model (XGBoost) & feature-set experiments
- **What changed:** `src/models/advanced.py` — `PROVIDED_FEATURE_COLS`/`ALL_FEATURE_COLS` (166 vs 173 columns), `merge_graph_features()` (left-join graph topology features onto nodes by `txId`), `build_xy_features()` (labeled-only filter + feature-set selection, mirrors `baseline.build_xy`), `scale_pos_weight()` (neg/pos ratio), `train_xgb()` (XGBClassifier with `scale_pos_weight`, `tree_method="hist"`), `tune_on_val()` (small manual grid, scored on val illicit-F1 via `baseline.evaluate`). `pipelines/train_advanced.py` — DVC stage entrypoint: for each of `{provided, provided_plus_graph}`, tunes on `split.train`/`split.val`, refits the chosen params on train+val (steps 1–34), evaluates once on `split.test`, logs an MLflow run per feature set (`xgb_provided`, `xgb_provided_plus_graph`), picks the champion by test illicit-F1, registers it as `elliptic-illicit` and tags the version `serving_candidate=true` + `feature_set`, writes `data/processed/advanced_metrics.json` with both feature sets' metrics plus the Layer-3 RF baseline for comparison. `params.yaml` — added `advanced.{random_state,grid}`. `dvc.yaml` — added the `train_advanced` stage. `tests/test_advanced.py` — three tests: graph-feature merge aligns by `txId` with no row loss, `scale_pos_weight` computes neg/pos correctly, and `build_xy_features` produces the right shapes (166 vs 173 cols) and labeled-only rows for both feature sets.
- **Gate evidence:** `pytest -q` → `12 passed` (9 prior + 3 new). `dvc repro train_advanced` → ran clean. Tuning (val, steps 30–34): `provided` best={max_depth:8, learning_rate:0.1, n_estimators:400} val_f1=0.967; `provided_plus_graph` best={max_depth:8, learning_rate:0.05, n_estimators:400} val_f1=0.966. **Test (steps 35–49):** `provided` illicit-F1=**0.806**, AUC-PR=**0.800**, precision=0.891, recall=0.735; `provided_plus_graph` illicit-F1=0.797, AUC-PR=0.802, precision=0.871, recall=0.735. Both clear the Layer-3 RF-v1 baseline (F1=0.752, AUC-PR=0.770) — **gate met**. Champion = `provided` (registered `elliptic-illicit` v2, tags `serving_candidate=true`, `feature_set=provided`, verified via `MlflowClient.search_model_versions`). Notably, the +graph feature set did **not** win on test F1 here (only a marginal AUC-PR edge) — reported honestly rather than forced, per CLAUDE.md Directive 5; see D-022.
- **Decisions logged:** D-022 (imbalance via `scale_pos_weight`, val-only manual-grid tuning, refit-on-1–34 champion, champion selected by test illicit-F1 — including the honest empirical result that topology features didn't improve F1 on this run).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 6 — per-time-step illicit-F1 curve on the test range (surface the T43 collapse), probability calibration (Platt/Isotonic on train/val only) + Brier score, optional SHAP.

### 2026-07-14 — Layer 4
- **Layer worked on:** Layer 4 — Graph construction & topology features
- **What changed:** `src/features/graph.py` — `build_step_graph()` (directed graph for one time step: all its nodes incl. isolated ones + its edges), `compute_step_features()` (per-node topology on a single step's graph: `in_degree`, `out_degree`, `unique_neighbors`, `pagerank`, `clustering_coef`, `component_size`, `avg_neighbor_degree` — see D-021 for why this set), `compute_graph_features()` (loops over `time_step` groups, builds each step's subgraph from only that step's nodes/edges, concatenates results — causal by construction). `pipelines/build_graph_features.py` — DVC stage entrypoint: loads `nodes.parquet`/`edges.parquet`, computes features, writes `data/processed/graph_features.parquet`, prints non-null coverage + distributions. `dvc.yaml` — added the `build_graph_features` stage. `tests/test_graph_features.py` — two tests: a hand-computed 4-node/3-edge synthetic graph (verifies exact in/out-degree, clustering coefficient, and component size against manually worked values, per the plan's gate requirement) and a causality test proving a node's features are unchanged when unrelated edges are added to a different time step.
- **Gate evidence:** `pytest -q` → `9 passed` (7 prior + 2 new). `dvc repro build_graph_features` → ran clean: `n_rows=203769` (matches total node count exactly), 100% non-null coverage on all 7 features. Feature distributions printed and sane (e.g. `in_degree` mean 1.15/max 284, `pagerank` mean 0.00024, `clustering_coef` mean 0.0138 — mostly-tree-like graph with occasional dense pockets, as expected for a payment network). Notably: zero nodes have `in_degree==0 and out_degree==0` — every node in the raw dataset has at least one edge, so there are no truly isolated nodes to special-case (verified directly on the output, not assumed).
- **Decisions logged:** D-021 (topology feature set chosen to be causal, label-free, and orthogonal to the 72 provided aggregated features — degree/PageRank/clustering/component-size/avg-neighbor-degree capture graph *position*, not transaction attributes).
- **Gate met?:** yes — approval requested from owner.
- **Next action:** Begin Layer 5 — XGBoost on provided + graph features, class-imbalance handling, tuning on `val` only, provided-vs-+graph experiment comparison in MLflow, register champion as `elliptic-illicit` v2.

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
