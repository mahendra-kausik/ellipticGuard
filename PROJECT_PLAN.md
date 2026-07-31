# PROJECT_PLAN.md — EllipticGuard

An AML illicit-transaction detector on the Elliptic Bitcoin graph. Deployable, monitored, reproducible. Classical ML on causal graph features, honest temporal validation, FastAPI serving, drift monitoring, replayed retraining loop.

---

## 1. Goal & framing
Detect illicit Bitcoin transactions in a real, labeled transaction graph, and wrap the model in a full ML lifecycle. The resume signal is: graph feature engineering + imbalanced modeling + honest temporal validation + end-to-end MLOps, with a genuine concept-drift story (the time-step-43 darknet-market shutdown).

**Explicit honesty stance:** there is no live Bitcoin feed. The 49 time steps are fixed. "Retraining" is a **replay/simulation** of streaming, and we say so everywhere. The value we demonstrate is that monitoring *detects* drift and the loop *responds*; we also show the case where retraining does **not** recover (T43) and argue the correct response is human escalation.

---

## 2. Dataset specification (VERIFIED — confirm the two flagged items at load time)

Source: the Elliptic Bitcoin Dataset (Elliptic + MIT-IBM Watson AI Lab). Available on Kaggle (`ellipticco/elliptic-data-set`), Hugging Face, and via PyTorch Geometric (`EllipticBitcoinDataset`). ~200 MB.

Three files:

| File | Rows | Columns | Notes |
|---|---|---|---|
| `elliptic_txs_features.csv` | 203,769 | 167 | col 0 = `txId`; col 1 = time step (1–49); cols 2–166 = features. **This file has NO header in the canonical Kaggle version — load with `header=None`.** |
| `elliptic_txs_classes.csv` | 203,769 | 2 | `txId`, `class`. **Has a header.** |
| `elliptic_txs_edgelist.csv` | 234,355 | 2 | `txId1`, `txId2`. **Has a header.** Directed payment-flow edges. |

**Features (166 total):** 94 "local" features (transaction-level: number of inputs/outputs, fees, output volumes, and simple aggregates of the transaction's own inputs/outputs) + 72 "aggregated" features (one-hop neighbor aggregations — max/min/std/correlation of the local features over immediate neighbors). Features are anonymized; exact semantics are undisclosed. **Per the paper, the time step is the first local feature**, i.e. it lives in col 1 and is counted within the 166.

**Labels (`class`):** `1` = **illicit**, `2` = **licit**, `unknown` = unlabeled.
- Counts: **4,545 illicit**, **42,019 licit**, **~157,205 unknown**. Total labeled = **46,564**.
- **Imbalance — state this precisely:** illicit is ~2% of *all* nodes but **~9.8% of labeled nodes** (4,545 / 46,564). We train and evaluate on **labeled nodes only**, so the operative imbalance is ~9.8% positive / ~90.2% negative.
- Standard remap: illicit `1` → `1` (positive), licit `2` → `0` (negative), `unknown` → excluded from supervised train/eval (kept for optional graph-structure/semi-supervised use).

**Temporal structure:** 49 time steps, ~2 weeks apart. **Edges exist only *within* a time step** — each step is its own connected component; there are no cross-time-step edges. Consequence: graph-topology features computed per-time-step subgraph are **causal by construction** (a node's neighbors are always contemporaneous, never future). The only leakage surfaces left are (a) the train/test partition and (b) fitting any transform on non-train data.

**What this enables (the verification the owner asked for):**
- Time-step column ⇒ a defensible **temporal split** is possible. ✔
- Edgelist ⇒ **graph-topology feature engineering** is possible (degree, PageRank, clustering, component size). ✔
- Classes ⇒ **supervised labels** exist; no manual labeling needed. ✔
- All three requirements for this project are satisfied by the dataset as shipped.

---

## 3. Benchmarks to target (reference — VERIFY against arXiv:1908.02591, Weber et al. 2019, Table 1)
Illicit-class F1 on the temporal split, "all features" (AF):
- Logistic Regression ≈ 0.45–0.48
- **Random Forest ≈ 0.788** ← primary target for our classical model
- Random Forest + node embeddings ≈ 0.796
- GCN ≈ 0.628 · Skip-GCN ≈ 0.705 · EvolveGCN ≈ 0.720

Treat these as *reference points to reproduce and cite honestly*, not numbers to force. Report our own measured values.

---

## 4. Tech stack (rationale in `DECISIONS.md`)
- **Data/model versioning:** DVC + Google Drive remote.
- **Experiment tracking + model registry:** MLflow (local backend; registry via API).
- **Modeling:** scikit-learn (LR, RF), XGBoost; networkx (or igraph) for graph features.
- **Serving:** FastAPI + Docker → Google Cloud Run (`us-central1`, scale-to-zero). Amended from
  the original "Hugging Face Spaces (free CPU-basic)": HF stopped running Docker Spaces on its
  free tier (D-029). Cloud Run runs on the owner's existing GCP trial credit, not a pure free
  tier — a deliberate, time-limited deviation from this document's free-tier-only rule, recorded
  in D-034.
- **Monitoring:** Evidently (drift); NannyML CBPE (label-free perf estimation, optional).
- **CI/CD + retraining:** GitHub Actions.
- **Testing:** pytest.
- **Optional GNN comparison:** plain PyTorch (CPU, in-repo), not PyTorch Geometric on Colab — owner-approved deviation, see D-030.

---

## 5. Repository layout (target)
```
elliptic-guard/
├── CLAUDE.md  PROJECT_PLAN.md  DECISIONS.md  SESSION_LOG.md
├── .env.example  .gitignore  requirements.txt  README.md
├── Dockerfile       # root-level: HF Docker Spaces require it here (path not configurable)
├── dvc.yaml  params.yaml
├── data/            # DVC-tracked, not in git
├── src/
│   ├── data/        # loading, integrity, split
│   ├── features/    # graph construction + topology features
│   ├── models/      # train, evaluate, calibrate, registry
│   ├── monitoring/  # evidently / nannyml
│   └── serving/     # fastapi app
├── pipelines/       # dvc stage entrypoints
├── tests/
├── api/             # exported @production weights shipped into the image (D-028)
└── .github/workflows/
```

---

## 6. The layered build plan

> Rule: implement one layer, meet its gate, STOP and ask. Each layer adds ≥1 pytest test and any needed `DECISIONS.md` entries.

### Layer 0 — Scaffolding & environment
Build: repo structure, `requirements.txt`, `.gitignore`, `.env.example`, git init, DVC init (+ Drive remote configured but empty), MLflow tracking dir, empty `dvc.yaml`/`params.yaml`, pytest smoke test.
**Gate:** `pytest` runs green (1 trivial test); `dvc status` and `mlflow` import cleanly; repo committed.

### Layer 1 — Data ingestion & integrity
Build: download instructions/script; loaders that **confirm headers and shapes at runtime**; join features+classes into a node table; parse edgelist; remap labels; DVC-track raw + assembled data.
Sanity prints: row counts (expect 203,769 nodes / 234,355 edges), label counts (4,545 / 42,019 / ~157,205), time-step range (1–49), `X.shape[1]`, null counts.
**Gate:** an integrity test asserts the exact node/edge/label counts and time-step range; assembled table materialized and DVC-tracked. If any count mismatches, STOP and report.

### Layer 2 — Temporal split harness & EDA (CRITICAL)
Build: a single reusable `make_temporal_split()` → train (1–34), val (30–34 held out from train fitting), test (35–49); EDA on per-time-step counts and illicit rate; confirm no cross-time-step edges (assert it in a test).
**Gate:** test proves (a) no txId appears in more than one partition, (b) all test time steps > all train time steps, (c) zero edges cross time steps. The split function is the ONLY sanctioned way to partition data downstream.

### Layer 3 — Baseline models
Build: Logistic Regression + Random Forest on the **provided 166 features**, using the temporal split; scaler/imputer fit on **train only**; metrics = illicit-class precision/recall/F1 + AUC-PR (+ confusion matrix). Log runs to MLflow. Register the RF baseline as registry name `elliptic-illicit` **v1**.
**Gate:** MLflow shows both runs with AUC-PR and illicit F1; RF illicit-F1 is in a sane range (roughly ≥ 0.6 on AF; if far off, investigate before proceeding); v1 registered. Accuracy is NOT reported as a headline metric.

### Layer 4 — Graph construction & topology features
Build: per-time-step graph (networkx/igraph); compute causal features per node — in/out-degree, unique-neighbor count, PageRank, local clustering coefficient, connected-component size (within step), and 1-hop neighbor label-free aggregates; assemble an engineered feature table; DVC-track it.
Sanity prints: feature distributions, non-null coverage, that features for a step use only that step's subgraph.
**Gate:** a test confirms graph features for a sample of nodes match a hand-computed value on a tiny subgraph; feature table versioned; a short note in `DECISIONS.md` on why topology features add signal beyond the 72 aggregated features.

### Layer 5 — Advanced model & experiments
Build: XGBoost on provided + graph features; handle imbalance (class weights / `scale_pos_weight`); tune on the **val** slice only; compare feature sets (provided-only vs +graph) as MLflow experiments. Register the champion as **v2**.
**Gate:** champion beats the Layer-3 baseline on illicit-F1 and AUC-PR on the **test** set; improvement attributable to graph features is shown in the experiment comparison; v2 registered and tagged as the serving candidate. Target illicit-F1 ≈ 0.79 (report actual honestly).

### Layer 6 — Honest evaluation, calibration & the drift story
Build: per-time-step illicit-F1 curve on the test range → surface the **T43 collapse**; probability calibration (Platt/Isotonic fit on train/val only) + Brier score; optional SHAP (owner's notebook as reference, cleaned) for top-feature explanation.
**Gate:** the per-time-step curve is produced and the post-T43 degradation is visible and documented; calibration reported; `DECISIONS.md` entry states the honest interpretation (monitoring detects; retraining does not recover a regime change; correct response = escalation).

### Layer 7 — Serving API
Build: FastAPI app; endpoint accepts a transaction's feature vector (and optionally its 1-hop neighborhood) → returns illicit probability + calibrated score + model version; loads the **Production** model from the MLflow registry (not a hardcoded path); `/health` endpoint; Dockerfile; local run.
**Gate:** `POST /predict` returns a valid scored response locally for a known illicit and a known licit example; `/health` returns OK; a test hits the app via `TestClient`.

### Layer 8 (OPTIONAL) — GNN comparison
Build: clean re-implementation of GCN, GraphSAGE, and EvolveGCN in plain PyTorch, run **local CPU, in-repo** (not Colab/PyG — owner-approved deviation, D-030: the graph is small enough for full-batch CPU training in minutes, and in-repo means the GNNs import the exact same split, preprocessor, class weight, and `evaluate()` code as the classical champion, so fairness is provable by shared imports and reproducible via `dvc repro`); compare to the classical champion under the **same temporal split** and metric.
**Gate:** a fair, same-split comparison table (classical vs GCN vs GraphSAGE vs EvolveGCN); any notebook bugs fixed are logged in `DECISIONS.md`. If skipped, record the skip in `SESSION_LOG.md`.

### Layer 9 — Monitoring & observability
Build: Evidently feature-drift + target-drift reports across time steps (train as reference vs later steps as current); optional NannyML CBPE to estimate illicit performance without labels; structured logging + request latency capture + a `/metrics` endpoint on the API.
**Gate:** a drift report artifact is generated showing drift increasing toward the later steps; latency (p50/p95) is captured from the running API.

### Layer 10 — Retraining loop & CI/CD (replay)
Build: a replay driver that feeds time steps in sequence to simulate streaming; a drift-or-performance **flag** that fires when illicit-F1 on the newest labeled window drops below a threshold OR drift exceeds a bound; a GitHub Actions workflow that runs `pytest` + `dvc repro` + a champion/challenger check (promote new model only if it beats current on a held-out window; else reject); an F1-threshold **quality gate** that fails CI on regression.
**Gate:** CI runs green on a normal commit; the replay demonstrably fires the flag at/after T43; champion/challenger promotion logic is shown to promote in a routine case and reject/hold in the T43 case. All three behaviors captured in `SESSION_LOG.md`.

### Layer 11 — Deployment & documentation
Build: deploy the container to Google Cloud Run (amended from Hugging Face Spaces — D-029/D-034);
finalize `README.md` (architecture diagram, how to reproduce via `dvc repro`, results table,
honest limitations); ensure `DECISIONS.md` is complete; write the resume-metrics section.
**Gate:** a live Cloud Run URL serves predictions; README reproduces the pipeline from a clean clone (documented steps); decisions log complete.

---

## 7. Resume metrics (fill with real measured values at the end)
**Model / statistical:** illicit-class F1 (target ≈ 0.79), AUC-PR, precision & recall at the operating threshold, per-time-step F1 curve (with T43 annotation), calibration/Brier.
**System / MLOps:** inference latency p50/p95 (target < 100 ms CPU), model size (< 50 MB), pipeline reproducibility (`dvc repro` one-command), retraining cadence (per replayed step / on-flag), drift-detection lead time, CI build time, API `/health` uptime.

## 8. Free-tier caveats (re-verify before relying on limits)
- DVC-on-Google-Drive: ~15 GB on a personal account; API rate limits on large/many artifacts — keep artifact count modest.
- Google Cloud Run always-free tier: 2M requests/mo, 180,000 vCPU-s/mo, 360,000 GiB-s/mo (select
  US regions) — enough for an idle scale-to-zero demo at $0. **The EllipticGuard deployment does
  not rely on this tier**: it runs under a billing-enabled project on trial credit, which is a
  documented, time-limited exception (D-034), not a claim that Cloud Run itself is free-tier-only
  here.
- GitHub Actions: free minutes are ample for this CPU pipeline; keep the CI job lean.
- MLflow registry stages work best against a backed tracking server; on a local file store, use the registry API and note in `DECISIONS.md` that a hosted tracking server is the production upgrade.
