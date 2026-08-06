# CODE_TOUR.md — flow-by-flow reading index

This is a table of contents, not a walkthrough. At the start of a session, say
which flow to trace (e.g. "walk me through Flow 3") and Claude reads that
flow's entry below, opens the listed files in order, and traces the real code
live — cross-referencing the `DECISIONS.md` entries and tests named. Flows are
independent and can be taken in any order; a reasonable first pass is
**1 → 2 → 3 → 5 → 6 → 4 → 7 → 8**.

Other docs answer different questions — this one only answers "where is the
code and what order do I read it in":

| Doc | Answers |
|---|---|
| `PROJECT_PLAN.md` | build order + per-layer acceptance gates |
| `SESSION_LOG.md` | current status, what changed last session |
| `DECISIONS.md` | why each non-trivial choice was made |
| `README.md` | results pitch, headline numbers |
| `FLASHCARDS.md` | interview Q&A |
| **`CODE_TOUR.md`** | **where the code is, read in what order** |

## Flows at a glance

| # | Flow | Layers | Entry pipeline |
|---|---|---|---|
| 1 | Data ingestion & integrity | 1 | `assemble` |
| 2 | Temporal split & EDA | 2 | `split_eda` |
| 3 | Graph features | 4 | `build_graph_features` |
| 4 | Models: baseline → champion | 3, 5 | `train_baseline`, `train_advanced` |
| 5 | Evaluation, calibration & SHAP | 6 | `evaluate_champion` |
| 6 | Monitoring: the T43 story | 9 | `monitor_drift` |
| 7 | Retraining replay & CI gate | 10 | `replay_retraining` |
| 8 | Serving & deployment | 7, 11 | `src/serving/app.py`, `Dockerfile` |
| 8b | GNN comparison (optional) | 8 | `train_gnn` |

---

## Flow 1 — Data ingestion & integrity

**Covers:** Layer 1, DVC stage `assemble`

**Entrypoint → modules → outputs:**
`pipelines/assemble.py` → `src/data/loaders.py` (`load_features`, `load_classes`,
`load_edgelist`, `assemble_node_table`, `check_integrity`) → `data/processed/nodes.parquet`,
`data/processed/edges.parquet`

**Key decisions:** D-004 (load raw CSVs directly), D-015 (165-vs-166 feature
reconciliation), D-016 (unknown-labeled nodes kept, not dropped), D-002
(evaluate on labeled nodes only)

**Tests that guard it:** `tests/test_data_integrity.py`

**Questions this flow answers:**
- Why validate CSV shape/headers at load time instead of trusting the file?
- Why is `time_step` counted as the model's 166th feature?
- Why does the label map send "unknown" to NaN instead of 0 (licit)?
- Why does `MODEL_FEATURE_COLS` live in `src/data/loaders.py` and not
  `src/models/baseline.py`?

---

## Flow 2 — Temporal split & EDA

**Covers:** Layer 2 (CRITICAL), DVC stage `split_eda`

**Entrypoint → modules → outputs:**
`pipelines/split_eda.py` → `src/data/split.py` (`make_temporal_split`,
`split_edges`), `src/data/eda.py` (`compute_time_step_stats`), `params.yaml:split`
→ `data/processed/eda_per_time_step.csv`

**Key decisions:** D-001 (temporal split, never random), D-019 (`train` means
the fit-only slice, steps 1–29, not the full 1–34 range), D-002 (labeled-only
rate denominator)

**Tests that guard it:** `tests/test_temporal_split.py`

**Questions this flow answers:**
- Why is val carved from steps 30–34 (tail of train) instead of the front or a
  random sample?
- What's the difference between "train" (1–29, fit-only) and "the training
  range" (1–34)?
- Why does `split_edges` use only `txId1` to assign an edge to a partition, and
  what test proves that's safe?
- Why is the illicit rate divided by (illicit + licit), not total nodes?

---

## Flow 3 — Graph features

**Covers:** Layer 4, DVC stage `build_graph_features`

**Entrypoint → modules → outputs:**
`pipelines/build_graph_features.py` → `src/features/graph.py`
(`build_step_graph`, `compute_step_features`, `compute_graph_features`) →
`data/processed/graph_features.parquet`

**Key decisions:** D-005 (graph-topology features on top of the 72 aggregated
features), D-006 (causal features by construction, no cross-time aggregation),
D-021 (the 7 causal, label-free features chosen)

**Tests that guard it:** `tests/test_graph_features.py`

**Questions this flow answers:**
- Why is building one graph per time step "causal by construction" rather than
  just "carefully causal"?
- Why are some features computed on the directed graph and others on the
  undirected version?
- Why are nodes added to the graph before edges (what breaks if you don't)?
- Why did these features end up *not* winning (0.797 vs 0.806 F1), and what's
  the honest explanation? (See also Flow 4 for where this gets measured.)

---

## Flow 4 — Models: baseline → champion

**Covers:** Layers 3 and 5, DVC stages `train_baseline`, `train_advanced`

**Entrypoint → modules → outputs:**
`pipelines/train_baseline.py` → `src/models/baseline.py` (LR, RF) →
`data/processed/baseline_metrics.json`
`pipelines/train_advanced.py` → `src/models/advanced.py` (XGBoost, grid search,
graph-feature variant) → `data/processed/advanced_metrics.json`

**Key decisions:** D-007 (XGBoost as champion), D-008 (class weighting, not
resampling), D-022 (`scale_pos_weight`, val-only tuning, refit on 1–34, champion
picked by test illicit-F1), D-003 (illicit-F1 + AUC-PR, never accuracy)

**Tests that guard it:** `tests/test_baseline.py`, `tests/test_advanced.py`

**Questions this flow answers:**
- Why class weighting over SMOTE/resampling for a ~9.8% imbalance?
- Why is the grid searched on val, never test, and what would leak if it
  weren't?
- Why is the final champion refit on the full 1–34 range after tuning?
- Why did provided-only features beat provided+graph, and why report that
  instead of dropping the losing experiment?

---

## Flow 5 — Evaluation, calibration & SHAP

**Covers:** Layer 6, DVC stage `evaluate_champion`

**Entrypoint → modules → outputs:**
`pipelines/evaluate_champion.py` → `src/models/evaluate.py` (shared `evaluate()`
used by classical models and the GNNs alike, calibration, SHAP) →
`data/processed/per_time_step_f1.csv`, `data/processed/evaluation.json`

**Key decisions:** D-003 (metric choice), D-009 (probability calibration +
Brier score), D-023 (T43 identified as a regime change; evaluation model
refit train-only; calibrated model logged but not registered)

**Tests that guard it:** `tests/test_evaluation.py`

**Questions this flow answers:**
- Why does one `evaluate()` function serve every model, including the GNNs in
  Flow 8b — what does sharing it buy you?
- Why is the calibrator fit on val with `cv="prefit"` rather than refit end to
  end?
- Where does the per-time-step F1 curve first surface the T43 collapse, before
  Flow 6 investigates why?
- Why serve a raw probability instead of the (marginally better) calibrated
  one? (Ties into Flow 8.)

---

## Flow 6 — Monitoring: the T43 story

**Covers:** Layer 9, DVC stage `monitor_drift`

**Entrypoint → modules → outputs:**
`pipelines/monitor_drift.py` → `src/monitoring/drift.py` (feature drift, target
drift, Evidently) → `data/processed/drift_by_time_step.csv`,
`data/processed/drift_report.html`

**Key decisions:** D-025 (feature vs target drift measured separately,
`time_step` excluded from drift-tested columns), D-033 (NannyML CBPE declined —
its no-concept-drift assumption is exactly what T43 violates), D-012
(retraining framed as replay, T43 framed honestly)

**Tests that guard it:** `tests/test_monitoring.py`

**Questions this flow answers:**
- Why split feature drift from target drift instead of one combined score?
- What exactly stayed flat and what exactly spiked at step 43, and why does
  that gap matter?
- Why would a label-free monitor (like NannyML CBPE) have missed this
  collapse entirely?
- What real-world event does T43 correspond to, and why does that make it a
  regime change rather than staleness?

---

## Flow 7 — Retraining replay & CI gate

**Covers:** Layer 10, DVC stage `replay_retraining`

**Entrypoint → modules → outputs:**
`pipelines/replay_retraining.py` → `src/retraining/replay.py` (drift-or-
performance flag, champion/challenger promotion) → `data/processed/replay_log.csv`,
`metrics/quality_gate.json`

**Key decisions:** D-026 (replay champion reuses the Layer 6 train-only model;
in-sample challenger eval rejected as leakage; lean data-free CI), D-012
(replay framing), D-027 (`pythonpath = .` CI fix)

**Tests that guard it:** `tests/test_retraining.py`

**Questions this flow answers:**
- What is the drift-or-performance flag, and what two things does it OR
  together (`params.yaml:retrain`)?
- Why does the champion/challenger correctly *promote* at step 39 but *hold*
  at steps 43 and 45?
- Why would evaluating a challenger in-sample be leakage, and what's done
  instead?
- What does the GitHub Actions quality gate actually check, and why is it
  data-free?

---

## Flow 8 — Serving & deployment

**Covers:** Layers 7 and 11

**Entrypoint → modules → outputs:**
`src/serving/app.py` (`/predict`, `/health`, `/metrics`) ← `pipelines/export_model.py`
(registry alias → `api/model/` weights) ← `pipelines/promote_model.py` (sets
`elliptic-illicit@production`) → `Dockerfile`, `requirements-serve.txt` →
container / Cloud Run

**Key decisions:** D-010 (MLflow tracking + registry), D-020 (sqlite tracking
backend), D-024 (serve raw probability, no separate calibrated field; local
sqlite registry stores absolute paths), D-028 (container loads via `MODEL_URI`
path, not the registry, and why), D-029 (HF Spaces free tier dropped Docker),
D-034 (Cloud Run on trial credit; 919 MB → 300 MB image slim)

**Tests that guard it:** `tests/test_serving.py`, `tests/test_serving_loader.py`

**Questions this flow answers:**
- Why can't the deployed container just query the MLflow registry directly —
  what's baked into `mlflow.db` that breaks it?
- What's the actual difference between `requirements.txt` and
  `requirements-serve.txt`, and how did that cut image size 919→300 MB?
- Why does `/predict` return the raw probability instead of the
  slightly-better-calibrated one from Flow 5?
- What does `/metrics` track, and why an in-process deque instead of an
  external time-series store?
- Why Cloud Run instead of the originally planned Hugging Face Spaces, and
  what's the practical consequence (time-limited demo)?

---

## Flow 8b — GNN comparison (optional, Layer 8)

**Covers:** Layer 8, DVC stage `train_gnn` — off the critical path, nice-to-have

**Entrypoint → modules → outputs:**
`pipelines/train_gnn.py` → `src/models/gnn.py` (hand-rolled GCN, GraphSAGE,
EvolveGCN in plain PyTorch) → `data/processed/gnn_metrics.json`,
`data/processed/gnn_per_time_step.csv`

**Key decisions:** D-030 (plain PyTorch in-repo, notebook comparison rejected
not adapted), D-031 (GNNs measurably lose to XGBoost; GCN under the notebook's
published figure, and why that's real not a bug), D-032 (epoch/patience budget
raised 60/8 → 200/20; GCN and GraphSAGE improve, EvolveGCN doesn't)

**Tests that guard it:** `tests/test_gnn.py`

**Questions this flow answers:**
- Why plain PyTorch instead of PyTorch Geometric?
- What does "identical budget across all three models" buy as a fairness
  argument, and where does `params.yaml:gnn` enforce it?
- Why does raising the epoch budget help GCN/GraphSAGE but not EvolveGCN —
  what does that say about EvolveGCN's ceiling?
- Do the GNNs also collapse at T43, and what does that corroborate?

---

## Cross-cutting threads

Things that don't belong to one flow — worth noticing as they recur:

- **Leakage discipline.** Every flow that fits something (scaler, model,
  calibrator, threshold) fits on train or val only, never test. Traced clearest
  in Flows 2, 4, and 5.
- **`params.yaml` + `dvc.yaml` as the reproducibility spine.** Every pipeline
  stage declares its deps and params explicitly; changing a param invalidates
  and reruns only the affected downstream stages (`dvc repro`).
- **The honesty pattern.** Negative results are kept and reported, not hidden:
  graph features losing (Flow 3/4), GNNs losing (Flow 8b), NannyML declined
  twice (Flow 6), the deployment's registry-vs-path limitation stated plainly
  (Flow 8).
- **`needs_data` test marker.** Splits tests that require the real dataset
  (local only) from the data-free subset CI actually runs
  (`pytest -m "not needs_data"`).
