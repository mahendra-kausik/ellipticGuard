# ProjectInfo.md — EllipticGuard resume fact pack

> **How to use this file:** Paste this together with a target **job description**, and ask Claude to write resume bullets for that role. Instructions for the writer:
> - Match the JD's keywords/skills against the **Framing angles by role** and **Skills demonstrated** sections; lead with the facts that fit.
> - Every number here is a real measured value. **Use them verbatim — never round, invent, or inflate.** If a stat isn't here, don't claim it.
> - Prefer quantified bullets (metric + what you did + why it matters).
> - Respect the honesty stance: retraining is a **replayed/simulated** stream over fixed historical data, not a live feed. Don't imply a live production system or a live URL.
> - 3–5 bullets is usually right for one project on a resume; pick the strongest for the role, don't dump everything.

---

## One-liner
An anti-money-laundering (AML) illicit-Bitcoin-transaction detector on the Elliptic dataset, built as a deployable, monitored, reproducible ML system — not a notebook.

## 30-second pitch
Classical ML on a 200K-node Bitcoin transaction graph, classifying illicit vs. licit transactions under a **strict temporal split** (no leakage). Champion XGBoost hits **illicit-class F1 = 0.806** on a strictly future test window. Wrapped in a full lifecycle: DVC-reproducible pipeline, MLflow registry, FastAPI/Docker serving, Evidently drift monitoring, and a champion/challenger replay retraining loop. The centrepiece is a genuine **concept-drift story** (a darknet-market shutdown at time step 43) that the model can't survive, monitoring detects, and retraining *cannot* fix — arguing the correct response is human escalation.

---

## Core narrative (what makes it interview-defensible)
- **Leakage prevention is the whole point.** Temporal split: train = steps 1–34, validation = 30–34 (carved from the *end* of train, never the future), test = 35–49. No random split anywhere. Every scaler/encoder/calibrator fit on train only. Graph features computed within each time-step subgraph (edges never cross steps), so topology features are causal by construction.
- **Imbalanced classification done honestly.** ~9.8% illicit over labeled nodes. Illicit-class F1/AUC-PR are the headline metrics; **accuracy is never reported** because ~90% of labeled nodes are licit.
- **The T43 concept-drift story.** Per-time-step F1 collapses from **0.855** (steps 35–42) to **0.028** (43+), with steps 43/45/47 scoring exactly 0.000 — a real darknet-market shutdown changed what illicit activity looks like on-chain. Key finding: **feature drift stayed flat; only target (label) drift spiked** at the exact step of collapse. A feature-only monitoring setup would have missed it entirely — a real monitoring blind spot.
- **Retraining that knows when to give up.** The replay loop promotes challengers in routine steps but *holds* at T43/T45, where a model trained on all prior data still can't beat the champion (both near zero). Conclusion: regime change ≠ staleness; more data doesn't help, so escalate to a human.
- **Reported results honestly, including a negative one.** Seven engineered graph-topology features produced only a marginal AUC-PR edge and slightly *lower* F1 than the provided features alone — likely because the dataset's 72 aggregated features already encode one-hop neighborhood info. The provided-only model won and was shipped as champion. This is reported as-is, not buried.
- **Classical beats GNNs, fairly.** GCN/GraphSAGE/EvolveGCN re-implemented from scratch in plain PyTorch, trained under the *exact same* split/features/preprocessor/metric — all lose to the XGBoost champion, consistent with the Weber et al. 2019 reference.

---

## Fact bank (verbatim metrics — do not alter)

**Model / statistical** (test window, time steps 35–49):
- Champion (XGBoost, provided 166 features, registry v2): **illicit F1 0.806, AUC-PR 0.800, precision 0.891, recall 0.735**
- Baselines: Random Forest (registry v1) F1 0.752 / AUC-PR 0.770; Logistic Regression F1 0.245
- XGBoost + 7 graph-topology features: F1 0.797, AUC-PR 0.802 (marginally better AUC-PR, worse F1 — not selected)
- Reference (Weber et al. 2019, same split): RF ≈ 0.788 illicit-F1 — our RF 0.752, champion 0.806, reported as measured, not tuned toward theirs
- Per-time-step F1: mean **0.855** (steps 35–42) → **0.028** (steps 43+); steps 43/45/47 = 0.000
- Calibration: test Brier 0.0268 (uncalibrated) → **0.0264** (sigmoid); margin small enough that the served model returns a raw, honestly-labeled probability
- Target-drift score: avg 0.052 (steps 30–42) vs **0.117** (steps 43–49); the four highest values are steps 43–46
- Feature drift (share of drifted columns): flat, 0.45–0.76 across steps 30–49, corr with time step = 0.06 — no trend
- Top SHAP features: `feat_52`, `feat_58`, `feat_89` (anonymized, so ranking shows *which* matter, not what they mean)

**Optional GNN comparison** (mean ± std, 3 seeds, plain PyTorch, CPU, 200-epoch/patience-20 budget):
- GraphSAGE 0.565 ± 0.008 | GCN 0.554 ± 0.033 | EvolveGCN 0.120 ± 0.028 — all below the 0.806 champion

**Data scale:**
- 203,769 nodes / 234,355 directed payment-flow edges / 49 time steps (~2 weeks apart)
- 46,564 labeled nodes (4,545 illicit / 42,019 licit); ~157,205 unknown, excluded from supervised train/eval
- Operative class balance: **~9.8% illicit** over labeled nodes
- 166 features: 94 local (transaction-level) + 72 aggregated (one-hop neighbor)

**System / MLOps:**
- Inference latency: **p50 0.73 ms, p95 6.48 ms** (CPU, in-process; target was < 100 ms)
- Model size: **1.1 MB** (target was < 50 MB)
- Container-verified serving: real illicit tx scores 0.9946, real licit tx scores 0.0253 (inside Linux container, no MLflow registry present)
- `dvc repro` rebuilds the full 8-stage pipeline from raw CSVs in one command
- Retraining cadence: per replayed time step, gated on a drift-or-performance flag
- CI: GitHub Actions, data-free (`pytest -m "not needs_data"` + F1 ≥ 0.6 quality gate that fails CI on regression)
- Tests: **27 passing**, including leakage guards on the temporal split and every fitted transform

---

## Tech stack (categorized for keyword matching)
- **Language:** Python 3.11+ (developed on 3.13)
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost, networkx (graph features), SHAP
- **Data/model versioning:** DVC + Google Drive remote
- **Experiment tracking + model registry:** MLflow (local backend, registry via API; champion tagged `elliptic-illicit@production`)
- **Serving:** FastAPI + Uvicorn, Docker (`/predict`, `/health`, `/metrics` endpoints), targeted at Hugging Face Spaces (free CPU)
- **Monitoring:** Evidently (feature + target drift)
- **CI/CD + retraining:** GitHub Actions (champion/challenger replay, quality gate)
- **Optional GNN:** plain PyTorch (CPU, in-repo — no PyTorch Geometric)
- **Testing:** pytest
- **Pipeline orchestration:** DVC DAG (`assemble → split_eda → train_baseline → build_graph_features → train_advanced → evaluate_champion → monitor_drift → replay_retraining`)

---

## Framing angles by role

**Data Scientist / ML** — lead with the modeling and validation rigor:
- Imbalanced classification (F1/AUC-PR not accuracy), class weighting / `scale_pos_weight`
- Honest temporal validation and leakage prevention (the strongest signal here)
- Graph-topology feature engineering (degree, PageRank, clustering, component size) — *and* the honesty of reporting it didn't win
- Probability calibration + Brier score
- Concept-drift analysis (T43), SHAP explainability
- The XGBoost-beats-GNN result under a fair same-split comparison

**ML Engineer / MLOps** — lead with the lifecycle:
- MLflow model registry + versioning (v1 baseline → v2 champion), champion/challenger promotion logic
- DVC-reproducible pipeline (`dvc repro`, one command, raw → served)
- Evidently drift monitoring with the label-vs-feature blind-spot finding
- Replay retraining loop gated on a drift-or-performance flag
- CI quality gate (F1 ≥ 0.6, fails on regression)
- Containerized serving, latency (sub-ms p50) and model-size budgets met

**Data Engineer** — lead with the pipeline and data integrity:
- 8-stage DVC DAG, deterministic one-command rebuild from raw CSVs
- Data-integrity checks (exact node/edge/label counts, time-step range, null coverage asserted in tests)
- Temporal partitioning harness (single sanctioned split function, tested for zero cross-partition leakage)
- Graph construction from a 234K-edge edgelist, per-time-step subgraphs

**Backend / Software Engineer** — lead with the service:
- FastAPI microservice: `POST /predict`, `GET /health`, `GET /metrics` (request count, p50/p95 latency)
- Dockerized, verified serving end-to-end in a clean Linux container
- Sub-millisecond p50 inference latency
- 27 automated tests, GitHub Actions CI
- Clean `src/` layout, pure/testable functions

---

## Skills demonstrated (mineable list)
Temporal validation · data-leakage prevention · imbalanced classification · gradient boosting (XGBoost) · graph feature engineering · concept-drift detection & response · probability calibration · model explainability (SHAP) · MLflow model registry · DVC data/model versioning · pipeline reproducibility (DVC DAG) · drift monitoring (Evidently) · champion/challenger retraining · CI/CD quality gates (GitHub Actions) · FastAPI service design · Docker containerization · latency/size budgeting · GNN implementation (GCN/GraphSAGE/EvolveGCN in PyTorch) · honest experimental reporting.

---

## Honest limitations (frame within these — they're also strong "I know the tradeoffs" talking points)
- **No live deployment URL yet.** Targeted Hugging Face Spaces' free CPU tier, which dropped Docker/Gradio Spaces behind a paid plan mid-build (only static Spaces stay free, and those can't run a server). Deployment postponed rather than quietly swapping hosts. The container *is* verified serving correctly.
- **Replayed, not live.** The 49 time steps are fixed historical data; "streaming" and "retraining" are an explicit replay/simulation.
- **The deployed container loads weights from a path, not the registry.** `mlflow.db` bakes absolute Windows artifact paths, so an export step ships the ~1 MB artifact into the image; the registry alias still decides *what* ships. A hosted MLflow tracking server is the real fix.
- **`PREDICT_THRESHOLD` = 0.5, untuned** — a sensible default, not a chosen precision/recall operating point.
- **The 72 aggregated features are opaque** (anonymized by the dataset publisher), so SHAP names *which* features matter, not what they mean.
- **Unknown-labeled nodes (157,205 of 203,769) are excluded** from supervised train/eval; kept only for graph structure.

---

## LaTeX resume variants

Two ready-to-paste `\resumeProjectHeading` blocks (same macros/format as the RideSync example). Update the GitHub URL if the repo name differs.

### Variant A — SDE / Software Engineer

```latex
\resumeProjectHeading
          {\href{https://github.com/mahendra-kausik/ellipticGuard}{\textbf{\large{\underline{EllipticGuard: AML Transaction Detector}}} \href{https://github.com/mahendra-kausik/EllipticGuard}{\raisebox{-0.1\height}\faExternalLink }} $|$ \large{\underline{Python, FastAPI, Docker, XGBoost, MLflow, DVC}}}{}
          \resumeItemListStart
            \resumeItem{\normalsize{Built and containerized a FastAPI microservice (\texttt{/predict}, \texttt{/health}, \texttt{/metrics}) serving illicit-transaction predictions at \textbf{0.73 ms p50 / 6.48 ms p95} latency on a 1.1 MB model; verified end-to-end in a clean Linux Docker container.}}

            \resumeItem{\normalsize{Engineered a reproducible 8-stage DVC pipeline rebuilding the full system from raw CSVs (203K nodes / 234K edges) in one command, with a MLflow model registry and champion/challenger versioning.}}

            \resumeItem{\normalsize{Set up GitHub Actions CI with 27 automated tests and an F1 quality gate that fails builds on model regression; added leakage guards on the temporal data split and every fitted transform.}}
          \resumeItemListEnd 
          \vspace{-13pt}
```

### Variant B — ML / Data Science

```latex
\resumeProjectHeading
          {\href{https://github.com/mahendra-kausik/ellipticGuard}{\textbf{\large{\underline{EllipticGuard: AML Transaction Detector}}} \href{https://github.com/mahendra-kausik/EllipticGuard}{\raisebox{-0.1\height}\faExternalLink }} $|$ \large{\underline{Python, XGBoost, scikit-learn, PyTorch, Evidently, MLflow}}}{}
          \resumeItemListStart
            \resumeItem{\normalsize{Trained an XGBoost illicit-Bitcoin-transaction classifier reaching \textbf{0.806 illicit-class F1} (0.800 AUC-PR) on a strict temporal split of the Elliptic graph (\textasciitilde9.8\% positive), beating RF, LR, and from-scratch GCN/GraphSAGE/EvolveGCN baselines.}}

            \resumeItem{\normalsize{Diagnosed a concept-drift event (F1 collapse 0.855\,$\rightarrow$\,0.028) detectable only via \textbf{target drift}, not feature drift; built Evidently monitoring and a champion/challenger retraining loop that correctly holds rather than promotes on regime change.}}

            \resumeItem{\normalsize{Prevented data leakage with a causal temporal split (train/val/test by time step) and train-only transform fitting; added probability calibration (Brier 0.0268\,$\rightarrow$\,0.0264) and SHAP feature attribution.}}
          \resumeItemListEnd 
          \vspace{-13pt}
```
