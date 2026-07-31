# EllipticGuard — Interview Flashcards

Cover the answer. Say it out loud. If you stall, the `(D-0XX)` tag tells you where to read.

Every number here is a measured value from a real run. Don't round them up in an interview.

---

## A. The 60-second pitch

**Q1. What is EllipticGuard?**
An AML illicit-transaction detector on the Elliptic Bitcoin dataset — 203,769 transactions, 234,355 edges, 49 ordered time steps. Built as a deployable system, not a notebook: DVC pipeline, MLflow registry, FastAPI container on Cloud Run, drift monitoring, replayed retraining loop. Headline: illicit-class **F1 0.806, AUC-PR 0.800** on a strictly-future test window.

**Q2. What's the one thing you want me to remember?**
The model looks good in aggregate (F1 0.806) and collapses to **F1 0.028 at time step 43** — a real darknet-market shutdown. Feature-drift monitoring would have missed it entirely; only target drift caught it. That's the project: honest validation surfacing a failure a naive setup would hide.

**Q3. Why this dataset?**
It's one of the few public financial-crime graphs with real labels and a real temporal axis, and it has a documented regime change baked in. That makes it possible to demonstrate the leakage and drift problems honestly instead of simulating them.

---

## B. Validation & leakage — the core discipline

**Q4. How do you split the data, and why? (D-001)**
Temporal, never random. Train 1–29 (what gets fit), val 30–34 (tuning/thresholds), test 35–49. Validation is carved from the **tail of the training range**, never the future. Fraud patterns evolve — a random split leaks future information and inflates illicit-F1 from roughly 0.45 to 0.80 in published/blogged runs. Temporal mirrors deployment: you only ever have the past.

**Q5. `split.train` is 1–29, but the plan says train is 1–34. Why the mismatch? (D-019)**
Deliberate naming. `train` always means "what got fit on," so a careless `scaler.fit(split.train)` can't silently touch the val range. If a layer needs 1–34 it concatenates `train + val` explicitly at the call site. Boundaries live in `params.yaml`, so the exact partition is DVC-versioned, not buried in code.

**Q6. But the final champion IS fit on 1–34. Isn't that leakage? (D-022)**
No. Val is inside the training range, not the future. Hyperparameters were chosen on val illicit-F1; once chosen, refitting on 1–34 uses more real non-leaking data before the single touch of test. Matches Weber et al.'s convention.

**Q7. Why not sklearn cross-validation for tuning? (D-022)**
A random CV fold violates the temporal order — it would train on step 40 to predict step 20. A single held-out val slice already respects time. Small manual grid (`max_depth`, `learning_rate`, `n_estimators`, 6 combos) scored on val.

**Q8. Where could leakage have snuck in, and did it? (D-026)**
It did, once. The first retraining replay trained each challenger on data **up to and including** the step it was scored on — in-sample. Result: F1 = 1.000 at every step, T43 included, silently erasing the whole story. Fixed to train strictly on `time_step < step`. Caught before any gate evidence was recorded. Same leakage discipline, just relocated into a new module.

**Q9. How do you guarantee graph features don't leak? (D-006)**
Structurally, not by convention. Edges in this dataset exist only *within* a time step — each step is its own component. All topology is computed per-time-step subgraph, never aggregated across steps, so every neighbor is contemporaneous by construction. A test asserts zero cross-time-step edges.

**Q10. Why don't you use a neighbor's label as a feature? (D-021)**
Direct leakage. At inference you don't know a neighbor's true label — that's exactly what you're predicting. Hand-labeled neighbors would propagate unrealistically. All 7 topology features are label-free.

---

## C. Data & labels

**Q11. What's the class imbalance? Careful — this is a trap. (D-002)**
**~9.8% illicit**, over the 46,564 *labeled* nodes (4,545 illicit / 42,019 licit). Not "2%" — that figure divides by all 203,769 nodes including ~157,205 unknowns, and understates the base rate the classifier actually faces.

**Q12. What do you do with the 157,205 unknown nodes? (D-016)**
Kept in the assembled table with `label = NaN`, excluded from supervised training and metrics by filtering `label.notna()`. They're needed for graph structure — dropping them early removes ~85% of edges, since most edges touch an unknown node. The GNN layer does the same thing via a `label_mask`: unknowns participate in message passing, not in loss.

**Q13. 165 or 166 features? (D-015)**
The CSV has 167 raw columns, no header: `txId` + `time_step` + 165 features. The paper's "166" counts `time_step` as the first local feature. We split `time_step` out as its own named column so downstream code filters by name, not by positional index.

**Q14. Why load raw CSVs instead of PyTorch Geometric's built-in loader? (D-004)**
Transparency. PyG's loader hides the label remap, the time-step column, and the 165-vs-166 question. For a project I need to defend end to end, I'd rather own the parsing and assert the shapes at runtime.

---

## D. Metrics

**Q15. Why isn't accuracy your headline metric? (D-003)**
~90% of labeled nodes are licit, so "predict everything licit" scores ~90% accuracy and catches zero fraud. Headline is illicit-class precision/recall/F1 plus AUC-PR.

**Q16. Why AUC-PR and not ROC-AUC?**
ROC-AUC is over-optimistic under heavy imbalance — a large true-negative pool flatters the false-positive rate. AUC-PR focuses on the positive class, which is the one that matters.

**Q17. What are the actual numbers?**

| Model | Illicit F1 | AUC-PR | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.245 | 0.204 | 0.140 | 0.947 |
| Random Forest (registry v1) | 0.752 | 0.770 | 0.849 | 0.675 |
| **XGBoost, 166 provided (v2, champion)** | **0.806** | **0.800** | 0.891 | 0.735 |
| XGBoost + 7 graph features | 0.797 | 0.802 | 0.871 | 0.735 |
| GraphSAGE | 0.565 ± 0.008 | 0.531 | — | — |
| GCN | 0.554 ± 0.033 | 0.558 | — | — |
| EvolveGCN | 0.120 ± 0.028 | 0.103 | — | — |

**Q18. Logistic regression has 0.947 recall — that's the best recall in the table. Why isn't it the champion?**
Precision 0.140. It flags almost everything; nine of every ten alerts are false. Useless for analyst triage. That's the whole reason F1 and AUC-PR are the headline, not recall.

**Q19. How do you compare to the literature?**
Weber et al. 2019 (arXiv:1908.02591) report RF ≈ 0.788 illicit-F1 on the same temporal split. Our RF measured 0.752, our XGBoost 0.806 — reported as measured, not tuned toward their number.

---

## E. Modeling

**Q20. Why XGBoost? (D-007)**
Strong on heterogeneous tabular features, handles non-linear interactions, trains fast on CPU, exposes importances for explanation. RF is the clean known baseline. Logistic regression is too weak alone; SVM is too slow at this size; GNNs were tested and lost.

**Q21. How do you handle imbalance? Why not SMOTE? (D-008)**
`scale_pos_weight = neg/pos` — class weighting, not resampling. SMOTE fabricates synthetic nodes in a graph-derived feature space where interpolated points may be structurally impossible, and it interacts badly with a temporal setting. Undersampling throws away signal. Weighting creates no new signal either, but it's leakage-free and simple.

**Q22. You built 7 graph-topology features. Did they help? (D-021, D-022)**
**No — and that's the honest answer.** in/out-degree, unique neighbors, PageRank, clustering coefficient, component size, avg neighbor degree. Result: F1 0.797 vs 0.806, AUC-PR marginally *higher* at 0.802. The champion was selected on test F1, so the provided-only model won and got registered as v2. Likely cause: the dataset's own 72 "aggregated" features already encode one-hop neighborhood info, so topology position was largely redundant for an 8-deep, 400-tree model.

**Q23. Why keep the feature work in the repo if it lost?**
Because the reasoning was sound and the result is data. Deleting a negative result to make the story tidy is exactly the failure mode this project is built to avoid. It also answers "did you check?" with a measurement instead of an opinion.

**Q24. What about calibration? (D-009, D-023)**
Tree ensembles are typically miscalibrated, and analysts triage by score, so probability quality matters operationally. Tested on test Brier: uncalibrated 0.0268 → sigmoid/Platt **0.0264** → isotonic 0.0266. The calibrator was fit on val only, wrapping a `FrozenEstimator` base model refit on train-only (1–29) so val was genuinely untouched.

**Q25. So does the API return a calibrated probability? (D-024)**
No — it returns v2's raw `predict_proba`. The margin was tiny (0.0268 → 0.0264) and the calibrated model has a *different, weaker base* (fit on 1–29, not 1–34). Serving that model's output as if it were v2's calibrated score would misrepresent it, and v2 itself can't be calibrated without touching test twice. Raw and honestly labeled beats a fabricated "calibrated" field.

---

## F. The T43 drift story — your strongest card

**Q26. Walk me through what happened at time step 43. (D-012, D-023)**
Per-time-step illicit-F1 averages **0.855** for steps 35–42, then **0.028** from step 43 on — steps 43, 45, and 47 score exactly 0.000. It corresponds to a real darknet-market shutdown that changed what illicit activity looks like on-chain. A single aggregate 0.806 hides this completely.

**Q27. Which drift signal caught it? (D-025)**
**Target drift, not feature drift.** The `label` column's drift score averages 0.052 for steps 30–42 vs **0.117** for 43–49; the four highest values in the entire series are steps 43–46. Meanwhile feature drift is *flat* — share-of-drifted-columns fluctuates 0.45–0.76 with no trend (correlation with time step 0.06; 0.583 pre-T43 vs 0.605 post-T43).

**Q28. Why does that gap matter?**
It's the finding. The covariates barely moved; the labels moved. That's **concept drift** by definition — `P(X)` stable, `P(y|X)` changed. A production system monitoring only label-free feature drift would have reported healthy right through a total detection failure. That's a real, named monitoring blind spot.

**Q29. You initially got a flat, boring drift curve. What did you fix? (D-025)**
Two things. `time_step` was being drift-tested as a feature — a later window's time_step differs by construction, not distribution, inflating every window's count by one. And target drift was folded into the same aggregate, burying the T43 signal inside 165 mostly-flat columns. Split into two independent Evidently reports; the real story appeared.

**Q30. Did you consider forcing a cleaner "drift increases over time" curve?**
That'd be gate-gaming — cherry-picking a stattest or threshold until the plot looked like the expected story. The honest target-drift result satisfies the gate's intent and is a more interesting finding than a blended number that trends nicely.

**Q31. Why did you *reject* NannyML CBPE? (D-033)**
CBPE estimates live performance from predicted probabilities under an explicit **no-concept-drift assumption** — it's built for covariate shift, where `P(X)` moves but `P(y|X)` holds. Layer 9 measured the exact opposite. CBPE would have seen stable inputs, stable probability distributions, and reported healthy performance while actual F1 fell 0.855 → 0.028. Adding a monitor whose one documented blind spot coincides with your headline finding is worse than not adding it. If a future variant faced genuine covariate shift, CBPE would be the right tool.

**Q32. What's the gap that creates?**
No label-free performance estimation at all — real for any deployment where labels arrive late, and AML labels genuinely do. Scoped and stated: this project's replay explicitly assumes per-step labels are available.

---

## G. Retraining

**Q33. Is your retraining loop live? (D-012)**
No, and I say so up front. There's no live Bitcoin feed; the 49 steps are fixed and historical. It's an explicit **replay** — steps fed in sequence to simulate streaming. Claiming a live pipeline collapses under one question.

**Q34. What does the loop actually show? (D-026)**
Both cases. Routine steps: the challenger wins and is **promoted** (step 39 — challenger 0.968 vs champion 0.884). T43 and step 45: the challenger, trained on everything before that step, still cannot beat the champion — both near zero — so the loop **holds**. Later challengers recover only partially (0.19–0.90) and never reach the ~0.85 pre-collapse level.

**Q35. Why is "hold" the correct behavior?**
T43 is a regime change, not staleness. More training data doesn't help when the thing you're detecting has changed shape — there's no post-shutdown pattern in steps 1–42 for a retrained model to have learned. The correct action is escalation to a human analyst. A loop that promotes anything merely new is worse than one that stops.

**Q36. Your first champion for the replay was different. Why did you change it? (D-026)**
The draft trained a deliberately stale champion on steps 1–20 to force a promotion. It scored near zero on *every* post-T43 step regardless of the regime break, conflating "champion is stale" with "T43 is unrecoverable." Replaced with the real Layer 6 train-only champion (0.759 test F1), so the promote/hold verdicts tie to documented evidence instead of an invented baseline.

---

## H. GNNs (optional Layer 8)

**Q37. It's a graph problem — why isn't your main model a GNN? (D-030, D-031)**
It is one of the models; it just lost. GCN 0.554, GraphSAGE 0.565, EvolveGCN 0.120 vs XGBoost 0.806 — all under identical split, features, preprocessor, class weight, optimizer, epoch budget, patience, and the same `evaluate()` function, 3 seeds each. Consistent with Weber et al.'s own table, where RF beat every GNN they tried. I shipped the model that actually won.

**Q38. How do you know your GNN implementation isn't just buggy? (D-031)**
Two checks. `tests/test_gnn.py` hand-verifies adjacency normalization, causal snapshotting, and masked loss against manually computed values. And a diagnostic at the notebook's own budget reached **val F1 0.766 but test F1 0.509** — a model that fits validation well but not test is a generalization gap, not broken tensor math. All three GNNs also collapse at T43, independently corroborating the drift story.

**Q39. Why plain PyTorch and not PyTorch Geometric? (D-030)**
The three architectures need two primitives: a normalized sparse adjacency matmul `D^-0.5 (A+I) D^-0.5`, and mean neighbor aggregation `D^-1 A`. ~40 lines. PyG is a heavy, wheel-fragile dependency for code I can read end to end — and code I can read is code I can defend. torch lives in a separate `requirements-gnn.txt` so it never enters the serving image.

**Q40. You raised the epoch budget after the gate. Isn't that tuning to a number? (D-032)**
It's a like-for-like re-run: 60/8 → 200/20 applied identically to all three models, every other knob untouched. GCN 0.450→0.554, GraphSAGE 0.496→0.565, EvolveGCN 0.120→0.120 *exactly* — reproducing to the same value confirms it was early-stopping well before epoch 60 either way, so its ceiling is architectural (truncated BPTT window = 1), not budget. Both runs are recorded; neither overwrote the other. Conclusion unchanged: all three still lose.

**Q41. You audited existing notebooks and rejected their comparison. Why? (D-030)**
Four reasons. (1) The headline "EvolveGCN is drift-resilient, 73.2% less F1 drop" is a **floor effect** — it started at 0.198 vs Static GCN's 0.710. You can't fall far from the floor. (2) The models were never comparable: class weights [1,9] vs [1,5], epochs 200 vs 500, so any delta confounds architecture with tuning. (3) A real bug — embeddings exported to the SHAP notebook were computed with a stale `W_state`, no warm-up, so those artifacts don't match the metrics reported in the same notebook. (4) Different split (val 35–36 / test 37–49), so the numbers aren't comparable to ours anyway.

---

## I. MLOps & serving

**Q42. Why MLflow's registry rather than just saving pickles? (D-010)**
Lineage, rollback, and auditability — "which model version made this prediction?" is a genuine AML compliance question. Runs capture the messy experiment search; the registry holds the small set that's ever servable. Only meaningful versions registered: v1 RF, v2 XGBoost champion, v3+ retrained.

**Q43. Why sqlite for MLflow and not the file store? (D-020)**
MLflow 3.x refuses the plain `./mlruns` file store — it's maintenance-mode upstream and registry support is being phased out. `sqlite:///mlflow.db` is what MLflow's own error message steers you to, still fully local and free.

**Q44. Does the deployed container load from the registry? (D-028)**
**No, and this is a real limitation I state plainly.** Locally it resolves `models:/elliptic-illicit@production` — the registry is the source of truth. But the local sqlite DB bakes absolute Windows paths into `model_versions.source`, and `mlruns/` is git-ignored so the build context never receives it. `export_model.py` resolves the alias at build time and ships the ~1 MB artifact; the container reads `/app/model` via `MODEL_URI`. The alias still decides *what* ships, so promotion stays the single control point. The real fix is a hosted tracking server — no free tier offers one.

**Q45. How is promotion controlled? (D-024)**
`promote_model.py` sets an MLflow **alias** (`@production`) on the `serving_candidate` version. The app loads by alias, never a run ID or hard path. Aliases replace MLflow's deprecated Staging/Production stages.

**Q46. You cut the image 919 MB → 300 MB. How? (D-034)**
Three moves. (1) A separate `requirements-serve.txt` dropping dvc/mlflow/shap/matplotlib/seaborn/evidently/pytest — training and CI deps the serving path never imports. (2) `app.py` no longer imports mlflow at module level; it's behind a `_load_from_registry()` helper only reached for registry URIs, and `MODEL_FEATURE_COLS` moved out of `baseline.py` (which pulls sklearn) into pandas-only `loaders.py`. (3) `xgboost` installed `--no-deps` — its PyPI wheel hard-requires `nvidia-nccl-cu12`, a **~400 MB GPU-only library**, for CPU-only use. That was the single largest thing in the image and I found it by measuring (`du -sh site-packages/*`), not by estimating.

**Q47. How did you verify the slimming didn't break anything?**
Built both images, ran both containers side by side, posted the same real illicit and licit fixture rows to each: `0.9945693016052246` and `0.025330474600195885` — **bit-for-bit identical**. Plus a new data-free test asserting `XGBClassifier().load_model()` and `mlflow.xgboost.load_model()` agree. 35/35 tests pass.

**Q48. Also added a `.dockerignore` — why does that matter? (D-034)**
There wasn't one. `gcloud run deploy --source .` would have uploaded `.env` and `mlruns/` to Cloud Build as build context — the Dockerfile never `COPY`d them into the image, but they'd still have left the machine. A secrets-hygiene fix, not a size fix.

**Q49. Why Cloud Run, and what's the catch? (D-029, D-034)**
Original plan was HF Spaces free Docker. Mid-build HF moved Docker and Gradio Spaces behind a paid plan — only static Spaces stay free, and a static Space has no Python process, so `/predict` can't exist. Rejected Render (512 MB RAM), Fly.io (no free tier), Koyeb (absorbed into Mistral). Cloud Run runs on existing GCP **trial credit**, so it's a deliberately time-limited demo expected to lapse around the 3-month mark — a stated deviation from the free-tier-only rule, not a quiet one. `docker run -p 7860:7860 ellipticguard` is the permanent path.

**Q50. There's a lesson in that HF episode. What is it? (D-029)**
I had asserted Docker Spaces were free based on HF's own documentation, which still published a "CPU Basic — FREE" hardware table. The signup page said otherwise. **Vendor docs are not verification when the product's own UI contradicts them** — prefer the live flow as ground truth for pricing and quota claims.

**Q51. What does CI actually run? (D-026, D-027)**
`pytest -m "not needs_data"` — 27 of 35 tests — plus a quality gate reading a git-committed `metrics/quality_gate.json` that fails the build if champion F1 drops below 0.6. Data-dependent tests are excluded because `data/`, `mlruns/`, and `mlflow.db` are git-ignored and DVC-pulling them needs a Google Drive OAuth secret I chose not to put in Actions. Stated as a documented gap, not a silent one.

**Q52. What broke CI on first push, and how did you fix it? (D-027)**
All local runs used `python -m pytest`, which auto-prepends CWD to `sys.path`. CI calls bare `pytest`, which doesn't — 9 collection errors. Fixed with `pythonpath = .` in `pytest.ini`, which fixes every invocation method at once, rather than changing the CI command and leaving the same footgun for the next person. Verified by reproducing it in a fresh clone first.

**Q53. Performance numbers?**
Inference p50 **0.73 ms**, p95 **6.48 ms** CPU in-process (target was <100 ms). Model 1.1 MB (target <50 MB). `/metrics` uses a stdlib `deque(maxlen=1000)` with a nearest-rank percentile helper — no `prometheus_client`, since a single-worker demo doesn't need a shared metrics store. It resets on restart and is per-process; that's the upgrade path if it ever runs multi-worker.

---

## J. The hard questions — know these cold

**Q54. What's the weakest part of this project?**
The deployed container loads weights from a path, not the registry (D-028). Locally the registry is the source of truth; in the container it isn't. A hosted MLflow tracking server removes it entirely, and no free tier offers one.

**Q55. What would you do differently with more time or budget?**
Hosted MLflow tracking server (kills the path-vs-registry gap). Tune `PREDICT_THRESHOLD` — it's 0.5, a sensible default, not an operating point chosen from a precision/recall tradeoff against a real analyst review capacity. And a proper per-model GNN hyperparameter search, though that reintroduces the unequal-knobs confound I deliberately avoided.

**Q56. What's the biggest limitation of the *result*?**
The 72 aggregated features are anonymized by the dataset publisher. SHAP ranks `feat_52`, `feat_58`, `feat_89` highest — I can say *which* features matter, but not what they mean. For a real AML system that's a serious explainability gap, since analysts need a reason, not a rank.

**Q57. How much of this is real production capability?**
Honest boundary: the model, the temporal validation, the drift measurement, and the container are real and verified. The retraining loop is a **replay** over historical data. The deployment is a single Cloud Run instance on trial credit. It's a working system, not production at scale, and every one of those lines is written into the README's Limitations section.

**Q58. Did you use AI to build this?**
Yes — and the reason I can answer every question above is that every non-trivial choice was logged with its alternatives and tradeoffs at the moment it was made. 34 entries in `DECISIONS.md`. Ask me about any of them.
