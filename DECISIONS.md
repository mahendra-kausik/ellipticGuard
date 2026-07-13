# DECISIONS.md — EllipticGuard

Every non-trivial choice, written so the owner can defend it in an interview. **Trivial choices (variable names, obvious formatting) are not logged.** When unsure, log it. Never edit a past entry to reflect a reversal — add a new entry and fill in `Supersedes:`.

Entries D-001–D-012 are pre-seeded from planning. Claude Code appends D-013+ during the build (including any bugs fixed in the owner's notebooks), during the post-layer doc update.

---

### Entry template (copy this)

```
## D-XXX — <short title>
- **Date / Layer:** <when>
- **Context:** <what problem/choice prompted this>
- **Decision:** <what we chose>
- **Why:** <the core reason(s)>
- **Alternatives Considered:** <options + why not>
- **Tradeoffs / risks:** <what we give up; what could go wrong; how we'd mitigate>
- **Supersedes:** <D-YYY if applicable>
```

---

## D-001 — Temporal split, not random
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 2
- **Context:** Fraud patterns evolve; the dataset has 49 ordered time steps.
- **Decision:** Train on steps 1–34, test on 35–49, with a validation slice from the tail of the training range (steps 30–34). No random split anywhere.
- **Why:** A random split leaks future information into training and massively inflates illicit-class scores (published/blogged runs jump from ~0.45 to ~0.80 illicit-F1 under a random split). A temporal split mirrors real deployment: you only ever have the past to predict the future.
- **Alternatives Considered:** Random k-fold or stratified split — rejected as leaky and unrealistic for a temporal fraud problem.
- **Tradeoffs / risks:** Lower, less flattering numbers. Risk: someone reads the honest number as "worse." Mitigation: report the random-vs-temporal gap explicitly — it's a strength, not a weakness.
- **Supersedes:** —

## D-002 — Evaluate on labeled nodes only; state the imbalance honestly
- **Date / Layer:** Planning (pre-seed) — enforced at Layers 1–3
- **Context:** 203,769 nodes but only 46,564 labeled (4,545 illicit / 42,019 licit); the rest are `unknown`.
- **Decision:** Train/evaluate supervised models on labeled nodes only. Report imbalance as **~9.8% illicit of labeled nodes**, not "2% of all nodes."
- **Why:** Supervised metrics are only meaningful where labels exist. The 2%-of-all figure understates the real base rate the classifier faces; ~9.8% is the operative number.
- **Alternatives Considered:** Treat `unknown` as negative (wrong — many unknowns are likely illicit); full semi-supervised approach (heavier, out of scope for the core model).
- **Tradeoffs / risks:** We drop label-free rows for supervised metrics. Mitigation: keep `unknown` nodes available for graph structure/topology so they still inform features.
- **Supersedes:** —

## D-003 — Primary metric = illicit-class F1 + AUC-PR, never accuracy
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 3
- **Context:** ~90% of labeled nodes are licit.
- **Decision:** Headline metrics are illicit-class precision/recall/F1 and AUC-PR. Accuracy is not a headline.
- **Why:** Predicting "all licit" yields ~90% accuracy while catching zero fraud. AUC-PR focuses on the positive (illicit) class under imbalance; F1 balances precision/recall at the operating point.
- **Alternatives Considered:** Accuracy (misleading here); ROC-AUC (over-optimistic under heavy imbalance).
- **Tradeoffs / risks:** Less intuitive to a lay reader than "96% accuracy." Mitigation: pair the metric with a one-line explanation of why accuracy is the wrong tool here.
- **Supersedes:** —

## D-004 — Load raw CSVs directly rather than a prepackaged loader
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 1
- **Context:** PyTorch Geometric ships `EllipticBitcoinDataset`, which pre-processes the data.
- **Decision:** Load raw CSVs ourselves for the classical pipeline; confirm headers and feature count at runtime. (PyG still allowed, optionally, for the GNN layer.)
- **Why:** For a resume project the owner must understand the data end to end. Raw loading makes the label remap, the time-step column, and the 165-vs-166 feature question explicit rather than hidden.
- **Alternatives Considered:** Use PyG's loader everywhere — rejected for transparency reasons in the core pipeline.
- **Tradeoffs / risks:** A little more code, and a chance of a header/parsing mistake. Mitigation: runtime assertions on shapes, counts, and header presence in Layer 1.
- **Supersedes:** —

## D-005 — Graph-topology features on top of the 72 aggregated features
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 4
- **Context:** The dataset already includes 72 one-hop "aggregated" features.
- **Decision:** Engineer additional topology features: in/out-degree, unique neighbors, PageRank, local clustering coefficient, connected-component size — computed within each time-step subgraph.
- **Why:** The 72 aggregated features summarize neighbors' *transaction attributes*, not the *graph structure* (centrality, connectivity). Topology captures a complementary signal and is the reason a classical model can approach GNN performance.
- **Alternatives Considered:** Provided features only (leaves signal on the table); jump straight to a GNN (heavier, less defensible as feature engineering).
- **Tradeoffs / risks:** Extra computation. Mitigation: cheap on this graph size; cache the engineered table via DVC.
- **Supersedes:** —

## D-006 — Causal features by construction (no cross-time aggregation)
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 4
- **Context:** Edges in this dataset exist only within a time step; each step is its own connected component.
- **Decision:** Compute all graph features per-time-step subgraph; never aggregate across steps.
- **Why:** Same-step-only neighbors are always contemporaneous, so per-step topology features cannot leak future information — a structural guarantee, not just a convention.
- **Alternatives Considered:** One global graph across all steps — structurally meaningless here and would risk mixing time.
- **Tradeoffs / risks:** None material. Risk: a future maintainer builds a global graph by mistake. Mitigation: a Layer-2 test asserts zero cross-time-step edges.
- **Supersedes:** —

## D-007 — XGBoost as the champion classical model
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 5
- **Context:** Need a strong, CPU-friendly model on tabular + graph features under imbalance.
- **Decision:** Random Forest as baseline; XGBoost (with `scale_pos_weight`/class weighting) as the tuned champion.
- **Why:** Gradient-boosted trees are strong on heterogeneous tabular features, handle non-linearities/interactions, train fast on CPU, and expose importances for explanation. RF gives a clean, known baseline.
- **Alternatives Considered:** Logistic Regression (too weak alone); SVM (slow at this size); GNN (heavier — kept optional).
- **Tradeoffs / risks:** Trees don't use graph structure natively. Mitigation: that's exactly why we feed them engineered topology features (D-005).
- **Supersedes:** —

## D-008 — Handle imbalance via class weighting, not resampling
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 5
- **Context:** ~9.8% positive class.
- **Decision:** Use class weights / `scale_pos_weight`; avoid SMOTE-style oversampling.
- **Why:** Synthetic oversampling on graph-derived features risks fabricating implausible nodes and interacts badly with a temporal setting. Class weighting is simpler and leakage-free.
- **Alternatives Considered:** SMOTE/ADASYN (fabrication + leakage risk); undersampling (throws away signal).
- **Tradeoffs / risks:** Weighting creates no new signal. Mitigation: acceptable — we rely on features, not synthetic data; tune the weight on the val slice.
- **Supersedes:** —

## D-009 — Probability calibration + Brier score
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 6
- **Context:** An AML score is only useful if the probability is trustworthy for triage/thresholding.
- **Decision:** Calibrate (Platt or isotonic, fit on train/val only) and report Brier score.
- **Why:** Tree ensembles are often miscalibrated. Analysts triage by score, so calibrated probabilities matter operationally — and it's a maturity signal most freshers skip.
- **Alternatives Considered:** Report raw uncalibrated scores — cheaper but less trustworthy.
- **Tradeoffs / risks:** A small extra step with a leakage trap. Mitigation: fit calibration strictly on train/val, never test.
- **Supersedes:** —

## D-010 — MLflow for tracking + a small, meaningful model registry
- **Date / Layer:** Planning (pre-seed) — enforced at Layers 3 and 10
- **Context:** Many experiments will run; only a few models should ever be "servable."
- **Decision:** Log all runs to MLflow; register only meaningful versions — v1 (RF baseline), v2 (graph-feature champion), v3+ (retrained). Serve whatever is tagged Production.
- **Why:** Separates the messy experiment search (runs) from the small set of promotable models (registry). Enables champion/challenger promotion, one-command rollback, and "which model made this prediction?" auditability — a real AML concern.
- **Alternatives Considered:** Save pickles by filename — no lineage, no rollback, no audit trail.
- **Tradeoffs / risks:** The registry is fullest-featured against a backed tracking server; on a local file store some stage features are limited. Mitigation: use the registry API locally and note the hosted-server upgrade path. Risk: registering every run bloats the registry — mitigation: register only v1/v2/retrained.
- **Supersedes:** —

## D-011 — DVC + Google Drive for reproducibility on free tier
- **Date / Layer:** Planning (pre-seed) — enforced at Layers 0–1
- **Context:** Need reproducible data/feature/model artifacts without paid storage.
- **Decision:** DVC with a Google Drive remote; a `dvc.yaml` DAG (assemble → features → train → evaluate) reproducible via `dvc repro`.
- **Why:** DVC is the most resume-recognized data-versioning tool, sits on top of git, and is free with Drive. `dvc repro` makes the "reproducible pipeline" claim literally true.
- **Alternatives Considered:** Commit data to git (bad practice, bloat); lakeFS/S3 (paid/heavier).
- **Tradeoffs / risks:** Drive has a ~15 GB cap and API rate limits. Mitigation: keep artifact count modest; re-verify limits before relying on them.
- **Supersedes:** —

## D-012 — Retraining is an explicit REPLAY, and T43 is framed honestly
- **Date / Layer:** Planning (pre-seed) — enforced at Layer 10
- **Context:** No live Bitcoin feed exists; the 49 steps are fixed. A darknet market shut down at ~step 43, collapsing illicit recall thereafter.
- **Decision:** Simulate streaming by replaying steps in order. Show two cases: (a) routine drift where retraining on the expanded window helps and a challenger is promoted; (b) the T43 regime change where retraining does **not** recover, so the flag's correct action is to alert/escalate and hold, not silently "fix."
- **Why:** Claiming a live pipeline we don't have collapses under one question. The honest framing — monitoring *detects/surfaces* a regime change that retraining can't cure — is more senior than a fairy-tale recovery. (Weber et al. found even per-step retraining failed to recover post-shutdown; reproduce on our own split before asserting.)
- **Alternatives Considered:** Present T43 as "detected → retrained → recovered" (false here); ignore T43 (wastes the best story).
- **Tradeoffs / risks:** Less tidy narrative. Risk: our replay doesn't reproduce the exact collapse. Mitigation: verify empirically first and report whatever the real curve shows.
- **Supersedes:** —

---
<!-- D-013+ appended by Claude Code during the post-layer doc update. Include notebook-bug fixes here. -->

## D-013 — DVC Google Drive remote configured with `--local`, ID never committed
- **Date / Layer:** Layer 0
- **Context:** `dvc remote add` writes to the tracked `.dvc/config` by default, which would put `GDRIVE_FOLDER_ID` (from `.env`) into git history.
- **Decision:** Ran `dvc remote add -d --local storage gdrive://$GDRIVE_FOLDER_ID`, sourcing the ID from `.env` at the shell rather than typing it into any tracked file. This writes to `.dvc/config.local`, which is git-ignored; the tracked `.dvc/config` stays empty of the ID.
- **Why:** Satisfies the "never hardcode the folder ID in a tracked file" rule while still giving every layer a working default remote locally.
- **Alternatives Considered:** Put the ID directly in `.dvc/config` and rely on `.gitignore` for the whole `.dvc/` dir — rejected because it's easy to accidentally force-add `.dvc/config` later; keeping the ID only in the local, always-ignored file is safer by construction.
- **Tradeoffs / risks:** Each clone (including CI) must re-run this remote-add step from its own `.env`; documented so it isn't a surprise later.
- **Supersedes:** —

## D-014 — venv named `.venv`, Python 3.13.3 used despite "3.11+" pin note
- **Date / Layer:** Layer 0
- **Context:** `requirements.txt` says "Python 3.11+ required"; the only Python available locally is 3.13.3.
- **Decision:** Used `.venv` (not `venv`) as the environment directory name, and installed `requirements.txt` unmodified against Python 3.13.3 — full install succeeded with no pin conflicts.
- **Why:** `.venv` is the more common convention and sorts out of the way in file listings; no compatibility issue actually surfaced on 3.13, so no pins needed tightening.
- **Alternatives Considered:** Pin to a specific 3.11 interpreter via pyenv/similar — unnecessary since 3.13 installed cleanly.
- **Tradeoffs / risks:** If a Layer-1+ dependency later breaks on 3.13, revisit and pin more tightly then.
- **Supersedes:** —

## D-015 — 165-vs-166 feature count reconciled: `feat_0..feat_164` + separate `time_step` column
- **Date / Layer:** Layer 1
- **Context:** PROJECT_PLAN.md flagged 165 vs 166 as an item to confirm, not assume, at load time.
- **Decision:** Confirmed at runtime: `elliptic_txs_features.csv` has 167 raw columns (no header) = `txId` + `time_step` + 165 feature columns. We keep `time_step` as its own named column and name the remaining 165 as `feat_0..feat_164`. The paper's "166 features" figure counts `time_step` itself as the first local feature; our column layout is the same data, just with `time_step` split out and named instead of left as `feat_0`.
- **Why:** Loaders in Layer 2+ (temporal split, per-step graphs) need to filter/group by time step constantly — leaving it unnamed inside a generic `feat_i` block would make every downstream call site index into the array positionally instead of by name, which is more error-prone to get right and to review.
- **Alternatives Considered:** Match the owner's preprocessing notebook exactly, which also separates `time_step` from `feat_0..feat_164` — no divergence found; naming here already agrees with the notebook.
- **Tradeoffs / risks:** None — this is a naming/shape reconciliation, not a modeling choice. Any code that expects "166 raw feature columns" must know one of those 166 is `time_step`.
- **Supersedes:** —

## D-016 — Unknown-labeled nodes kept in the assembled node table, not dropped
- **Date / Layer:** Layer 1
- **Context:** ~157,205 of 203,769 nodes have `class == "unknown"`. Layer 1 only assembles the node table; it does not yet decide who trains on what.
- **Decision:** `assemble_node_table()` joins all 203,769 rows and maps `label` to `1` (illicit), `0` (licit), or `NaN` (unknown) — no rows are dropped at this stage.
- **Why:** Per D-002, unknown nodes must stay available for graph-structure use even though they're excluded from supervised train/eval. Dropping them here would make that impossible later and would also shrink the edge list's connectivity (edges touching an unknown node would look "broken"). Supervised layers (3, 5) are responsible for filtering on `label.notna()` when they build `X`/`y`.
- **Alternatives Considered:** Drop unknown rows immediately in Layer 1 — rejected; matches a mistake flagged in the owner's preprocessing notebook, where dropping unknowns early was found to remove ~85% of edges.
- **Tradeoffs / risks:** Every consumer of the assembled node table must remember to filter `label.notna()` for supervised work. Mitigation: documented here and will be enforced by `make_temporal_split()` in Layer 2.
- **Supersedes:** —

## D-017 — `pyOpenSSL`/`cryptography` pinned to `24.2.1`/`43.0.3` in `.venv`
- **Date / Layer:** Post-Layer-1 (environment fix, not tied to a specific layer)
- **Context:** Attempted `dvc push` to exercise the Google Drive remote; it failed immediately with `module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'` — a known incompatibility between `pydrive2` (via DVC's gdrive remote) and newer `cryptography` releases (breaks starting ~v42+) that ships with a `pyOpenSSL` too old to match.
- **Decision:** Pinned `pyopenssl==24.2.1` and `cryptography==43.0.3` — the newest pair that satisfies both `pydrive2`'s ceiling (`pyOpenSSL<=24.2.1`) and `mlflow`'s floor (`cryptography>=43.0.0`).
- **Why:** Needed a version pair both packages can agree on; a first attempt (`pip install -U pyopenssl`) over-corrected to `cryptography 49.0.0`/`pyopenssl 26.3.0`, which broke both `mlflow` and `pydrive2`'s constraints instead of fixing them.
- **Alternatives Considered:** Leaving versions unpinned and re-resolving — rejected, produced the broken pair above; downgrading `mlflow` — rejected as unnecessary and riskier than pinning two small crypto libs.
- **Tradeoffs / risks:** `asyncssh` (unrelated to this pipeline, present as a transitive dependency of some tool) now wants `cryptography>=48.0.1` and shows a resolver warning — not used anywhere in EllipticGuard's code paths, so left as-is. Revisit this pin if a future dependency add creates a real conflict.
- **Supersedes:** —

## D-018 — `dvc push` to Google Drive remains unauthenticated; not a Layer 1/2 blocker
- **Date / Layer:** Post-Layer-1
- **Context:** After fixing D-017's SSL issue, `dvc push` reached Google's OAuth step and was blocked by Google itself ("This app tried to access sensitive info... blocked to keep your account safe") — DVC's default shared OAuth client for gdrive is not verified for sensitive-scope (Drive) access under the owner's account security settings.
- **Decision:** Do not chase this further right now. `PROJECT_PLAN.md`'s Layer 1 gate only requires the assembled data to be **DVC-tracked locally** (`dvc.lock` md5 hashes) — already satisfied — not pushed to a remote. Remote push is deferred until it's actually needed (e.g., reproducibility from a clean clone, or before Layer 11 deployment).
- **Why:** Fixing the Google OAuth block requires the owner to create their own Google Cloud OAuth client (Drive API + Desktop-app credentials) — an owner-only action, and not something blocking current work. Chasing it now would violate the "don't pre-populate tasks for later layers" rule just added to `CLAUDE.md`.
- **Alternatives Considered:** Push now using the shared client anyway — not possible, Google itself refuses the flow, not a code fix.
- **Tradeoffs / risks:** Data/model artifacts exist only locally + in git-tracked `dvc.lock` hashes until the owner sets up their own OAuth client. Acceptable for now; revisit before any layer that needs the remote populated (clean-clone reproduction, deployment).
- **Supersedes:** —
