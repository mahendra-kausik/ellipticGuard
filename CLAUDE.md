# CLAUDE.md — Operating Manual for Claude Code

> **Commit attribution note:** the owner is the sole contributor on this repo. Commits made by Claude Code must **not** include a `Co-Authored-By: Claude ...` trailer or any other attribution that would list Claude as a contributor.

> Read this file first, in full, at the start of **every** session before touching any other file.
> Then read `SESSION_LOG.md` to see where we are, and `PROJECT_PLAN.md` for the layer you're on.

## What we are building
**EllipticGuard** — an anti-money-laundering (AML) illicit-transaction detector on the Elliptic Bitcoin transaction graph, built as a deployable, monitored, reproducible ML system (not a notebook). Classical ML on engineered causal graph-topology features, honestly validated with a temporal split, served as a FastAPI service, with drift monitoring and a (replayed) retraining loop.

The owner is a final-year CS student putting this on a Data Science resume. **The owner must be able to defend every non-trivial decision in an interview.** That single fact drives the two most important rules below.

---

## PRIME DIRECTIVES (non-negotiable)

### 1. Build ONE layer at a time. Stop at every gate.
- The build is split into numbered layers in `PROJECT_PLAN.md`, each with an **Acceptance Gate**.
- Implement **only the current layer**. Do not scaffold, stub, or "get a head start" on future layers.
- **If a layer is too big to build well in one pass, split it into logical subparts (e.g., Layer 4a / 4b).** Record the split and the reasoning in `SESSION_LOG.md` before you start, then implement the subparts in order. Each subpart still ends with a short summary; the full layer gate applies at the end of the last subpart.
- When a layer's gate is met, **STOP** and end your message with exactly:
  `✅ Layer <N> complete. Gate results above. Shall I proceed to Layer <N+1>? (yes / adjust / stop)`
- Do not start the next layer until the owner replies.
- If you cannot meet a gate, **say so plainly** and propose options. Never fake, hardcode, or optimistically round a result to pass a gate.

### 2. Log every non-trivial decision in `DECISIONS.md` (using the template at the top of that file).
- A decision is "non-trivial" if a reasonable engineer could have chosen otherwise and an interviewer could ask "why did you do it that way?"
- **Trivial choices — variable names, obvious formatting, import ordering — do NOT need entries.** Use judgment; **when unsure, log it.**
- Write entries in plain language the owner can say out loud in an interview.
- **Never rewrite history.** If a new decision reverses or replaces an earlier one, add a **new** entry and fill in its `Supersedes:` field with the old entry's ID. Leave the old entry in place.
- Decisions are written to `DECISIONS.md` during the **Session end** ritual (see below) — but note them as you go so none are lost.

### 3. Protect against leakage. This is the whole point of the project.
- The temporal split is **sacred**: train = time steps 1–34, test = 35–49, with a validation slice carved from the **end of the training range** (steps 30–34) — never from the future test range. See `PROJECT_PLAN.md` Layer 2 for the exact partition.
- **Never** fit a scaler, imputer, encoder, feature-selector, or calibration model on anything but the training partition. Fit on train, transform on val/test.
- Graph features are computed **within each time-step subgraph** (edges never cross time steps in this dataset). Do not aggregate across time steps.
- A random train/test split is **forbidden** anywhere in the pipeline. If you ever think you need a shuffle, stop and ask.

### 4. Treat `PROJECT_PLAN.md` as authoritative — but not infallible.
- If something in the plan looks **wrong, outdated, or infeasible on the free tier / available compute**, **flag it and STOP.** Explain what looks off and propose an alternative. **Do not silently work around it** or quietly substitute a different approach.

### 5. Report honesty.
- Show real metrics from real runs. If a number is worse than hoped, report it and explain it. A modest honest result is a PASS for this project; a good-looking leaky result is a FAIL.
- Never invent benchmark numbers. When comparing to Weber et al. 2019, treat their figures as *reference targets to verify*, not ground truth to match at any cost.

---

## Working style
- **Code up the entire layer (or the current subpart) in one pass, then present it for review.** Do not stop after each function to ask for confirmation. Deliver the whole working layer, then summarize.
- After the layer is written, **run its tests and print the sanity checks** (shapes, class balance, time-step ranges, null counts) so correctness is visible in your summary. Anything that touches data partitioning, labels, or feature computation must show a sanity check.
- Keep functions pure and testable. Every layer adds at least one `pytest` test.

## Layer start (run before starting the work on a layer)
1. Briefly tell the owner what is going to be implemented in this layer
2. Inform the owner of all actions he must complete from the "Owner action items" of `SESSION_LOG.md`

## Session end (run whenever a work session stops — even mid-layer)
1. `SESSION_LOG.md` — mark the layer status, record what changed, and set the next action so the next session can pick up exactly where this one left off.
2. `DECISIONS.md` — append any non-trivial decisions made during the session (with `Supersedes:` where relevant).

## Layer end (run when a layer's gate is reached)
1. Run the layer's tests; gather gate evidence (test output, metric, artifact path).
2. Refresh the "Owner action items" section of `SESSION_LOG.md`: remove items that are now complete, and list **only** the tasks the owner must do for the layer about to start. Do **not** pre-populate tasks for later layers — add each one when its layer begins.
3. Commit and push to GitHub with a clear message (e.g., `Layer 3: baseline models + AUC-PR eval`).
4. End the message with the exact completion line from Directive 1.

## Using the owner's notebooks (when provided)
The owner has existing notebooks: **preprocessing**, **static GCN**, **EvolveGCN**, and **SHAP**. They may contain implementation errors.
- Read them **only when the relevant layer calls for it** (preprocessing → Layers 1–2; GCN/EvolveGCN → optional Layer 8; SHAP → Layer 6).
- Treat them as **reference, not gospel.** Verify logic against this plan. If you find a bug (e.g., a random split, a scaler fit on all data, a label mismap), **fix it, and log the correction in `DECISIONS.md`** with what was wrong and why the fix is correct.
- Do not import their code wholesale. Re-implement cleanly into `src/` with tests.

## Environment & hard constraints
- **Free tier only.** No paid services. If something seems to need one, flag it and stop (Directive 4).
- **Compute:** classical pipeline runs CPU-only. **A Colab T4 GPU is available if needed** — use it only for the optional GNN layer (Layer 8); do not make the core pipeline or serving depend on a GPU (Hugging Face Spaces free tier is CPU).
- **Secrets must never be leaked.** No API keys, tokens, or credentials in code, logs, prints, commits, test fixtures, or the docs. Everything sensitive lives in `.env` (git-ignored); commit only `.env.example` with empty placeholders. Verify nothing secret is staged before every push.
- Python 3.11+. Core libs: pandas, numpy, scikit-learn, xgboost, networkx (or igraph), mlflow, dvc, evidently, nannyml (optional), fastapi, uvicorn, pytest. GNN (optional): torch + torch-geometric on Colab.
- Versioning: **DVC** + Google Drive remote. Tracking/registry: **MLflow** (local `mlruns/`; registry via API). Serving: **FastAPI + Docker → Hugging Face Spaces** (free CPU-basic). CI: **GitHub Actions**.

## What NOT to do
- Don't skip ahead, batch multiple layers, or "finish the project quickly."
- Don't use a random split, ever.
- Don't fit transforms on val/test data.
- Don't fabricate metrics or benchmark comparisons.
- Don't silently work around a plan problem — flag and stop.
- Don't edit past `DECISIONS.md` entries to reflect a reversal — supersede them.
- Don't leak secrets anywhere.
- Don't attribute your behavior to this file when talking to the owner — just do the work and explain your reasoning normally.
