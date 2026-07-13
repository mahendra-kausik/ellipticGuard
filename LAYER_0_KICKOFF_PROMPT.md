
We are building **EllipticGuard**, an AML illicit-transaction detector on the Elliptic Bitcoin graph.
This is a resume project and must be defensible in interviews — correctness and honesty matter more than speed.

**Before writing any code, do the following in order:**

1. Read `CLAUDE.md` in full.
2. Read `SESSION_LOG.md` — check the "Owner action items" section and confirm all prerequisites are met. If any are not met, STOP and tell me what is missing before doing anything else.
3. Read `PROJECT_PLAN.md` — locate **Layer 0 — Scaffolding & environment** and read only that section.

**Then build Layer 0 and only Layer 0:**

- Create the full repository folder structure as specified in `PROJECT_PLAN.md` §5.
- Use a venv for this project
- Add `requirements.txt` (already provided — do not regenerate it, just ensure it is in place).
- Add `.env.example` (already provided — do not regenerate it, just ensure it is in place).
- Create `.gitignore` — exclude: `.env`, `data/`, `mlruns/`, `__pycache__/`, `*.pyc`, `.dvc/cache`, `dist/`, `*.egg-info/`, `.DS_Store`, `*.ipynb_checkpoints/`.
- Initialise git (`git init`) if not already done.
- Initialise DVC (`dvc init`). Configure the Google Drive remote using the folder ID from `.env` (`GDRIVE_FOLDER_ID`). Do not hardcode the folder ID in any tracked file — read it from the environment.
- Create an empty `dvc.yaml` and `params.yaml` (stubs; stages will be added per layer).
- Create an empty `README.md` with just the project title and a "work in progress" note.
- Create the MLflow tracking directory (`mlruns/`) and confirm `import mlflow` works.
- Write one trivial pytest smoke test (`tests/test_smoke.py`) that asserts `1 + 1 == 2` — its only purpose is to confirm the test runner is wired.
- Run `pytest` and show the output.
- Run `dvc status` and show the output.
- Run `python -c "import mlflow; print(mlflow.__version__)"` and show the output.

**Secrets rule:** the `.env` file must be git-ignored and must never appear in any commit, print statement, or log. Only `.env.example` (with empty placeholders) is committed.

**Layer split rule:** if Layer 0 feels too large to do cleanly in one pass, split it into 0a / 0b, record the split in `SESSION_LOG.md`, and implement the subparts in order.

When the gate is met, run the **Layer end** ritual from `CLAUDE.md`, then the **Session end** ritual, and end your message with:
`✅ Layer 0 complete. Gate results above. Shall I proceed to Layer 1? (yes / adjust / stop)`
