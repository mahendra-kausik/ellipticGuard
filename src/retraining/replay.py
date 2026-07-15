"""Retraining loop & replay (Layer 10): stream the 49 fixed time steps in
sequence, fire a drift-or-performance flag, and run a champion/challenger
promotion check.

There is no live feed (PROJECT_PLAN.md §1) — this is an explicit replay/
simulation of streaming over the dataset's own time steps, not real online
learning. The champion passed in by the pipeline entrypoint is the Layer 6
train-only champion (steps 1-29) — the same model whose T43 collapse Layer 6
already documented — so the replay's story ties directly to that prior
evidence rather than an artificial staleness knob. See D-026.

Reuses existing building blocks, no re-implementation:
  - `evaluate` (src/models/baseline.py) for illicit-F1 on a window.
  - `build_xy_features`, `train_xgb`, `scale_pos_weight` (src/models/advanced.py).
"""
import pandas as pd

from src.models.advanced import build_xy_features, scale_pos_weight, train_xgb
from src.models.baseline import evaluate


def window_f1(model, df: pd.DataFrame, feature_cols: list[str], step: int) -> float | None:
    """Illicit-F1 on one time step's labeled nodes. None if the step has no labeled rows."""
    window = df[df["time_step"] == step]
    labeled = window[window["label"].notna()]
    if labeled.empty:
        return None
    X, y = labeled[feature_cols], labeled["label"].astype(int)
    return evaluate(model, X, y)["illicit_f1"]


def flag_decision(f1: float | None, drift_score: float | None, f1_floor: float, drift_ceiling: float) -> dict:
    """Fires when the window's illicit-F1 drops below the floor OR its target-drift
    score exceeds the ceiling. Either signal alone is sufficient — this is the
    "drift-or-performance" flag from PROJECT_PLAN.md Layer 10."""
    reasons = []
    if f1 is not None and f1 < f1_floor:
        reasons.append(f"f1={f1:.3f} < floor={f1_floor}")
    if drift_score is not None and drift_score > drift_ceiling:
        reasons.append(f"drift={drift_score:.3f} > ceiling={drift_ceiling}")
    return {"fired": bool(reasons), "reason": "; ".join(reasons) if reasons else "nominal"}


def train_challenger(df_up_to_step: pd.DataFrame, feature_cols: list[str], params: dict, random_state: int):
    """Refit a challenger on all labeled data available up to and including the
    current replayed step — the simulated "retrain on the latest data" action."""
    X, y = build_xy_features(df_up_to_step, feature_cols)
    spw = scale_pos_weight(y)
    return train_xgb(X, y, params, spw, random_state)


def champion_challenger(champion, challenger, holdout_df: pd.DataFrame, feature_cols: list[str], margin: float = 0.0) -> dict:
    """Promote only if the challenger beats the champion on the held-out window
    by at least `margin` illicit-F1. Never touches the champion's own training data."""
    X, y = build_xy_features(holdout_df, feature_cols)
    champion_f1 = evaluate(champion, X, y)["illicit_f1"]
    challenger_f1 = evaluate(challenger, X, y)["illicit_f1"]
    promote = challenger_f1 > champion_f1 + margin
    reason = (
        f"challenger {challenger_f1:.3f} beats champion {champion_f1:.3f} by >= {margin}"
        if promote
        else f"challenger {challenger_f1:.3f} does not beat champion {champion_f1:.3f} by >= {margin} — hold"
    )
    return {"promote": promote, "champion_f1": champion_f1, "challenger_f1": challenger_f1, "reason": reason}


def run_replay(
    df: pd.DataFrame,
    feature_cols: list[str],
    start_step: int,
    end_step: int,
    champion,
    drift_by_step: dict[int, float],
    f1_floor: float,
    drift_ceiling: float,
    params: dict,
    random_state: int,
    promote_margin: float = 0.0,
) -> pd.DataFrame:
    """Stream steps `start_step..end_step`, one replayed "tick" per step.

    At every step: score the fixed champion on that step's window, check the
    drift-or-performance flag (an independent escalation signal — logged, not
    gating), train a challenger on everything seen BEFORE this step (never
    including it), and evaluate both on the step's window — genuinely
    out-of-sample for both models, exactly like scoring an incoming, not-yet-
    seen window in a real retraining loop. Training the challenger on the
    current step itself would let it trivially memorize that step's labels
    (in-sample F1=1.0 always, T43 included) and defeat the entire point of the
    T43 demonstration — CLAUDE.md Directive 3 forbids that kind of leakage.
    The champion is NOT swapped mid-replay, so every step's promote/hold
    verdict is directly comparable. This is what lets one replay honestly
    demonstrate all three PROJECT_PLAN.md Layer 10 gate behaviors: the flag
    firing at/after T43, a routine promote (a challenger with a bit more
    recent same-regime data edges out the champion), and a T43 hold (a
    challenger trained on everything before T43 still can't predict T43's own
    collapsed window — a regime change, not a recoverable drift).
    """
    rows = []
    for step in range(start_step, end_step + 1):
        f1 = window_f1(champion, df, feature_cols, step)
        drift_score = drift_by_step.get(step)
        flag = flag_decision(f1, drift_score, f1_floor, drift_ceiling)

        df_seen = df[df["time_step"] < step]
        holdout = df[df["time_step"] == step]
        challenger = train_challenger(df_seen, feature_cols, params, random_state)
        result = champion_challenger(champion, challenger, holdout, feature_cols, promote_margin)
        action = "promote" if result["promote"] else "hold"

        rows.append({
            "time_step": step,
            "window_f1": f1,
            "drift_score": drift_score,
            "flag_fired": flag["fired"],
            "flag_reason": flag["reason"],
            "action": action,
            "champion_f1": result["champion_f1"],
            "challenger_f1": result["challenger_f1"],
        })
    return pd.DataFrame(rows)
