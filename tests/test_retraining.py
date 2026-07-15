"""Layer 10 gate tests: the drift-or-performance flag fires/stays quiet
correctly, champion/challenger promotes a genuinely-better model and holds a
genuinely-worse one, and a full replay over a tiny synthetic frame returns a
valid per-step schema. All synthetic/data-free — runs in CI (no `needs_data`).
"""
import numpy as np
import pandas as pd

from src.data.loaders import FEATURE_COLS
from src.models.advanced import PROVIDED_FEATURE_COLS, build_xy_features, scale_pos_weight, train_xgb
from src.retraining.replay import champion_challenger, flag_decision, run_replay, train_challenger


def test_flag_fires_below_f1_floor():
    flag = flag_decision(f1=0.2, drift_score=0.02, f1_floor=0.5, drift_ceiling=0.1)
    assert flag["fired"]
    assert "f1=" in flag["reason"]


def test_flag_fires_above_drift_ceiling():
    flag = flag_decision(f1=0.9, drift_score=0.5, f1_floor=0.5, drift_ceiling=0.1)
    assert flag["fired"]
    assert "drift=" in flag["reason"]


def test_flag_quiet_when_nominal():
    flag = flag_decision(f1=0.9, drift_score=0.02, f1_floor=0.5, drift_ceiling=0.1)
    assert not flag["fired"]
    assert flag["reason"] == "nominal"


def _fake_df(steps: range, n_per_step: int = 40, illicit_signal: bool = True, seed: int = 0) -> pd.DataFrame:
    """Synthetic labeled frame. When illicit_signal, feat_0 separates the classes
    so a model trained on it clearly beats a model with no signal. `seed` must
    differ across independently-generated frames (train/holdout/challenger) —
    otherwise numpy's rng restarting at the same seed makes separately-called
    frames coincidentally duplicate each other's rows, faking a "perfect" score."""
    rng = np.random.default_rng(seed)
    rows = []
    for step in steps:
        for i in range(n_per_step):
            label = 1 if i < n_per_step // 4 else 0
            row = {"txId": step * 100000 + i, "time_step": step, "label": label}
            for col in FEATURE_COLS:
                if col == "feat_0" and illicit_signal:
                    row[col] = rng.normal(3.0 if label == 1 else -3.0, 0.5)
                else:
                    row[col] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_champion_challenger_promotes_strictly_better_model():
    train_df = _fake_df(range(1, 5), illicit_signal=False, seed=1)  # champion sees no real signal
    holdout_df = _fake_df(range(5, 6), illicit_signal=True, seed=2)  # holdout has real signal
    challenger_train_df = _fake_df(range(1, 5), illicit_signal=True, seed=3)  # different rows, same signal rule
    challenger_df = pd.concat([challenger_train_df, holdout_df], ignore_index=True)  # sees signal + the holdout itself

    X_c, y_c = build_xy_features(train_df, PROVIDED_FEATURE_COLS)
    champion = train_xgb(X_c, y_c, {"max_depth": 3, "n_estimators": 20}, scale_pos_weight(y_c), random_state=42)
    challenger = train_challenger(challenger_df, PROVIDED_FEATURE_COLS, {"max_depth": 3, "n_estimators": 20}, random_state=42)

    result = champion_challenger(champion, challenger, holdout_df, PROVIDED_FEATURE_COLS)
    assert result["promote"]
    assert result["challenger_f1"] > result["champion_f1"]


def test_champion_challenger_holds_when_challenger_not_better():
    # Champion and challenger trained identically — neither should beat the other by the margin.
    df = _fake_df(range(1, 6), illicit_signal=True)
    X, y = build_xy_features(df, PROVIDED_FEATURE_COLS)
    params = {"max_depth": 3, "n_estimators": 20}
    champion = train_xgb(X, y, params, scale_pos_weight(y), random_state=42)
    challenger = train_xgb(X, y, params, scale_pos_weight(y), random_state=42)

    result = champion_challenger(champion, challenger, df, PROVIDED_FEATURE_COLS, margin=0.01)
    assert not result["promote"]


def test_run_replay_schema_and_flag_fires_on_collapse():
    train_df = _fake_df(range(1, 5), illicit_signal=False, seed=1)
    X, y = build_xy_features(train_df, PROVIDED_FEATURE_COLS)
    champion = train_xgb(X, y, {"max_depth": 3, "n_estimators": 20}, scale_pos_weight(y), random_state=42)

    # Steps 5-6 have real signal (champion should score OK-ish); step 7 has all-one-class
    # labels, which collapses illicit-F1 to 0 — simulates a T43-style regime change.
    # A challenger is trained on data strictly BEFORE each replayed step (never the
    # step itself — see run_replay's docstring), so steps 1-4 must be present too
    # (reusing train_df, which already has that range's rows and labels).
    df = pd.concat([
        train_df,
        _fake_df(range(5, 7), illicit_signal=True, seed=2),
        _fake_df(range(7, 8), illicit_signal=True, seed=3).assign(label=0),
    ], ignore_index=True)
    drift_by_step = {5: 0.02, 6: 0.03, 7: 0.4}

    log = run_replay(
        df, PROVIDED_FEATURE_COLS, start_step=5, end_step=7, champion=champion,
        drift_by_step=drift_by_step, f1_floor=0.3, drift_ceiling=0.1,
        params={"max_depth": 3, "n_estimators": 20}, random_state=42,
    )

    assert list(log["time_step"]) == [5, 6, 7]
    assert set(log.columns) == {
        "time_step", "window_f1", "drift_score", "flag_fired", "flag_reason",
        "action", "champion_f1", "challenger_f1",
    }
    assert log.loc[log["time_step"] == 7, "flag_fired"].iloc[0]  # collapse step trips the flag
    assert log["action"].isin(["promote", "hold"]).all()
