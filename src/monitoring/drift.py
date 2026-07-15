"""Feature/target drift monitoring (Layer 9).

Reference = `split.train` (the fit-only steps 1-29 the champion was trained on — never
val/test, per CLAUDE.md Directive 3). Current = each later time step (30-49) in turn.

Uses Evidently 0.7.x's `Report`/`Dataset`/`DataDefinition` API (the pre-0.7 `ColumnMapping`
API is gone). Feature drift and target drift are computed as two independent Evidently
calls, not folded into one report:
  - Feature drift uses ALL nodes in a window (label-agnostic, mirrors what's available at
    serving time) over the anonymized covariates only (`feature_cols`, e.g. feat_0..feat_164
    — deliberately excludes `time_step`, which trivially differs from the reference range
    by construction and would inflate every window's drifted-column count regardless of
    real distributional shift).
  - Target drift uses labeled nodes only (D-002's labeled-only convention, matching
    `src/data/eda.py`'s illicit-rate denominator) so unknown-labeled rows don't dilute the
    comparison.
"""
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data.split import TemporalSplit


def _feature_dataset(df: pd.DataFrame, feature_cols: list[str]) -> Dataset:
    dd = DataDefinition(numerical_columns=feature_cols)
    return Dataset.from_pandas(df[feature_cols], data_definition=dd)


def _target_dataset(df: pd.DataFrame) -> Dataset:
    labeled = df[df["label"].notna()][["label"]].astype({"label": "category"})
    dd = DataDefinition(categorical_columns=["label"])
    return Dataset.from_pandas(labeled, data_definition=dd)


def _feature_drift_share(reference: Dataset, current: Dataset, n_columns: int) -> dict:
    report = Report([DataDriftPreset()])
    run = report.run(reference_data=reference, current_data=current)
    counts = run.dict()["metrics"][0]["value"]  # DriftedColumnsCount: {"count", "share"}
    return {"n_drifted": counts["count"], "n_columns": n_columns, "share_drifted": counts["share"]}


def _target_drift_score(reference: Dataset, current: Dataset) -> float:
    report = Report([DataDriftPreset()])
    run = report.run(reference_data=reference, current_data=current)
    return run.dict()["metrics"][1]["value"]  # ValueDrift(column=label, ...)


def drift_by_time_step(nodes_df: pd.DataFrame, split: TemporalSplit, feature_cols: list[str]) -> pd.DataFrame:
    """One row per time step after the reference range: feature-drift share + target-drift score."""
    ref_features = _feature_dataset(split.train, feature_cols)
    ref_target = _target_dataset(split.train)
    ref_steps = set(split.train["time_step"].unique())
    later_steps = sorted(set(nodes_df["time_step"].unique()) - ref_steps)

    rows = []
    for step in later_steps:
        window = nodes_df[nodes_df["time_step"] == step]
        feature_summary = _feature_drift_share(ref_features, _feature_dataset(window, feature_cols), len(feature_cols))
        target_score = _target_drift_score(ref_target, _target_dataset(window))
        rows.append({"time_step": step, **feature_summary, "target_drift_score": target_score})
    return pd.DataFrame(rows)


def save_report_html(nodes_df: pd.DataFrame, split: TemporalSplit, feature_cols: list[str], current_step: int, path: Path) -> None:
    """Save one full human-readable Evidently feature-drift report for a representative late window."""
    reference = _feature_dataset(split.train, feature_cols)
    window = nodes_df[nodes_df["time_step"] == current_step]
    current = _feature_dataset(window, feature_cols)
    report = Report([DataDriftPreset()])
    run = report.run(reference_data=reference, current_data=current)
    run.save_html(str(path))
