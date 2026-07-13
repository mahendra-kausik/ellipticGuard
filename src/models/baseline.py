"""Baseline models (Layer 3): Logistic Regression + Random Forest on the provided 166 features.

Scaler/imputer are fit on `split.train` only (CLAUDE.md Directive 3). Headline metrics
are illicit-class precision/recall/F1 + AUC-PR; accuracy is never reported (D-003).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.loaders import FEATURE_COLS

MODEL_FEATURE_COLS = ["time_step"] + FEATURE_COLS  # 166 features, per PROJECT_PLAN.md §2


def build_xy(nodes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to labeled rows only (D-002) and split into X/y."""
    labeled = nodes_df[nodes_df["label"].notna()]
    return labeled[MODEL_FEATURE_COLS], labeled["label"].astype(int)


@dataclass(frozen=True)
class Preprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X))


def fit_preprocessor(X_train: pd.DataFrame) -> Preprocessor:
    """Fit imputer + scaler on train only — never call with val/test data."""
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    return Preprocessor(imputer=imputer, scaler=scaler)


def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series, random_state: int, max_iter: int) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced", max_iter=max_iter, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: pd.Series, random_state: int, n_estimators: int, max_depth: int | None
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X: np.ndarray, y: pd.Series) -> dict:
    """Illicit-class (positive=1) precision/recall/F1 + AUC-PR + confusion matrix. No accuracy (D-003)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "illicit_precision": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "illicit_recall": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "illicit_f1": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "auc_pr": float(average_precision_score(y, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
