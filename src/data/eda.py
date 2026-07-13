"""Per-time-step EDA (Layer 2): node counts and illicit rate by time step.

Illicit rate is reported over *labeled* nodes only (illicit + licit), per D-002 —
unknown-labeled nodes are excluded from the rate denominator, not treated as licit.
"""
import pandas as pd


def compute_time_step_stats(nodes_df: pd.DataFrame) -> pd.DataFrame:
    grouped = nodes_df.groupby("time_step")["label"]
    stats = pd.DataFrame({
        "n_nodes": grouped.size(),
        "illicit": grouped.apply(lambda s: (s == 1).sum()),
        "licit": grouped.apply(lambda s: (s == 0).sum()),
        "unknown": grouped.apply(lambda s: s.isna().sum()),
    })
    labeled = stats["illicit"] + stats["licit"]
    stats["illicit_rate_labeled"] = (stats["illicit"] / labeled).where(labeled > 0)
    return stats.reset_index()
