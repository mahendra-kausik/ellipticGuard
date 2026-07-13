"""DVC stage entrypoint (Layer 4): compute per-time-step graph topology features.

Run from the repo root: `python pipelines/build_graph_features.py`
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.graph import GRAPH_FEATURE_COLS, compute_graph_features

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges_df = pd.read_parquet(PROCESSED_DIR / "edges.parquet")

    features_df = compute_graph_features(nodes_df, edges_df)

    out_path = PROCESSED_DIR / "graph_features.parquet"
    features_df.to_parquet(out_path, index=False)

    print("=== Layer 4 graph feature summary ===")
    print(f"  n_rows={len(features_df)} (expect {len(nodes_df)})")
    print(f"  non-null coverage:\n{features_df[GRAPH_FEATURE_COLS].notna().mean().to_string()}")
    print(f"  feature distributions:\n{features_df[GRAPH_FEATURE_COLS].describe().to_string()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
