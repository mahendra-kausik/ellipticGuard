"""DVC stage entrypoint (Layer 2): run the temporal split + per-time-step EDA.

Run from the repo root: `python pipelines/split_eda.py`
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.eda import compute_time_step_stats
from src.data.split import make_temporal_split, split_edges

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_PATH = REPO_ROOT / "params.yaml"


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())["split"]

    nodes_df = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges_df = pd.read_parquet(PROCESSED_DIR / "edges.parquet")

    split = make_temporal_split(
        nodes_df,
        train_end=params["train_end"],
        val_start=params["val_start"],
        test_start=params["test_start"],
        test_end=params["test_end"],
    )
    edge_split = split_edges(edges_df, nodes_df, split)

    print("=== Layer 2 temporal split summary ===")
    for name, part_nodes, part_edges in (
        ("train", split.train, edge_split.train),
        ("val", split.val, edge_split.val),
        ("test", split.test, edge_split.test),
    ):
        steps = part_nodes["time_step"]
        print(
            f"  {name}: n_nodes={len(part_nodes)}, n_edges={len(part_edges)}, "
            f"time_step_range=[{steps.min()}, {steps.max()}]"
        )

    eda_df = compute_time_step_stats(nodes_df)
    eda_path = PROCESSED_DIR / "eda_per_time_step.csv"
    eda_df.to_csv(eda_path, index=False)
    print(f"Wrote {eda_path}")
    print(eda_df.to_string(index=False))


if __name__ == "__main__":
    main()
