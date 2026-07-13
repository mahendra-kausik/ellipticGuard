"""DVC stage entrypoint (Layer 1): assemble raw CSVs into a node table + edge table.

Run from the repo root: `python pipelines/assemble.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import assemble_node_table, check_integrity, load_classes, load_edgelist, load_features

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    features_df = load_features(RAW_DIR / "elliptic_txs_features.csv")
    classes_df = load_classes(RAW_DIR / "elliptic_txs_classes.csv")
    edges_df = load_edgelist(RAW_DIR / "elliptic_txs_edgelist.csv")

    nodes_df = assemble_node_table(features_df, classes_df)
    summary = check_integrity(nodes_df, edges_df)

    print("=== Layer 1 integrity summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    nodes_path = PROCESSED_DIR / "nodes.parquet"
    edges_path = PROCESSED_DIR / "edges.parquet"
    nodes_df.to_parquet(nodes_path, index=False)
    edges_df.to_parquet(edges_path, index=False)
    print(f"Wrote {nodes_path}")
    print(f"Wrote {edges_path}")


if __name__ == "__main__":
    main()
