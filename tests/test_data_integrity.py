from pathlib import Path

from src.data.loaders import assemble_node_table, check_integrity, load_classes, load_edgelist, load_features

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_layer1_integrity():
    features_df = load_features(RAW_DIR / "elliptic_txs_features.csv")
    classes_df = load_classes(RAW_DIR / "elliptic_txs_classes.csv")
    edges_df = load_edgelist(RAW_DIR / "elliptic_txs_edgelist.csv")

    nodes_df = assemble_node_table(features_df, classes_df)
    summary = check_integrity(nodes_df, edges_df)

    assert summary["n_nodes"] == 203_769
    assert summary["n_edges"] == 234_355
    assert summary["time_step_min"] == 1
    assert summary["time_step_max"] == 49
    assert summary["n_feature_cols"] == 165
    assert summary["null_counts_features"] == 0
    assert summary["illicit"] == 4_545
    assert summary["licit"] == 42_019
    assert summary["unknown"] == 157_205
