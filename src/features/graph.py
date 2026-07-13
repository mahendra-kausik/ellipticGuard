"""Per-time-step graph topology features (Layer 4).

Edges never cross time steps (verified in Layer 2's test_no_cross_time_step_edges),
so building one graph per time step and computing features only within it is
causal by construction — a node's features never see a future or another step's
subgraph.
"""
import networkx as nx
import pandas as pd

GRAPH_FEATURE_COLS = [
    "in_degree",
    "out_degree",
    "unique_neighbors",
    "pagerank",
    "clustering_coef",
    "component_size",
    "avg_neighbor_degree",
]


def build_step_graph(node_ids: pd.Series, edges_in_step: pd.DataFrame) -> nx.DiGraph:
    """Directed graph for one time step: all its nodes (incl. isolated ones) + its edges."""
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges_in_step[["txId1", "txId2"]].itertuples(index=False, name=None))
    return G


def compute_step_features(G: nx.DiGraph) -> pd.DataFrame:
    """Topology features for every node in a single time-step graph."""
    undirected = G.to_undirected()
    pagerank = nx.pagerank(G)
    clustering = nx.clustering(undirected)
    avg_neighbor_degree = nx.average_neighbor_degree(undirected)

    component_size = {}
    for component in nx.connected_components(undirected):
        for node in component:
            component_size[node] = len(component)

    return pd.DataFrame(
        {
            "txId": list(G.nodes),
            "in_degree": [G.in_degree(n) for n in G.nodes],
            "out_degree": [G.out_degree(n) for n in G.nodes],
            "unique_neighbors": [undirected.degree(n) for n in G.nodes],
            "pagerank": [pagerank[n] for n in G.nodes],
            "clustering_coef": [clustering[n] for n in G.nodes],
            "component_size": [component_size[n] for n in G.nodes],
            "avg_neighbor_degree": [avg_neighbor_degree[n] for n in G.nodes],
        }
    )


def compute_graph_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full engineered feature table, one time step's subgraph at a time."""
    frames = []
    for step, node_ids in nodes_df.groupby("time_step")["txId"]:
        node_id_set = set(node_ids)
        edges_in_step = edges_df[
            edges_df["txId1"].isin(node_id_set) & edges_df["txId2"].isin(node_id_set)
        ]
        G = build_step_graph(node_ids, edges_in_step)
        feats = compute_step_features(G)
        feats["time_step"] = step
        frames.append(feats)

    result = pd.concat(frames, ignore_index=True)
    return result[["txId", "time_step"] + GRAPH_FEATURE_COLS]
