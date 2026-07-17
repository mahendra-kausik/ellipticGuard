"""GNN comparison models (Layer 8, optional): GCN, GraphSAGE, EvolveGCN.

Re-implemented in plain PyTorch (not PyTorch Geometric — see DECISIONS.md D-030)
under our own temporal split, so the comparison to the XGBoost champion is
provably fair: same split, same 165 features, same preprocessor, same
`evaluate()` metric code. The owner's notebooks were audited and rejected as a
usable comparison (mismatched class weights, a floor-effect artifact in the
headline claim, a stale-state export bug, a different split) — see D-030.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.data.loaders import FEATURE_COLS
from src.models.baseline import Preprocessor

N_FEATURES = len(FEATURE_COLS)  # 165: feat_0..feat_164. time_step is constant
# within a snapshot (every node in it shares the same step), so it carries zero
# information to a per-snapshot GNN — deliberately excluded, unlike MODEL_FEATURE_COLS.


@dataclass
class Snapshot:
    """One time step's causal subgraph, ready for a GNN forward pass."""

    time_step: int
    tx_ids: np.ndarray       # (n,) txId per node, row order matches x/y/label_mask
    x: torch.Tensor          # (n, 165) float32, scaled by a train-fit preprocessor
    edge_index: torch.Tensor  # (2, E) long — directed edges, both endpoints in this step
    y: torch.Tensor          # (n,) long, 0/1 (unknown rows filled 0 — excluded via label_mask)
    label_mask: torch.Tensor  # (n,) bool, True where the label is known


def build_snapshots(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, preprocessor: Preprocessor) -> list[Snapshot]:
    """One snapshot per time step, causal by construction.

    Edges never cross time steps in this dataset (Layer 2's
    `test_no_cross_time_step_edges` already proves it), so restricting each
    snapshot to that step's nodes and edges whose both endpoints are in that
    step is enough — no time-based filtering of `edges_df` is needed beyond
    the endpoint-membership check.
    """
    snapshots = []
    for step, step_df in nodes_df.groupby("time_step"):
        step_df = step_df.reset_index(drop=True)
        tx_ids = step_df["txId"].to_numpy()
        id_to_row = pd.Series(np.arange(len(tx_ids)), index=tx_ids)

        x = torch.tensor(preprocessor.transform(step_df[FEATURE_COLS]), dtype=torch.float32)

        node_id_set = set(tx_ids)
        step_edges = edges_df[edges_df["txId1"].isin(node_id_set) & edges_df["txId2"].isin(node_id_set)]
        if len(step_edges):
            src = id_to_row[step_edges["txId1"]].to_numpy()
            dst = id_to_row[step_edges["txId2"]].to_numpy()
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        label_mask = torch.tensor(step_df["label"].notna().to_numpy(), dtype=torch.bool)
        y = torch.tensor(step_df["label"].fillna(0).to_numpy(), dtype=torch.long)

        snapshots.append(
            Snapshot(time_step=int(step), tx_ids=tx_ids, x=x, edge_index=edge_index, y=y, label_mask=label_mask)
        )
    return snapshots


def _undirected_unique_edges(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Symmetrize (both directions count as connected) and dedupe.

    GCN/SAGE message passing treats the graph as undirected (standard practice —
    the raw txId1->txId2 direction is a transaction-flow artifact, not a
    constraint on which nodes can inform which). Deduping avoids double-counting
    degree when the raw edge list already contains both directions of a pair.
    """
    if edge_index.numel() == 0:
        return edge_index
    sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    codes = sym[0] * n + sym[1]
    unique_codes = torch.unique(codes)
    return torch.stack([unique_codes // n, unique_codes % n])


def normalize_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops: Â = D^-1/2 (A+I) D^-1/2, sparse."""
    sym = _undirected_unique_edges(edge_index, n)
    self_loops = torch.arange(n, dtype=torch.long)
    idx = torch.cat([sym, torch.stack([self_loops, self_loops])], dim=1)
    values = torch.ones(idx.shape[1])

    deg = torch.zeros(n).scatter_add_(0, idx[0], values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_values = deg_inv_sqrt[idx[0]] * values * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, norm_values, (n, n)).coalesce()


def row_normalize_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Row-normalized adjacency, no self-loops: D^-1 A — mean neighbour aggregation for SAGE."""
    sym = _undirected_unique_edges(edge_index, n)
    if sym.numel() == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n))
    values = torch.ones(sym.shape[1])
    deg = torch.zeros(n).scatter_add_(0, sym[0], values)
    deg_inv = deg.pow(-1.0)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    norm_values = deg_inv[sym[0]] * values
    return torch.sparse_coo_tensor(sym, norm_values, (n, n)).coalesce()


class GCN(nn.Module):
    """165 -> 64 -> 32 -> 2. Each layer: Â @ (X @ W) + b."""

    ADJ_TYPE = "sym"

    def __init__(self, in_dim: int = N_FEATURES, h1: int = 64, h2: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(in_dim, h1), nn.Linear(h1, h2), nn.Linear(h2, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(torch.sparse.mm(adj, self.l1(x)))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(torch.sparse.mm(adj, self.l2(h)))
        h = F.dropout(h, self.dropout, self.training)
        return torch.sparse.mm(adj, self.l3(h))


class GraphSAGE(nn.Module):
    """165 -> 64 -> 32 -> 2, mean aggregation: X @ W_self + (D^-1 A X) @ W_neigh."""

    ADJ_TYPE = "row"

    def __init__(self, in_dim: int = N_FEATURES, h1: int = 64, h2: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.self1, self.neigh1 = nn.Linear(in_dim, h1), nn.Linear(in_dim, h1)
        self.self2, self.neigh2 = nn.Linear(h1, h2), nn.Linear(h1, h2)
        self.self3, self.neigh3 = nn.Linear(h2, out_dim), nn.Linear(h2, out_dim)
        self.dropout = dropout

    def _layer(self, x: torch.Tensor, adj: torch.Tensor, w_self: nn.Linear, w_neigh: nn.Linear) -> torch.Tensor:
        return w_self(x) + w_neigh(torch.sparse.mm(adj, x))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self._layer(x, adj, self.self1, self.neigh1))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self._layer(h, adj, self.self2, self.neigh2))
        h = F.dropout(h, self.dropout, self.training)
        return self._layer(h, adj, self.self3, self.neigh3)


class EvolveGCN(nn.Module):
    """EvolveGCN-O: 165 -> 64 -> 32 -> 2, each layer's weight matrix evolved by a
    GRUCell across chronological snapshots instead of staying fixed (GCN's case).

    `evolve()` advances every layer's weight one GRU step and is called by the
    caller between chronologically-ordered snapshots. It detaches the previous
    weight before feeding it to the GRU, truncating backprop-through-time to a
    window of 1 snapshot — otherwise the freed autograd graph from an earlier
    snapshot's already-completed backward pass would be walked again on the next
    snapshot's backward, which errors. `reset_state()` returns to the learned
    initial weights (call at the start of each training epoch / eval pass).
    """

    ADJ_TYPE = "sym"

    def __init__(self, in_dim: int = N_FEATURES, h1: int = 64, h2: int = 32, out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dim, h1))
        self.W2 = nn.Parameter(torch.empty(h1, h2))
        self.W3 = nn.Parameter(torch.empty(h2, out_dim))
        for w in (self.W1, self.W2, self.W3):
            nn.init.xavier_uniform_(w)
        self.gru1 = nn.GRUCell(h1, h1)
        self.gru2 = nn.GRUCell(h2, h2)
        self.gru3 = nn.GRUCell(out_dim, out_dim)
        self.dropout = dropout
        self.reset_state()

    def reset_state(self) -> None:
        # .clone() (not a bare reference) so `_w1` stays a plain tensor, never
        # auto-registered as an nn.Parameter by nn.Module.__setattr__ — evolve()
        # must be able to overwrite it with a GRU-produced tensor afterwards.
        self._w1, self._w2, self._w3 = self.W1.clone(), self.W2.clone(), self.W3.clone()

    def evolve(self) -> None:
        self._w1 = self.gru1(self._w1.detach(), self._w1.detach())
        self._w2 = self.gru2(self._w2.detach(), self._w2.detach())
        self._w3 = self.gru3(self._w3.detach(), self._w3.detach())

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(torch.sparse.mm(adj, x @ self._w1))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(torch.sparse.mm(adj, h @ self._w2))
        h = F.dropout(h, self.dropout, self.training)
        return torch.sparse.mm(adj, h @ self._w3)


def _adj_for(model: nn.Module, snapshot: Snapshot) -> torch.Tensor:
    fn = row_normalize_adj if getattr(model, "ADJ_TYPE", "sym") == "row" else normalize_adj
    return fn(snapshot.edge_index, snapshot.x.shape[0])


def masked_cross_entropy(
    logits: torch.Tensor, y: torch.Tensor, label_mask: torch.Tensor, class_weight: float = 1.0
) -> torch.Tensor:
    """Weighted CE over labeled nodes only — unknown-labelled nodes (label_mask
    False) still take part in message passing (via the forward pass that
    produced `logits`) but never contribute to the loss."""
    weight = torch.tensor([1.0, class_weight], dtype=logits.dtype)
    return F.cross_entropy(logits[label_mask], y[label_mask], weight=weight)


def train_gnn(model: nn.Module, snapshots: list[Snapshot], train_idx: list[int], val_idx: list[int], cfg: dict) -> dict:
    """Train on `train_idx` snapshots, early-stopping on val illicit-F1.

    `train_idx`/`val_idx` are positions into `snapshots`, assumed chronological.
    For EvolveGCN, `model.evolve()` advances the weights once per snapshot seen
    (train or val) so the recurrent state tracks calendar order; GCN/GraphSAGE
    have no such state and ignore it.
    """
    from sklearn.metrics import f1_score

    adjs = [_adj_for(model, s) for s in snapshots]
    is_recurrent = hasattr(model, "evolve")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val_f1, best_state, patience_left, epoch = -1.0, None, cfg["patience"], 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        if is_recurrent:
            model.reset_state()
        for i in train_idx:
            s = snapshots[i]
            if s.label_mask.any():
                logits = model(s.x, adjs[i])
                loss = masked_cross_entropy(logits, s.y, s.label_mask, cfg["class_weight"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if is_recurrent:
                model.evolve()

        model.eval()
        y_true_all, y_prob_all = [], []
        with torch.no_grad():
            for i in val_idx:
                s = snapshots[i]
                if s.label_mask.any():
                    probs = F.softmax(model(s.x, adjs[i]), dim=1)[:, 1]
                    y_true_all.append(s.y[s.label_mask].numpy())
                    y_prob_all.append(probs[s.label_mask].numpy())
                if is_recurrent:
                    model.evolve()

        if not y_true_all:
            continue
        y_true = np.concatenate(y_true_all)
        y_prob = np.concatenate(y_prob_all)
        val_f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), pos_label=1, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_f1": best_val_f1, "epochs_run": epoch + 1}


def predict_snapshots(model: nn.Module, snapshots: list[Snapshot], idx: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `model` over the given snapshot positions. Returns (tx_ids, y_true,
    y_prob) stacked across snapshots, labeled rows only."""
    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()
    tx_ids_all, y_true_all, y_prob_all = [], [], []
    with torch.no_grad():
        for i in idx:
            s = snapshots[i]
            adj = _adj_for(model, s)
            if s.label_mask.any():
                probs = F.softmax(model(s.x, adj), dim=1)[:, 1]
                mask = s.label_mask.numpy()
                tx_ids_all.append(s.tx_ids[mask])
                y_true_all.append(s.y[s.label_mask].numpy())
                y_prob_all.append(probs[s.label_mask].numpy())
            if hasattr(model, "evolve"):
                model.evolve()
    return np.concatenate(tx_ids_all), np.concatenate(y_true_all), np.concatenate(y_prob_all)


class SnapshotPredictor:
    """`.predict`/`.predict_proba` adapter so GNN predictions flow through the
    exact same `baseline.evaluate`/`per_time_step_metrics` code as XGBoost —
    the fairness claim rests on this being literally the same metric function,
    not a reimplementation. Keyed by DataFrame *index* (not row position):
    `per_time_step_metrics` calls `evaluate()` on per-time-step subsets of a
    larger frame, and label-filtering (`df[df["label"].notna()]`) preserves the
    original index, so a lookup keyed on it works for any such subset.
    """

    def __init__(self, index_to_prob: dict):
        self._index_to_prob = index_to_prob

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.array([self._index_to_prob[i] for i in X.index])
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_evaluate_adapter(
    model: nn.Module, snapshots: list[Snapshot], idx: list[int], source_df: pd.DataFrame
) -> SnapshotPredictor:
    """Predict over `idx` snapshots and key the result by `source_df`'s index
    (e.g. `split.test`) so the adapter aligns with whatever row subset
    `evaluate()`/`per_time_step_metrics()` slices from it."""
    tx_ids, _, y_prob = predict_snapshots(model, snapshots, idx)
    txid_to_index = pd.Series(source_df.index, index=source_df["txId"])
    index_to_prob = {int(txid_to_index[tx]): float(p) for tx, p in zip(tx_ids, y_prob)}
    return SnapshotPredictor(index_to_prob)
