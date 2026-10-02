from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from graphspot.detectors.bwgnn import BWGNN, _require_torch, _simple_undirected
from graphspot.graph import as_graph


class GHRN(BWGNN):
    """Graph Heterophily Reduction Network (Gao, Wang, Li, Feng, Li, Chen: "Addressing
    Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum", WWW 2023),
    implemented clean-room from the paper, over graphspot's BWGNN encoder.

    Anomalies are heterophilous: their neighbors are mostly normal, and those
    inter-class edges push spectral energy toward high frequencies that a GNN then
    has to fight. GHRN estimates which edges are inter-class and drops them:

    1. Train the encoder (BWGNN) on the graph.
    2. Build a label signal Y: one-hot truth on labeled nodes, predicted class
       probabilities everywhere else.
    3. High-pass it, Z = L Y, with L the symmetric normalized Laplacian. For an
       edge (i, j), z_i . z_j is positive when both endpoints sit on the same side
       of their neighborhoods' label average and negative when they disagree.
    4. Delete the `prune_ratio` fraction of undirected edges with the lowest score,
       then retrain a fresh encoder on the pruned graph. Repeat for `rounds`.

    Inductive: `decision_function` replays the same procedure on the new graph with
    no labels: each stored stage predicts, prunes, and hands the pruned graph to the
    next, and the last stage scores. Pruning on unseen data uses predictions only,
    so it never needs labels it does not have.
    """

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        prune_ratio: float = 0.015,
        rounds: int = 1,
        order: int = 2,
        hidden: int = 64,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(
            level=level,
            order=order,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            contamination=contamination,
            random_state=random_state,
        )
        if not 0.0 <= prune_ratio < 1.0:
            raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}")
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")
        self.prune_ratio = prune_ratio
        self.rounds = rounds

    def fit(self, graph: Any, y: np.ndarray | None = None) -> GHRN:
        _require_torch()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("GHRN needs node features (graph.x)")
        y = self._validate_labels(g, y, self.level)

        adj = _simple_undirected(g.adj)
        self.stages_: list[tuple[Any, Any]] = []
        for _ in range(self.rounds):
            lap = self._laplacian_from_adj(adj)
            self._train(lap, g.x, y)
            p_anom = self._probs(lap, g.x)  # leaves the modules in eval mode
            self.stages_.append((copy.deepcopy(self.pre_), copy.deepcopy(self.post_)))
            adj = self._prune(adj, lap, p_anom, y)

        self.n_edges_pruned_ = int((_simple_undirected(g.adj).nnz - adj.nnz) // 2)
        lap = self._laplacian_from_adj(adj)
        self._train(lap, g.x, y)
        self._finalize_fit(self._probs(lap, g.x))
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("GHRN needs node features (graph.x)")
        return self._probs(self._laplacian_from_adj(self._replay_pruning(g)), g.x)

    def _replay_pruning(self, g) -> sp.csr_matrix:
        """Run every stored stage on `g` without labels. Swaps the stage models in and
        restores the final model, so scoring leaves the detector unchanged.
        """
        final = (self.pre_, self.post_)
        adj = _simple_undirected(g.adj)
        unlabeled = np.full(g.n_nodes, -1, dtype=np.int64)
        try:
            for pre, post in self.stages_:
                self.pre_, self.post_ = pre, post
                lap = self._laplacian_from_adj(adj)
                adj = self._prune(adj, lap, self._probs(lap, g.x), unlabeled)
        finally:
            self.pre_, self.post_ = final
        return adj

    def _prune(self, adj: sp.csr_matrix, lap, p_anom: np.ndarray, y: np.ndarray):
        """Drop the `prune_ratio` most heterophilous undirected edges of `adj`."""
        upper = sp.triu(adj, k=1).tocoo()
        n_drop = int(np.floor(self.prune_ratio * upper.nnz))
        if n_drop == 0:
            return adj
        scores = self.edge_heterophily(lap, p_anom, y, upper.row, upper.col)
        # stable sort: ties broken by edge order, so pruning is deterministic
        drop = np.argsort(scores, kind="stable")[:n_drop]
        keep = np.ones(upper.nnz, dtype=bool)
        keep[drop] = False
        rows, cols = upper.row[keep], upper.col[keep]
        kept = sp.coo_matrix((np.ones(keep.sum()), (rows, cols)), shape=adj.shape)
        return sp.csr_matrix(kept + kept.T)

    @staticmethod
    def edge_heterophily(lap, p_anom, y, rows, cols) -> np.ndarray:
        """z_i . z_j for each edge, with Z = L Y the high-passed label signal. Low
        (negative) means the endpoints disagree: a likely inter-class edge.
        """
        torch = _require_torch()
        sig = np.column_stack([1.0 - p_anom, p_anom])
        labeled = y >= 0
        sig[labeled] = np.eye(2)[y[labeled]]
        z = torch.sparse.mm(lap, torch.tensor(sig, dtype=torch.float32)).numpy()
        return np.einsum("ij,ij->i", z[rows], z[cols]).astype(np.float64)
