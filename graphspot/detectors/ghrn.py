from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from graphspot.detectors.bwgnn import BWGNN, _require_torch, _simple_undirected
from graphspot.graph import as_graph


class GHRN(BWGNN):
    """Graph Heterophily Resistant Network (Gao, Wang, He, Liu, Feng, Zhang:
    "Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph
    Spectrum", WWW 2023), clean-room (no reference code consulted), over graphspot's
    BWGNN encoder.

    Anomalies are heterophilous: their neighbors are mostly normal, and those
    inter-class edges push spectral energy toward high frequencies that a GNN then
    has to fight. GHRN estimates which edges are inter-class and drops them:

    1. Train the encoder (BWGNN) on the graph.
    2. Build a label signal Y: one-hot truth on labeled nodes, predicted class
       probabilities everywhere else.
    3. Measure 1-hop label change with the random-walk Laplacian,
       Z = (I - D^-1 A) Y, so z_i = y_i - mean of y over i's neighbors. For an edge
       (i, j), z_i . z_j < 0 when the endpoints deviate from their neighborhoods in
       opposite directions: evidence of an inter-class edge. A constant signal gives
       Z = 0 regardless of degrees, so degree mismatch alone never looks
       heterophilous (the symmetric normalization does not have this property).
    4. Delete up to `prune_ratio` of the undirected edges, lowest score first, only
       among edges with a negative score. Retrain a fresh encoder on the pruned
       graph. Repeat for `rounds`.

    graphspot extensions beyond the paper's transductive protocol:

    - Inductive scoring. `decision_function` replays the stages on the new graph
      with no labels: each stored stage predicts, prunes, and hands the pruned
      graph to the next, and the last stage scores.
    - `decision_scores_` is computed by the same label-free replay, so it equals
      `decision_function(train_graph)` exactly. Labels shape the pruning used for
      training only.
    - The negative-score guard in step 4, and running on the homogeneous union of
      relations (`edge_type` is ignored), as BWGNN does.

    The paper's PDF was not reachable when this was written, so the operator,
    the `prune_ratio` default and the retraining protocol follow the paper's
    published description ("1-hop label changing of the center node", pruning
    inter-class edges, then training) and still need checking against its
    equations and hyperparameter tables.
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
            adj = self._prune(adj, p_anom, y)

        self.n_edges_pruned_ = int((_simple_undirected(g.adj).nnz - adj.nnz) // 2)
        self._train(self._laplacian_from_adj(adj), g.x, y)
        self._finalize_fit(self._score(g))
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("GHRN needs node features (graph.x)")
        return self._score(g)

    def _score(self, g) -> np.ndarray:
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
                p_anom = self._probs(self._laplacian_from_adj(adj), g.x)
                adj = self._prune(adj, p_anom, unlabeled)
        finally:
            self.pre_, self.post_ = final
        return adj

    def _prune(self, adj: sp.csr_matrix, p_anom: np.ndarray, y: np.ndarray):
        """Drop up to `prune_ratio` of the undirected edges of `adj`, most
        heterophilous first, never an edge without a negative score.
        """
        upper = sp.triu(adj, k=1).tocoo()
        n_drop = int(np.floor(self.prune_ratio * upper.nnz))
        if n_drop == 0:
            return adj
        scores = self.edge_heterophily(adj, p_anom, y, upper.row, upper.col)
        # stable sort: ties broken by edge order, so pruning is deterministic
        order = np.argsort(scores, kind="stable")[:n_drop]
        drop = order[scores[order] < 0]
        keep = np.ones(upper.nnz, dtype=bool)
        keep[drop] = False
        rows, cols = upper.row[keep], upper.col[keep]
        kept = sp.coo_matrix((np.ones(keep.sum()), (rows, cols)), shape=adj.shape)
        return sp.csr_matrix(kept + kept.T)

    @staticmethod
    def edge_heterophily(adj, p_anom, y, rows, cols) -> np.ndarray:
        """z_i . z_j for each edge (rows[k], cols[k]), with Z = (I - D^-1 A) Y the
        1-hop label change of each node. Negative means the endpoints disagree: a
        likely inter-class edge. `adj` is symmetric and binary; isolated nodes get
        z = y, but no edge touches them.
        """
        sig = np.column_stack([1.0 - p_anom, p_anom]).astype(np.float64)
        labeled = y >= 0
        sig[labeled] = np.eye(2)[y[labeled]]
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float64), where=deg > 0)
        z = sig - inv[:, None] * (adj @ sig)
        return np.einsum("ij,ij->i", z[rows], z[cols])
