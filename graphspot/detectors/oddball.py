from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from graphspot.base import BaseDetector
from graphspot.graph import Graph, as_graph

_CHUNK_ROWS = 4096


def _egonet_features(g: Graph) -> np.ndarray:
    """Per node: neighbor count N and egonet edge count E, on the binarized
    symmetrized graph. E = deg + (edges among neighbors), and each edge (u, v) with
    both endpoints adjacent to i closes a triangle through i, so the chunked
    ((A @ A) * A).sum(axis=1) counts it twice, once per direction. Cost is
    O(sum of squared degrees); fine for transaction-shaped graphs, measured before
    being claimed for hub-heavy ones.
    """
    adj = g.adj + g.adj.T
    adj = sp.csr_matrix(adj)
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    twice_neighbor_edges = np.zeros(n)
    for lo in range(0, n, _CHUNK_ROWS):
        block = adj[lo : lo + _CHUNK_ROWS]
        twice_neighbor_edges[lo : lo + _CHUNK_ROWS] = np.asarray(
            (block @ adj).multiply(block).sum(axis=1)
        ).ravel()
    edges = deg + twice_neighbor_edges / 2.0
    return np.column_stack([deg, edges])


class OddBall(BaseDetector):
    """OddBall (Akoglu, McGlohon, Faloutsos, PAKDD 2010), egonet density power law.

    Unsupervised and feature-free. Fits log E = theta log N + log C over the fit
    graph's egonets and scores each node by its out-of-line distance
    (max(E, E_hat) / min(E, E_hat)) * log(|E - E_hat| + 1), so both near-cliques
    (E far above the law) and near-stars (E far below it) score high. The fitted
    law transfers: `decision_function` measures a new graph's egonets against it.
    """

    supported_levels = ("node",)
    inductive = True

    def fit(self, graph: Any, y: np.ndarray | None = None) -> OddBall:
        feats = _egonet_features(as_graph(graph))
        active = feats[:, 0] >= 1
        if active.sum() < 3:
            raise ValueError("OddBall needs at least 3 non-isolated nodes to fit a power law")
        log_n = np.log10(feats[active, 0])
        log_e = np.log10(feats[active, 1])
        self.theta_, self.log_c_ = np.polyfit(log_n, log_e, 1)
        self._finalize_fit(self._out_of_line(feats))
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        return self._out_of_line(_egonet_features(as_graph(graph)))

    def _out_of_line(self, feats: np.ndarray) -> np.ndarray:
        n, e = feats[:, 0], feats[:, 1]
        scores = np.zeros(len(feats))
        active = n >= 1
        expected = 10.0**self.log_c_ * n[active] ** self.theta_
        actual = e[active]
        hi = np.maximum(actual, expected)
        lo = np.maximum(np.minimum(actual, expected), 1e-12)
        scores[active] = (hi / lo) * np.log(np.abs(actual - expected) + 1.0)
        return scores
