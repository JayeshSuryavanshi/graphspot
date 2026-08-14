from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from graphspot.graph import as_graph

_AGGREGATORS = ("mean", "max", "min", "std", "sum")
_COL_BLOCK = 8


def _neighbor_reduce(adj: sp.csr_matrix, x: np.ndarray, ufunc: np.ufunc) -> np.ndarray:
    """Per-row ufunc reduction over neighbor feature rows, chunked by feature column.

    `reduceat` returns the element at the start index for empty segments and cannot take a
    start index equal to nnz, so empty rows are masked to 0 afterwards and starts are clipped.
    """
    indptr, indices = adj.indptr, adj.indices
    n, d = adj.shape[0], x.shape[1]
    out = np.zeros((n, d), dtype=np.float64)
    if len(indices) == 0:
        return out
    counts = np.diff(indptr)
    starts = np.minimum(indptr[:-1], len(indices) - 1)
    empty = counts == 0
    for lo in range(0, d, _COL_BLOCK):
        gathered = x[indices, lo : lo + _COL_BLOCK]
        out[:, lo : lo + _COL_BLOCK] = ufunc.reduceat(gathered, starts, axis=0)
    out[empty] = 0.0
    return out


def _neighbor_mean(adj: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return inv[:, None] * (adj @ x)


class NeighborAggregation:
    """k-hop neighborhood feature aggregation as a fit/transform pair.

    Hop k output applies every aggregator to the hop k-1 representation; the representation
    propagated to the next hop is the neighbor mean. Cost is linear in hops (SIGN-style),
    never a materialized A^k. Every output feature is a (hop, aggregator, base_feature)
    triple, so importances map back to statements like "2hop_mean(chargeback_rate)".
    """

    def __init__(
        self,
        *,
        hops: int = 2,
        aggregators: Sequence[str] = ("mean", "max", "std"),
        include_self: bool = True,
        include_degree: bool = True,
    ):
        unknown = set(aggregators) - set(_AGGREGATORS)
        if unknown:
            raise ValueError(f"Unknown aggregators {sorted(unknown)}; choose from {_AGGREGATORS}")
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        self.hops = hops
        self.aggregators = tuple(aggregators)
        self.include_self = include_self
        self.include_degree = include_degree

    def fit(self, graph: Any) -> NeighborAggregation:
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("NeighborAggregation needs node features (graph.x)")
        self.n_features_in_ = g.x.shape[1]
        base = g.feature_names or [f"f{i}" for i in range(self.n_features_in_)]
        names: list[str] = []
        if self.include_self:
            names += list(base)
        for k in range(1, self.hops + 1):
            for agg in self.aggregators:
                names += [f"{k}hop_{agg}({b})" for b in base]
        if self.include_degree:
            names.append("log1p_degree")
        self.feature_names_ = names
        return self

    def transform(self, graph: Any) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("NeighborAggregation is not fitted; call fit first")
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("NeighborAggregation needs node features (graph.x)")
        if g.x.shape[1] != self.n_features_in_:
            raise ValueError(f"Graph has {g.x.shape[1]} features, fitted on {self.n_features_in_}")
        adj = g.adj
        blocks: list[np.ndarray] = []
        if self.include_self:
            blocks.append(g.x)
        cur = g.x
        for _ in range(self.hops):
            mean_k = _neighbor_mean(adj, cur)
            for agg in self.aggregators:
                if agg == "mean":
                    blocks.append(mean_k)
                elif agg == "sum":
                    blocks.append(adj @ cur)
                elif agg == "max":
                    blocks.append(_neighbor_reduce(adj, cur, np.maximum))
                elif agg == "min":
                    blocks.append(_neighbor_reduce(adj, cur, np.minimum))
                elif agg == "std":
                    sq = _neighbor_mean(adj, cur**2) - mean_k**2
                    blocks.append(np.sqrt(np.maximum(sq, 0.0)))
            cur = mean_k
        if self.include_degree:
            deg = np.asarray(adj.sum(axis=1)).ravel()
            blocks.append(np.log1p(deg)[:, None])
        return np.hstack(blocks)

    def fit_transform(self, graph: Any) -> np.ndarray:
        return self.fit(graph).transform(graph)

    def transform_edges(self, graph: Any) -> np.ndarray:
        """Edge features: both endpoints' aggregated neighborhoods, then raw edge attributes."""
        g = as_graph(graph)
        if g.edge_index is None:
            raise ValueError("Graph has no edges to score")
        node_feats = self.transform(g)
        src, dst = g.edge_index[0], g.edge_index[1]
        blocks = [node_feats[src], node_feats[dst]]
        if g.edge_attr is not None:
            blocks.append(g.edge_attr)
        return np.hstack(blocks)

    def edge_feature_names(self, graph: Any) -> list[str]:
        g = as_graph(graph)
        names = [f"src_{n}" for n in self.feature_names_]
        names += [f"dst_{n}" for n in self.feature_names_]
        if g.edge_attr is not None:
            base = g.edge_feature_names or [f"e{i}" for i in range(g.edge_attr.shape[1])]
            names += list(base)
        return names


def ensure_node_features(g):
    """Return `g`, or a copy carrying synthesized node features when it has none.

    A bare transaction log has no account-level features, but its edges do. The synthesized
    features are each node's mean incident edge attributes (both directions) plus log1p
    in/out degree, which is enough for neighborhood aggregation to work on a plain edge list.
    """
    from dataclasses import replace

    if g.x is not None:
        return g
    n = g.n_nodes
    deg_out = np.asarray(g.adj.sum(axis=1)).ravel()
    deg_in = np.asarray(g.adj.sum(axis=0)).ravel()
    blocks = [np.log1p(deg_out)[:, None], np.log1p(deg_in)[:, None]]
    names = ["log1p_out_degree", "log1p_in_degree"]
    if g.edge_attr is not None and g.edge_index is not None:
        d = g.edge_attr.shape[1]
        sums = np.zeros((n, d))
        counts = np.zeros(n)
        for endpoint in (g.edge_index[0], g.edge_index[1]):
            np.add.at(sums, endpoint, g.edge_attr)
            np.add.at(counts, endpoint, 1.0)
        means = sums / np.maximum(counts, 1.0)[:, None]
        blocks.append(means)
        base = g.edge_feature_names or [f"e{i}" for i in range(d)]
        names += [f"incident_mean({b})" for b in base]
    return replace(g, x=np.hstack(blocks), feature_names=names)
