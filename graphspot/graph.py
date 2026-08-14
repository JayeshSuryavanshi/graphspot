from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Graph:
    """A graph normalized to scipy/numpy containers. No torch anywhere in this module.

    `adj` is authoritative for structure (aggregation, degrees). `edge_index` and the
    per-edge arrays keep the original edge list, including duplicates, in input order.
    """

    adj: sp.csr_matrix
    x: np.ndarray | None = None
    edge_index: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    edge_type: np.ndarray | None = None
    edge_time: np.ndarray | None = None
    node_labels: np.ndarray | None = None
    edge_labels: np.ndarray | None = None
    node_time: np.ndarray | None = None
    node_index: pd.Index | None = None
    feature_names: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.adj = sp.csr_matrix(self.adj)
        if self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError(f"adj must be square, got {self.adj.shape}")
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=np.float64)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
            if self.x.shape[0] != self.n_nodes:
                raise ValueError(f"x has {self.x.shape[0]} rows for {self.n_nodes} nodes")
        if self.edge_index is None and self.adj.nnz:
            coo = self.adj.tocoo()
            self.edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)

    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def n_edges(self) -> int:
        return 0 if self.edge_index is None else self.edge_index.shape[1]

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        source: str,
        target: str,
        edge_features: Sequence[str] | None = None,
        time: str | None = None,
        node_features: pd.DataFrame | None = None,
        relation: str | None = None,
        directed: bool = True,
    ) -> Graph:
        node_ids = pd.unique(pd.concat([df[source], df[target]], ignore_index=True))
        if node_features is not None:
            extra = node_features.index.difference(pd.Index(node_ids))
            node_ids = np.concatenate([node_ids, extra.to_numpy()])
        index = pd.Index(node_ids, name="node_id")
        pos = pd.Series(np.arange(len(index)), index=index)
        src = pos.loc[df[source]].to_numpy(dtype=np.int64)
        dst = pos.loc[df[target]].to_numpy(dtype=np.int64)
        n = len(index)

        rows, cols = (src, dst) if directed else (np.r_[src, dst], np.r_[dst, src])
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64)

        x = None
        feature_names: list[str] = []
        if node_features is not None:
            aligned = node_features.reindex(index)
            x = aligned.to_numpy(dtype=np.float64)
            feature_names = [str(c) for c in node_features.columns]

        edge_attr = None
        edge_feature_names: list[str] = []
        if edge_features:
            edge_attr = df[list(edge_features)].to_numpy(dtype=np.float64)
            edge_feature_names = [str(c) for c in edge_features]

        edge_time = None
        if time is not None:
            t = df[time]
            if pd.api.types.is_datetime64_any_dtype(t):
                t = t.astype("int64") / 1e9
            edge_time = t.to_numpy(dtype=np.float64)

        edge_type = None
        if relation is not None:
            edge_type = pd.Categorical(df[relation]).codes.astype(np.int64)

        return cls(
            adj=adj,
            x=x,
            edge_index=np.vstack([src, dst]),
            edge_attr=edge_attr,
            edge_type=edge_type,
            edge_time=edge_time,
            node_index=index,
            feature_names=feature_names,
            edge_feature_names=edge_feature_names,
        )

    @classmethod
    def from_scipy(cls, adj: sp.spmatrix, *, x: np.ndarray | None = None) -> Graph:
        return cls(adj=sp.csr_matrix(adj), x=x)

    @classmethod
    def from_networkx(cls, g: Any, *, node_features: pd.DataFrame | None = None) -> Graph:
        import networkx as nx

        nodes = list(g.nodes())
        adj = nx.to_scipy_sparse_array(g, nodelist=nodes, format="csr")
        x = None
        names: list[str] = []
        if node_features is not None:
            aligned = node_features.reindex(pd.Index(nodes))
            x = aligned.to_numpy(dtype=np.float64)
            names = [str(c) for c in node_features.columns]
        return cls(adj=sp.csr_matrix(adj), x=x, node_index=pd.Index(nodes), feature_names=names)

    def subgraph(self, nodes: np.ndarray) -> Graph:
        """Structural subgraph over `nodes`, reindexed to 0..k-1. Per-edge arrays are dropped."""
        nodes = np.asarray(nodes)
        if nodes.dtype == bool:
            nodes = np.flatnonzero(nodes)
        adj = self.adj[nodes][:, nodes]
        return Graph(
            adj=adj,
            x=None if self.x is None else self.x[nodes],
            node_labels=None if self.node_labels is None else self.node_labels[nodes],
            node_index=None if self.node_index is None else self.node_index[nodes],
            feature_names=list(self.feature_names),
        )

    def before(self, t: float) -> Graph:
        """Edges strictly before `t`, over the same node set. Node ids are preserved."""
        if self.edge_time is None:
            raise ValueError("Graph has no edge_time; build it with from_pandas(time=...)")
        keep = self.edge_time < t
        ei = self.edge_index[:, keep]
        adj = sp.csr_matrix(
            (np.ones(ei.shape[1]), (ei[0], ei[1])),
            shape=(self.n_nodes, self.n_nodes),
            dtype=np.float64,
        )
        return Graph(
            adj=adj,
            x=self.x,
            edge_index=ei,
            edge_attr=None if self.edge_attr is None else self.edge_attr[keep],
            edge_type=None if self.edge_type is None else self.edge_type[keep],
            edge_time=self.edge_time[keep],
            node_labels=self.node_labels,
            edge_labels=None if self.edge_labels is None else self.edge_labels[keep],
            node_index=self.node_index,
            feature_names=list(self.feature_names),
            edge_feature_names=list(self.edge_feature_names),
        )


def as_graph(g: Any, **kw: Any) -> Graph:
    """Single normalization funnel used by every detector."""
    if isinstance(g, Graph):
        return g
    if sp.issparse(g):
        return Graph.from_scipy(g, **kw)
    if isinstance(g, np.ndarray) and g.ndim == 2 and g.shape[0] == g.shape[1]:
        return Graph.from_scipy(sp.csr_matrix(g), **kw)
    if isinstance(g, pd.DataFrame):
        return Graph.from_pandas(g, **kw)
    mod = type(g).__module__
    if mod.startswith("networkx"):
        return Graph.from_networkx(g, **kw)
    if hasattr(g, "edge_index") and hasattr(g, "num_nodes"):
        ei = np.asarray(g.edge_index.cpu().numpy(), dtype=np.int64)
        n = int(g.num_nodes)
        adj = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        x = None if g.x is None else np.asarray(g.x.cpu().numpy(), dtype=np.float64)
        return Graph(adj=adj, x=x, edge_index=ei)
    raise TypeError(f"Cannot interpret {type(g).__name__} as a graph")
