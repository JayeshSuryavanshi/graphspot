from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import graphspot
from graphspot import Graph, as_graph


def test_no_torch_at_import():
    import sys

    assert "torch" not in sys.modules, "importing graphspot must never pull in torch"
    assert graphspot.__version__


def test_from_pandas_roundtrip():
    df = pd.DataFrame(
        {
            "buyer": ["a", "b", "a", "c"],
            "seller": ["s1", "s1", "s2", "s2"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "ts": [1.0, 2.0, 3.0, 4.0],
        }
    )
    g = Graph.from_pandas(df, source="buyer", target="seller", edge_features=["amount"], time="ts")
    assert g.n_nodes == 5
    assert g.n_edges == 4
    assert g.edge_attr.shape == (4, 1)
    assert list(g.node_index[:3]) == ["a", "b", "c"]
    pos = {v: i for i, v in enumerate(g.node_index)}
    assert g.adj[pos["a"], pos["s1"]] == 1.0
    assert g.adj[pos["s1"], pos["a"]] == 0.0


def test_from_pandas_undirected_and_node_features():
    df = pd.DataFrame({"u": [0, 1], "v": [1, 2]})
    feats = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}, index=[0, 1, 2, 9])
    g = Graph.from_pandas(df, source="u", target="v", node_features=feats, directed=False)
    assert g.n_nodes == 4
    assert g.adj[0, 1] == g.adj[1, 0] == 1.0
    row = g.node_index.get_loc(9)
    assert g.x[row, 0] == pytest.approx(0.4)
    assert g.feature_names == ["score"]


def test_before_preserves_node_set():
    df = pd.DataFrame({"u": [0, 1, 2], "v": [1, 2, 0], "t": [1.0, 2.0, 3.0]})
    g = Graph.from_pandas(df, source="u", target="v", time="t")
    past = g.before(2.5)
    assert past.n_nodes == g.n_nodes
    assert past.n_edges == 2
    assert past.edge_time.max() == 2.0


def test_subgraph_reindexes():
    adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    g = Graph(adj=adj, x=np.arange(3.0)[:, None], node_labels=np.array([0, 1, 0]))
    sub = g.subgraph(np.array([1, 2]))
    assert sub.n_nodes == 2
    assert sub.adj[0, 1] == 1.0
    assert sub.x.ravel().tolist() == [1.0, 2.0]
    assert sub.node_labels.tolist() == [1, 0]


def test_as_graph_funnel():
    adj = sp.random(10, 10, density=0.2, format="csr")
    assert as_graph(adj).n_nodes == 10
    assert as_graph(np.eye(4)).n_nodes == 4
    g = Graph(adj=sp.csr_matrix(adj))
    assert as_graph(g) is g
    with pytest.raises(TypeError):
        as_graph(42)


def test_rectangular_adj_rejected():
    with pytest.raises(ValueError, match="square"):
        Graph(adj=sp.csr_matrix(np.ones((2, 3))))
