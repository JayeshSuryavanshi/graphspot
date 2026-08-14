from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from graphspot import Graph, NeighborAggregation


def path_graph():
    adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    return Graph(adj=adj, x=np.array([[1.0], [2.0], [4.0]]), feature_names=["v"])


def test_hand_computed_values():
    na = NeighborAggregation(hops=2, aggregators=("mean", "max", "std"))
    out = na.fit_transform(path_graph())
    cols = {name: i for i, name in enumerate(na.feature_names_)}

    assert out[:, cols["v"]].tolist() == [1.0, 2.0, 4.0]
    assert out[:, cols["1hop_mean(v)"]].tolist() == [2.0, 2.5, 2.0]
    assert out[:, cols["1hop_max(v)"]].tolist() == [2.0, 4.0, 2.0]
    assert out[:, cols["1hop_std(v)"]] == pytest.approx([0.0, 1.5, 0.0])
    # hop 2 aggregates the propagated 1-hop means [2, 2.5, 2]
    assert out[:, cols["2hop_mean(v)"]].tolist() == [2.5, 2.0, 2.5]
    assert out[:, cols["2hop_std(v)"]] == pytest.approx([0.0, 0.0, 0.0])
    assert out[:, cols["log1p_degree"]] == pytest.approx(np.log1p([1, 2, 1]))


def test_isolated_node_gets_zeros():
    adj = sp.csr_matrix((4, 4))
    adj[0, 1] = adj[1, 0] = 1.0
    g = Graph(adj=sp.csr_matrix(adj), x=np.ones((4, 2)) * 7)
    out = NeighborAggregation(hops=1, aggregators=("mean", "max", "min", "sum")).fit_transform(g)
    isolated = out[3]
    assert isolated[:2].tolist() == [7.0, 7.0]
    assert isolated[2:-1].tolist() == [0.0] * (len(isolated) - 3)


def test_column_chunking_matches_unchunked():
    rng = np.random.default_rng(0)
    adj = sp.random(60, 60, density=0.1, format="csr", random_state=0)
    adj.data[:] = 1.0
    x = rng.normal(size=(60, 19))
    g = Graph(adj=adj, x=x)
    na = NeighborAggregation(hops=1, aggregators=("max",), include_self=False, include_degree=False)
    out = na.fit_transform(g)
    dense = adj.toarray().astype(bool)
    for i in range(60):
        neigh = x[dense[i]]
        expected = neigh.max(axis=0) if len(neigh) else np.zeros(19)
        assert out[i, :19] == pytest.approx(expected)


def test_feature_count_and_names():
    na = NeighborAggregation(hops=3, aggregators=("mean", "sum"))
    out = na.fit_transform(path_graph())
    assert out.shape[1] == 1 + 3 * 2 * 1 + 1
    assert len(na.feature_names_) == out.shape[1]
    assert "3hop_sum(v)" in na.feature_names_


def test_transform_rejects_wrong_width():
    na = NeighborAggregation()
    na.fit(path_graph())
    bad = Graph(adj=path_graph().adj, x=np.ones((3, 5)))
    with pytest.raises(ValueError, match="features"):
        na.transform(bad)


def test_unknown_aggregator_rejected():
    with pytest.raises(ValueError, match="Unknown aggregators"):
        NeighborAggregation(aggregators=("mean", "median"))
