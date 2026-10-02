"""Multi-relation data model: relation names, edge_type validation, relations that
survive subgraph/before, and the opt-in relation loading of the DGL fraud .mat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from graphspot import Graph
from graphspot.datasets import load_amazon, load_yelpchi
from graphspot.splits import temporal_split


def rel_df():
    return pd.DataFrame(
        {
            "s": ["a", "b", "c", "a", "d"],
            "t": ["b", "c", "d", "c", "a"],
            "rel": ["same_user", "same_product", "same_user", "same_product", "same_user"],
            "ts": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_from_pandas_names_relations():
    g = Graph.from_pandas(rel_df(), source="s", target="t", relation="rel")
    assert g.relation_names == ["same_product", "same_user"]
    assert [g.relation_names[c] for c in g.edge_type] == rel_df()["rel"].tolist()


def test_edge_type_length_is_checked():
    adj = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    ei = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="edge_type has 3 rows for 2 edges"):
        Graph(adj=adj, edge_index=ei, edge_type=np.zeros(3, dtype=np.int64))


def test_subgraph_keeps_internal_edges_with_their_arrays():
    g = Graph.from_pandas(
        rel_df(), source="s", target="t", relation="rel", time="ts", edge_features=["ts"]
    )
    keep = np.array([0, 1, 2])  # a, b, c: drops every edge touching d
    sub = g.subgraph(keep)
    assert sub.relation_names == g.relation_names
    assert sub.edge_index.tolist() == [[0, 1, 0], [1, 2, 2]]
    assert sub.edge_type.tolist() == [g.edge_type[0], g.edge_type[1], g.edge_type[3]]
    assert sub.edge_time.tolist() == [1.0, 2.0, 4.0]
    assert sub.edge_attr[:, 0].tolist() == [1.0, 2.0, 4.0]


def test_temporal_split_and_before_keep_relations():
    g = Graph.from_pandas(rel_df(), source="s", target="t", relation="rel", time="ts")
    g.node_time = np.array([0.0, 0.0, 1.0, 1.0])
    train, test = temporal_split(g, cutoff=0.5)
    assert train.relation_names == g.relation_names and train.edge_type is not None
    assert g.before(3.0).relation_names == g.relation_names


def fake_mat(path, keys, n=6):
    rng = np.random.default_rng(0)
    rels = {}
    for k in keys:
        a = sp.random(n, n, density=0.3, format="csr", random_state=len(rels))
        rels[k] = sp.csr_matrix((a + a.T).sign())
    homo = sp.csr_matrix(sum(rels.values()).sign()) if rels else sp.csr_matrix((n, n))
    scipy.io.savemat(
        path,
        {
            "homo": homo,
            "features": sp.csr_matrix(rng.normal(size=(n, 3))),
            "label": (rng.random(n) < 0.3).astype(np.int64),
            **rels,
        },
    )
    return rels, homo


def test_yelpchi_relations_opt_in(tmp_path):
    rels, homo = fake_mat(tmp_path / "YelpChi.mat", ["net_rur", "net_rsr", "net_rtr"])
    plain = load_yelpchi(root=tmp_path)
    assert plain.edge_type is None and plain.relation_names == []
    assert plain.n_edges == homo.nnz  # default edge list unchanged: the union's

    g = load_yelpchi(root=tmp_path, relations=True)
    assert g.relation_names == ["R-U-R", "R-S-R", "R-T-R"]
    assert g.n_edges == sum(r.nnz for r in rels.values())
    for code, key in enumerate(["net_rur", "net_rsr", "net_rtr"]):
        ei = g.edge_index[:, g.edge_type == code]
        rebuilt = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=homo.shape)
        assert (rebuilt != rels[key]).nnz == 0
    assert (g.adj != plain.adj).nnz == 0  # adj stays the binarised union


def test_amazon_relations_and_missing_keys(tmp_path):
    fake_mat(tmp_path / "Amazon.mat", ["net_upu", "net_usu"])  # net_uvu missing
    with pytest.raises(ValueError, match="net_uvu"):
        load_amazon(root=tmp_path, relations=True)


# ------------------------------------------------- validation (review findings)


def test_per_edge_arrays_need_explicit_edge_index():
    """adj's own edge order is not the caller's: per-edge arrays without edge_index
    would silently attach to the wrong edges."""
    adj = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    for k in ("edge_type", "edge_time", "edge_attr", "edge_labels"):
        with pytest.raises(ValueError, match=f"{k} given without edge_index"):
            Graph(adj=adj, **{k: np.zeros(6)})
    ei = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="edge_time has 3 rows for 2 edges"):
        Graph(adj=adj, edge_index=ei, edge_time=np.zeros(3))
    assert Graph(adj=sp.csr_matrix((3, 3)), edge_type=np.zeros(0)).n_edges == 0


def test_relation_names_normalized_and_must_cover_codes():
    adj = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    ei = np.array([[0, 1], [1, 2]])
    for names in (np.array(["a", "b"]), pd.Index(["a", "b"]), ("a", "b")):
        g = Graph(adj=adj, edge_index=ei, edge_type=[0, 1], relation_names=names)
        assert g.relation_names == ["a", "b"] and isinstance(g.relation_names, list)
    with pytest.raises(ValueError, match="code 1 has no name"):
        Graph(adj=adj, edge_index=ei, edge_type=[0, 1], relation_names=["a"])


def test_subgraph_rejects_repeated_nodes():
    g = Graph.from_pandas(rel_df(), source="s", target="t", relation="rel")
    with pytest.raises(ValueError, match="unique"):
        g.subgraph(np.array([0, 1, 1]))


def test_before_keeps_node_time():
    g = Graph.from_pandas(rel_df(), source="s", target="t", time="ts")
    g.node_time = np.arange(g.n_nodes, dtype=float)
    assert np.array_equal(g.before(3.0).node_time, g.node_time)


class _Tensor:
    """Just enough of a torch tensor for as_graph's PyG branch, without torch."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Data:
    def __init__(self, edge_type):
        self.edge_index = _Tensor([[0, 1, 2], [1, 2, 0]])
        self.num_nodes = 3
        self.x = None
        self.edge_type = edge_type


def test_pyg_edge_type_is_lenient():
    from graphspot.graph import as_graph

    assert as_graph(_Data(_Tensor([0, 1, 0]))).edge_type.tolist() == [0, 1, 0]
    assert as_graph(_Data(_Tensor([[0], [1], [1]]))).edge_type.tolist() == [0, 1, 1]
    assert as_graph(_Data([1, 0, 1])).edge_type.tolist() == [1, 0, 1]  # not a tensor
    assert as_graph(_Data(None)).edge_type is None
    with pytest.warns(UserWarning, match="ignoring edge_type"):
        g = as_graph(_Data(_Tensor(np.eye(3))))  # one-hot: not one code per edge
    assert g.edge_type is None and g.n_edges == 3


def test_labels_must_be_binary():
    from graphspot.base import BaseDetector

    g = Graph(adj=sp.csr_matrix(np.ones((3, 3)) - np.eye(3)))
    with pytest.raises(ValueError, match=r"0 \(normal\), 1 \(anomaly\) or -1"):
        BaseDetector._validate_labels(g, np.array([0, 2, -1]), "node")
