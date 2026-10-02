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
    with pytest.raises(ValueError, match="edge_type"):
        Graph(adj=adj, edge_type=np.zeros(2, dtype=np.int64))


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
