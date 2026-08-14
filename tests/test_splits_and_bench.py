from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from graphspot import Graph, semi_supervised_split, stratified_split, train_labels
from graphspot.bench import bench_graph, format_table


def labels(n=1000, contamination=0.1, unlabeled=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < contamination).astype(np.int64)
    y[rng.random(n) < unlabeled] = -1
    return y


def test_stratified_split_partitions_labeled_nodes():
    y = labels()
    tr, va, te = stratified_split(y, seed=3)
    combined = np.concatenate([tr, va, te])
    assert len(combined) == len(set(combined.tolist())) == int((y >= 0).sum())
    assert (y[combined] >= 0).all()
    for part in (tr, va, te):
        assert 0 < y[part].mean() < 1  # both classes present

    tr2, va2, te2 = stratified_split(y, seed=3)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2)
    assert not np.array_equal(tr, stratified_split(y, seed=4)[0])


def test_stratified_ratios():
    y = labels(n=5000, unlabeled=0.0)
    tr, va, te = stratified_split(y, train_size=0.7, val_size=0.15, seed=0)
    assert len(tr) / len(y) == pytest.approx(0.7, abs=0.01)
    assert len(va) / len(y) == pytest.approx(0.15, abs=0.01)
    assert y[tr].mean() == pytest.approx(y.mean(), abs=0.02)


def test_semi_supervised_split_counts():
    y = labels(n=3000, contamination=0.1, unlabeled=0.1)
    tr, va, te = semi_supervised_split(y, seed=1)
    assert len(tr) == 100
    assert (y[tr] == 1).sum() == 20
    assert (y[tr] == 0).sum() == 80
    assert set(tr) & set(va) == set() and set(tr) & set(te) == set()
    with pytest.raises(ValueError, match="need"):
        semi_supervised_split(np.array([1, 0, 0, 1]))


def test_train_labels_masks_everything_else():
    y = np.array([0, 1, 0, 1, -1])
    out = train_labels(y, np.array([0, 3]))
    assert out.tolist() == [0, -1, -1, 1, -1]


def ring_graph(n_normal=250, ring=25, seed=0):
    rng = np.random.default_rng(seed)
    n = n_normal + ring
    adj = sp.random(n, n, density=0.01, format="lil", random_state=seed)
    members = np.arange(n_normal, n)
    for i in members:
        for j in members:
            if i != j and rng.random() < 0.8:
                adj[i, j] = 1.0
    adj = sp.csr_matrix((adj + adj.T).sign())
    x = rng.normal(size=(n, 6))
    y = np.zeros(n, dtype=np.int64)
    y[members] = 1
    return Graph(adj=adj, x=x, node_labels=y)


def test_bench_graph_rows_and_table():
    g = ring_graph()
    rows = bench_graph("ring", g, seeds=(0, 1))
    assert {r["detector"] for r in rows} == {"XGBGraph", "RFGraph", "FlatBaseline"}
    assert all(r["trials"] == 2 and 0 <= r["auprc"] <= 100 for r in rows)
    table = format_table(rows)
    assert "ring" in table and "XGBGraph" in table and "*" in table


def test_bench_uses_shipped_masks_when_present():
    g = ring_graph()
    n = g.n_nodes
    rng = np.random.default_rng(0)
    train = rng.random((2, n)) < 0.5
    test = ~train
    g.split_masks = {"train": train, "val": np.zeros_like(train), "test": test}
    rows = bench_graph("ring", g, seeds=(0, 1))
    assert rows[0]["trials"] == 2


def test_bench_semi_regime():
    g = ring_graph(n_normal=900, ring=40)
    rows = bench_graph("ring", g, seeds=(0,), regime="semi")
    assert all(r["regime"] == "semi" for r in rows)
