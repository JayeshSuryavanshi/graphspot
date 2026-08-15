from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from graphspot import Graph, temporal_split
from graphspot.detectors import OddBall
from graphspot.detectors.oddball import _egonet_features
from graphspot.metrics import base_rate_sweep, evaluate_temporal


def test_egonet_features_hand_computed():
    # triangle 0-1-2 plus pendant 3 attached to 0
    adj = sp.csr_matrix(
        np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
    )
    feats = _egonet_features(Graph(adj=adj))
    # node 0: neighbors {1,2,3}, egonet edges = deg(3) + edge(1,2) = 4
    assert feats[0].tolist() == [3.0, 4.0]
    # node 1: neighbors {0,2}, egonet edges = 2 + edge(0,2) = 3
    assert feats[1].tolist() == [2.0, 3.0]
    assert feats[3].tolist() == [1.0, 1.0]


def clique_in_sparse(n_sparse=400, clique=25, seed=0):
    n = n_sparse + clique
    adj = sp.random(n, n, density=0.004, format="lil", random_state=seed)
    for i in range(n_sparse, n):
        for j in range(n_sparse, n):
            if i != j:
                adj[i, j] = 1.0
    g = Graph(adj=sp.csr_matrix((adj + adj.T).sign()))
    y = np.zeros(n, dtype=np.int64)
    y[n_sparse:] = 1
    return g, y


def test_near_clique_scores_high():
    g, y = clique_in_sparse()
    det = OddBall(contamination=0.06).fit(g)
    assert det.decision_scores_[y == 1].min() > np.median(det.decision_scores_[y == 0])
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, det.decision_scores_) > 0.95


def test_oddball_is_inductive_and_pure():
    g_a, _ = clique_in_sparse(seed=0)
    g_b, y_b = clique_in_sparse(seed=9)
    det = OddBall().fit(g_a)
    before = pickle.dumps(det)
    scores = det.decision_function(g_b)
    assert pickle.dumps(det) == before
    assert scores[y_b == 1].mean() > scores[y_b == 0].mean()


def test_oddball_rejects_tiny_graph():
    with pytest.raises(ValueError, match="power law"):
        OddBall().fit(Graph(adj=sp.csr_matrix((5, 5))))


def timed_graph(steps=6, per_step=40, seed=0):
    rng = np.random.default_rng(seed)
    n = steps * per_step
    node_time = np.repeat(np.arange(steps, dtype=float), per_step)
    src, dst = [], []
    for t in range(steps):
        base = t * per_step
        for _ in range(per_step * 2):
            src.append(base + rng.integers(0, per_step))
            dst.append(base + rng.integers(0, per_step))
    adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    x = rng.normal(size=(n, 4))
    y = (rng.random(n) < 0.2).astype(np.int64)
    return Graph(adj=adj, x=x, node_labels=y, node_time=node_time)


def test_temporal_split_is_disjoint_and_carries_time():
    g = timed_graph()
    train, test = temporal_split(g, cutoff=3)
    assert train.n_nodes == 4 * 40
    assert test.n_nodes == 2 * 40
    assert train.node_time.max() <= 3
    assert test.node_time.min() > 3
    assert train.node_labels is not None and test.x is not None


def test_temporal_split_requires_node_time():
    g = Graph(adj=sp.eye(4, format="csr"))
    with pytest.raises(ValueError, match="node_time"):
        temporal_split(g, 1.0)


def test_evaluate_temporal_per_step():
    rng = np.random.default_rng(0)
    times = np.repeat([1.0, 2.0, 3.0], 100)
    y = (rng.random(300) < 0.3).astype(np.int64)
    scores = y * 0.8 + rng.random(300) * 0.4
    out = evaluate_temporal(y, scores, times)
    assert len(out["steps"]) == 3
    assert out["n_scored_steps"] == 3
    assert 0.5 < out["mean_auroc"] <= 1.0
    empty_step = evaluate_temporal(np.zeros(50, dtype=int), rng.random(50), np.full(50, 7.0))
    assert np.isnan(empty_step["mean_auprc"])


def test_base_rate_sweep_degrades_gracefully():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(np.int64)
    scores = y * 1.0 + rng.normal(0, 0.5, 5000)
    rows = base_rate_sweep(y, scores, rates=(0.05, 0.01), seed=1)
    assert [r["base_rate"] for r in rows] == [0.05, 0.01]
    assert rows[0]["auprc"] > rows[1]["auprc"] > 0.05
    again = base_rate_sweep(y, scores, rates=(0.05, 0.01), seed=1)
    assert rows[0]["auprc"] == again[0]["auprc"]
