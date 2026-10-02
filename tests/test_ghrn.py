from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

import sys  # noqa: E402

if sys.platform == "darwin" and "xgboost" in sys.modules:
    pytest.skip(
        "xgboost already loaded in this process; the dual-OpenMP conflict on macOS "
        "makes mixing unsafe. Run tests/test_ghrn.py in its own process.",
        allow_module_level=True,
    )

from graphspot import Graph  # noqa: E402
from graphspot.detectors import GHRN  # noqa: E402
from graphspot.detectors.bwgnn import _simple_undirected  # noqa: E402


def hetero_graph(n_normal=200, n_anom=20, seed=0):
    """Homophilous normal block, anomalies wired mostly to normals (heterophily),
    with weakly shifted features so the encoder has something to learn from."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_anom
    rows, cols = [], []
    for i in range(n_normal):
        for j in rng.choice(n_normal, size=4, replace=False):
            rows.append(i)
            cols.append(j)
    for a in range(n_normal, n):
        for j in rng.choice(n_normal, size=6, replace=False):
            rows.append(a)
            cols.append(j)
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = sp.csr_matrix((adj + adj.T).sign())
    x = rng.normal(size=(n, 8))
    x[n_normal:] += 1.5
    y = np.zeros(n, dtype=np.int64)
    y[n_normal:] = 1
    return Graph(adj=adj, x=x), y


def test_fit_contract_and_separation():
    from sklearn.metrics import roc_auc_score

    g, y = hetero_graph()
    det = GHRN(epochs=60, prune_ratio=0.05, random_state=0).fit(g, y)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert roc_auc_score(y, det.decision_scores_) > 0.9
    assert det.predict_proba().shape == (g.n_nodes, 2)
    n_undirected = _simple_undirected(g.adj).nnz // 2
    assert det.n_edges_pruned_ == int(np.floor(0.05 * n_undirected))
    assert len(det.stages_) == 1


def test_heterophily_score_targets_inter_class_edges():
    """With the true label signal, the lowest-scored edges are inter-class far above
    their base rate. This pins the high-pass edge score, independent of training."""
    g, y = hetero_graph()
    det = GHRN()
    adj = _simple_undirected(g.adj)
    upper = sp.triu(adj, k=1).tocoo()
    lap = det._laplacian_from_adj(adj)
    scores = det.edge_heterophily(lap, y.astype(float), y, upper.row, upper.col)
    inter = y[upper.row] != y[upper.col]
    lowest = np.argsort(scores, kind="stable")[: inter.sum()]
    assert inter[lowest].mean() > 0.9
    assert inter.mean() < 0.15


def test_zero_prune_ratio_prunes_nothing():
    g, y = hetero_graph(n_normal=60, n_anom=10)
    det = GHRN(epochs=5, prune_ratio=0.0, random_state=0).fit(g, y)
    assert det.n_edges_pruned_ == 0


def test_multiple_rounds():
    g, y = hetero_graph(n_normal=60, n_anom=10)
    det = GHRN(epochs=5, prune_ratio=0.05, rounds=2, random_state=0).fit(g, y)
    assert len(det.stages_) == 2
    assert det.decision_function(g).shape == (g.n_nodes,)


def test_inductive_on_unseen_graph():
    g_a, y_a = hetero_graph(seed=0)
    g_b, y_b = hetero_graph(seed=7)
    det = GHRN(epochs=60, prune_ratio=0.05, random_state=0).fit(g_a, y_a)
    scores = det.decision_function(g_b)
    assert scores.shape == (g_b.n_nodes,)
    assert scores[y_b == 1].mean() > scores[y_b == 0].mean()


def test_score_is_pure():
    g, y = hetero_graph()
    det = GHRN(epochs=20, prune_ratio=0.05, random_state=0).fit(g, y)
    before = pickle.dumps(det)
    s1 = det.decision_function(g)
    assert pickle.dumps(det) == before
    assert np.array_equal(s1, det.decision_function(g))


def test_seed_reproducibility():
    g, y = hetero_graph()
    a = GHRN(epochs=20, random_state=5).fit(g, y).decision_scores_
    b = GHRN(epochs=20, random_state=5).fit(g, y).decision_scores_
    assert np.array_equal(a, b)


def test_params_roundtrip():
    det = GHRN(prune_ratio=0.1, rounds=2, order=3)
    params = det.get_params()
    assert params["prune_ratio"] == 0.1 and params["rounds"] == 2 and params["order"] == 3
    assert det.set_params(prune_ratio=0.2).prune_ratio == 0.2


def test_rejects_bad_params_and_missing_features():
    with pytest.raises(ValueError, match="prune_ratio"):
        GHRN(prune_ratio=1.0)
    with pytest.raises(ValueError, match="rounds"):
        GHRN(rounds=0)
    g, y = hetero_graph(n_normal=30, n_anom=10)
    with pytest.raises(ValueError, match="node features"):
        GHRN(epochs=1).fit(Graph(adj=g.adj), y)
