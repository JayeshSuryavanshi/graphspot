from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from graphspot import Graph
from graphspot.detectors import FlatBaseline, RFGraph, XGBGraph

ALL = [XGBGraph, RFGraph, FlatBaseline]


def ring_graph(n_normal=300, ring=20, seed=0):
    """Sparse random background plus a dense collusive ring whose members look
    individually normal: features carry no signal, only structure does."""
    rng = np.random.default_rng(seed)
    n = n_normal + ring
    adj = sp.random(n, n, density=0.01, format="lil", random_state=seed)
    members = np.arange(n_normal, n)
    for i in members:
        for j in members:
            if i != j and rng.random() < 0.8:
                adj[i, j] = 1.0
    adj = sp.csr_matrix((adj + adj.T).sign())
    x = rng.normal(size=(n, 8))
    y = np.zeros(n, dtype=np.int64)
    y[members] = 1
    return Graph(adj=adj, x=x), y


@pytest.mark.parametrize("cls", ALL)
def test_fit_predict_contract(cls):
    g, y = ring_graph()
    det = cls(random_state=0, contamination=0.05).fit(g, y)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert det.labels_.dtype == np.int64
    assert set(np.unique(det.labels_)) <= {0, 1}
    assert det.predict().shape == (g.n_nodes,)
    proba = det.predict_proba()
    assert proba.shape == (g.n_nodes, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", [XGBGraph, RFGraph])
def test_inductive_scoring_on_unseen_graph(cls):
    train_g, train_y = ring_graph(seed=0)
    test_g, test_y = ring_graph(seed=1)
    det = cls(random_state=0).fit(train_g, train_y)
    scores = det.decision_function(test_g)
    assert scores.shape == (test_g.n_nodes,)
    ring_mean = scores[test_y == 1].mean()
    normal_mean = scores[test_y == 0].mean()
    assert ring_mean > normal_mean, "ring members should score higher on an unseen graph"


def test_graph_beats_flat_when_only_structure_matters():
    from sklearn.metrics import average_precision_score

    train_g, train_y = ring_graph(seed=0)
    test_g, test_y = ring_graph(seed=1)
    graph_ap = average_precision_score(
        test_y, XGBGraph(random_state=0).fit(train_g, train_y).decision_function(test_g)
    )
    flat_ap = average_precision_score(
        test_y, FlatBaseline(random_state=0).fit(train_g, train_y).decision_function(test_g)
    )
    assert graph_ap > flat_ap + 0.3


@pytest.mark.parametrize("cls", ALL)
def test_score_does_not_mutate(cls):
    g, y = ring_graph()
    det = cls(random_state=0).fit(g, y)
    before = pickle.dumps(det)
    first = det.decision_function(g)
    assert pickle.dumps(det) == before
    assert np.array_equal(first, det.decision_function(g))


@pytest.mark.parametrize("cls", ALL)
def test_get_set_params_roundtrip(cls):
    det = cls(random_state=7, contamination=0.02)
    params = det.get_params()
    assert params["random_state"] == 7
    clone = cls(**params)
    assert clone.get_params() == params
    det.set_params(contamination=0.1)
    assert det.contamination == 0.1
    with pytest.raises(ValueError, match="Invalid parameter"):
        det.set_params(nonsense=1)


def test_unlabeled_minus_one_excluded():
    g, y = ring_graph()
    y_semi = y.copy()
    rng = np.random.default_rng(0)
    hide = rng.random(len(y)) < 0.5
    y_semi[hide] = -1
    det = XGBGraph(random_state=0).fit(g, y_semi)
    assert det.decision_scores_.shape == (g.n_nodes,)


def test_supervised_requires_labels():
    g, _ = ring_graph()
    with pytest.raises(ValueError, match="supervised"):
        XGBGraph().fit(g)


def test_unsupported_level_rejected_at_construction():
    with pytest.raises(ValueError, match="supports levels"):
        XGBGraph(level="edge")


def test_explain_names_map_to_hops():
    g, y = ring_graph()
    det = XGBGraph(random_state=0).fit(g, y)
    top = det.explain(k=5)
    assert len(top) == 5
    assert all(isinstance(name, str) and imp >= 0 for name, imp in top)
