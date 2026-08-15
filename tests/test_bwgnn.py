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
        "makes mixing unsafe. Run tests/test_bwgnn.py in its own process.",
        allow_module_level=True,
    )

from graphspot import Graph  # noqa: E402
from graphspot.detectors import BWGNN  # noqa: E402


def test_filter_bank_sums_to_identity_scaled():
    """The Beta wavelet constants make the whole bank sum to (order+1)/2 * I exactly,
    a closed-form check on the clean-room filter implementation."""
    rng = np.random.default_rng(0)
    adj = sp.random(30, 30, density=0.15, format="csr", random_state=0)
    adj = sp.csr_matrix((adj + adj.T).sign())
    g = Graph(adj=adj, x=np.ones((30, 1)))
    for order in (1, 2, 3):
        det = BWGNN(order=order)
        lap = det._laplacian(g)
        h = torch.tensor(rng.normal(size=(30, 4)), dtype=torch.float32)
        total = sum(det._filter_bank(lap, h))
        expected = (order + 1) / 2.0 * h
        assert torch.allclose(total, expected, atol=1e-4), f"order {order}"


def blob_graph(n_normal=300, n_anom=40, seed=0):
    """Anomalies form a dense cluster and carry a shifted feature distribution, so
    both the spectral and feature paths have something to learn."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_anom
    adj = sp.random(n, n, density=0.02, format="lil", random_state=seed)
    for i in range(n_normal, n):
        for j in range(n_normal, n):
            if i != j and rng.random() < 0.5:
                adj[i, j] = 1.0
    adj = sp.csr_matrix((adj + adj.T).sign())
    x = rng.normal(size=(n, 8))
    x[n_normal:] += 1.5
    y = np.zeros(n, dtype=np.int64)
    y[n_normal:] = 1
    return Graph(adj=adj, x=x), y


def test_fit_contract_and_separation():
    from sklearn.metrics import roc_auc_score

    g, y = blob_graph()
    det = BWGNN(epochs=60, random_state=0).fit(g, y)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert roc_auc_score(y, det.decision_scores_) > 0.9
    assert det.predict_proba().shape == (g.n_nodes, 2)


def test_inductive_on_unseen_graph():
    g_a, y_a = blob_graph(seed=0)
    g_b, y_b = blob_graph(seed=7)
    det = BWGNN(epochs=60, random_state=0).fit(g_a, y_a)
    scores = det.decision_function(g_b)
    assert scores.shape == (g_b.n_nodes,)
    assert scores[y_b == 1].mean() > scores[y_b == 0].mean()


def test_score_is_pure():
    g, y = blob_graph()
    det = BWGNN(epochs=20, random_state=0).fit(g, y)
    before = pickle.dumps(det)
    s1 = det.decision_function(g)
    assert pickle.dumps(det) == before
    assert np.array_equal(s1, det.decision_function(g))


def test_seed_reproducibility():
    g, y = blob_graph()
    a = BWGNN(epochs=20, random_state=5).fit(g, y).decision_scores_
    b = BWGNN(epochs=20, random_state=5).fit(g, y).decision_scores_
    assert np.array_equal(a, b)


def test_semi_supervised_labels():
    g, y = blob_graph()
    y_semi = y.copy()
    y_semi[np.random.default_rng(0).random(len(y)) < 0.5] = -1
    det = BWGNN(epochs=20, random_state=0).fit(g, y_semi)
    assert det.decision_scores_.shape == (g.n_nodes,)


def test_rejects_bad_params_and_missing_features():
    with pytest.raises(ValueError, match="order"):
        BWGNN(order=0)
    g, y = blob_graph(n_normal=30, n_anom=10)
    with pytest.raises(ValueError, match="node features"):
        BWGNN(epochs=1).fit(Graph(adj=g.adj), y)


def test_darwin_omp_guard(monkeypatch):
    """Both guards must fire before any native library loads, so this is safe to
    test anywhere by faking the platform and the presence of the other library."""
    import sys

    from graphspot.detectors.bwgnn import _require_torch
    from graphspot.detectors.trees import _guard_darwin_omp

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "xgboost", object())
    with pytest.raises(RuntimeError, match="OpenMP"):
        _require_torch()
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="OpenMP"):
        _guard_darwin_omp()
