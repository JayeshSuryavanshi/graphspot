from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

pyod = pytest.importorskip("pyod")

from graphspot import Graph  # noqa: E402
from graphspot.compat import from_pyod  # noqa: E402


def outlier_graph(n_normal=300, n_anom=15, seed=0):
    """Scattered feature outliers on a sparse graph. NeighborAggregation keeps the
    self features, so an unsupervised tabular detector still sees the outliers
    through the adapter; smoothing-only features would hide them (neighbor means
    of noise shrink toward zero, which makes dense structures look MORE normal to
    an isolation forest, not less)."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_anom
    adj = sp.random(n, n, density=0.01, format="csr", random_state=seed)
    x = rng.normal(size=(n, 6))
    x[n_normal:] = rng.uniform(-8.0, 8.0, size=(n_anom, 6))
    y = np.r_[np.zeros(n_normal, dtype=np.int64), np.ones(n_anom, dtype=np.int64)]
    return Graph(adj=sp.csr_matrix(adj.sign()), x=x), y


def test_pyod_iforest_graph_aware():
    from pyod.models.iforest import IForest
    from sklearn.metrics import roc_auc_score

    g, y = outlier_graph()
    det = from_pyod(IForest(random_state=0)).fit(g)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert roc_auc_score(y, det.decision_scores_) > 0.8


def test_pyod_adapter_inductive_and_pure():
    from pyod.models.iforest import IForest

    g_a, _ = outlier_graph(seed=0)
    g_b, y_b = outlier_graph(seed=7)
    det = from_pyod(IForest(random_state=0)).fit(g_a)
    before = pickle.dumps(det)
    scores = det.decision_function(g_b)
    assert pickle.dumps(det) == before
    assert scores[y_b == 1].mean() > scores[y_b == 0].mean()


def test_pyod_adapter_raw_features_mode():
    from pyod.models.iforest import IForest

    g, _ = outlier_graph()
    det = from_pyod(IForest(random_state=0), use_graph=False).fit(g)
    assert det.decision_scores_.shape == (g.n_nodes,)


def test_estimator_required():
    with pytest.raises(ValueError, match="estimator"):
        from_pyod(None)
