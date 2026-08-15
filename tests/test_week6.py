from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import graphspot
from graphspot import Graph
from graphspot.detectors import FlatUnsupervised, Fraudar, OddBall


def test_list_detectors_shape_and_availability():
    rows = graphspot.list_detectors()
    names = {r["name"] for r in rows}
    assert {"XGBGraph", "Fraudar", "OddBall", "BWGNN", "FlatUnsupervised"} <= names
    for r in rows:
        assert set(r) >= {"name", "levels", "supervised", "requires", "available", "note"}
    core = [r for r in rows if not r["requires"]]
    assert all(r["available"] for r in core)


def blob_graph(seed=0, n=300, n_anom=15):
    """Scattered feature outliers: LOF is blind to clustered anomalies by design
    (a tight anomalous cluster is locally dense), so both LOF and IForest need
    the anomalies spread out to be a fair unsupervised fixture."""
    rng = np.random.default_rng(seed)
    adj = sp.random(n + n_anom, n + n_anom, density=0.01, format="csr", random_state=seed)
    x = rng.normal(size=(n + n_anom, 6))
    x[n:] = rng.uniform(-8.0, 8.0, size=(n_anom, 6))
    y = np.r_[np.zeros(n, dtype=np.int64), np.ones(n_anom, dtype=np.int64)]
    return Graph(adj=sp.csr_matrix(adj.sign()), x=x), y


@pytest.mark.parametrize("estimator", ["iforest", "lof"])
def test_flat_unsupervised_contract(estimator):
    from sklearn.metrics import roc_auc_score

    g, y = blob_graph()
    det = FlatUnsupervised(estimator=estimator, random_state=0).fit(g)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert roc_auc_score(y, det.decision_scores_) > 0.9
    g2, y2 = blob_graph(seed=5)
    scores = det.decision_function(g2)
    assert scores[y2 == 1].mean() > scores[y2 == 0].mean()


def test_flat_unsupervised_rejects_bad_estimator():
    with pytest.raises(ValueError, match="iforest"):
        FlatUnsupervised(estimator="ocsvm")


def test_oddball_explain_reads_as_law_violation():
    adj = sp.lil_matrix((60, 60))
    for i in range(50):
        adj[i, (i + 1) % 50] = adj[(i + 1) % 50, i] = 1.0
    for i in range(50, 60):
        for j in range(50, 60):
            if i != j:
                adj[i, j] = 1.0
    det = OddBall(contamination=0.15).fit(Graph(adj=sp.csr_matrix(adj)))
    top = det.explain(k=3)
    assert len(top) == 3
    assert all(r["ratio"] > 1.0 for r in top)
    one = det.explain(idx=top[0]["node"])
    assert one["egonet_edges"] > one["expected_edges"]


def test_fraudar_explain_blocks_and_membership():
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = [(f"b{rng.integers(0, 2000)}", f"s{rng.integers(0, 800)}", 0) for _ in range(1500)]
    ring_edges = [(f"rb{i}", f"rs{j}", 1) for i in range(30) for j in range(5)]
    df = pd.DataFrame(rows + ring_edges, columns=["b", "s", "ring"])
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    g = Graph.from_pandas(df, source="b", target="s")
    det = Fraudar().fit(g)

    blocks = det.explain()
    assert blocks and blocks[0]["n_edges"] > 0
    ring_idx = int(np.flatnonzero(df["ring"].to_numpy() == 1)[0])
    member = det.explain(idx=ring_idx)
    assert member is not None and member["density"] > 0
    normal_idx = int(np.flatnonzero(det.decision_scores_ == 0)[0])
    assert det.explain(idx=normal_idx) is None
