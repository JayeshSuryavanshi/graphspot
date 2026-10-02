from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

import sys  # noqa: E402

if sys.platform == "darwin" and "xgboost" in sys.modules:
    pytest.skip(
        "xgboost already loaded in this process; the dual-OpenMP conflict on macOS "
        "makes mixing unsafe. Run tests/test_care_gnn.py in its own process.",
        allow_module_level=True,
    )

from graphspot import Graph  # noqa: E402
from graphspot.detectors import CAREGNN  # noqa: E402
from graphspot.detectors.bwgnn import _simple_undirected  # noqa: E402
from graphspot.splits import temporal_split  # noqa: E402


def camo_graph(n_normal=400, n_anom=40, seed=0):
    """Two relations. 'homo': each node links to 3 same-class nodes. 'camo': each
    node links to 5 random normal nodes, so anomalies hide among normals there.
    Anomalies are shifted +1 on 4 of 8 features."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_anom
    y = np.zeros(n, dtype=np.int64)
    y[n_normal:] = 1
    src, dst, rel = [], [], []
    for v in range(n):
        same = np.flatnonzero(y == y[v])
        for u in rng.choice(same[same != v], size=3, replace=False):
            src.append(v), dst.append(u), rel.append("homo")
        for u in rng.choice(n_normal, size=5, replace=False):
            if u != v:
                src.append(v), dst.append(u), rel.append("camo")
    x = rng.normal(size=(n, 8))
    x[n_normal:, :4] += 1.0
    df = pd.DataFrame({"s": src, "d": dst, "rel": rel})
    feats = pd.DataFrame(x, index=pd.RangeIndex(n))
    g = Graph.from_pandas(
        df, source="s", target="d", relation="rel", directed=False, node_features=feats
    )
    order = g.node_index.to_numpy()  # from_pandas orders nodes by first appearance
    return g, y[order]


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s)


# ------------------------------------------------------------- pure helpers


def test_select_keeps_ceil_p_nearest_per_row():
    rng = np.random.default_rng(0)
    adj = sp.random(60, 60, density=0.15, format="csr", random_state=1)
    adj = _simple_undirected(adj)
    p_fraud = rng.random(60)
    x = rng.normal(size=(60, 3))
    rows = np.arange(60)
    for keep_p in (0.3, 0.5, 1.0):
        agg, kept = CAREGNN._select_mean(adj, rows, p_fraud, keep_p, x)
        for v in rows:
            nbrs = adj.indices[adj.indptr[v] : adj.indptr[v + 1]]
            if not len(nbrs):
                assert np.allclose(agg[v], 0.0)
                continue
            k = int(np.ceil(keep_p * len(nbrs) - 1e-9))
            dist = np.abs(p_fraud[v] - p_fraud[nbrs])
            chosen = nbrs[np.lexsort((nbrs, dist))[:k]]
            assert np.allclose(agg[v], x[chosen].mean(axis=0))
        assert len(kept) == sum(int(np.ceil(keep_p * d - 1e-9)) for d in np.diff(adj.indptr))
    full, _ = CAREGNN._select_mean(adj, rows, p_fraud, 1.0, x)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    mean = np.divide(adj @ x, deg[:, None], out=np.zeros_like(x), where=deg[:, None] > 0)
    assert np.allclose(full, mean)


def test_select_filters_camouflage_with_oracle_predictor():
    """With a perfect label predictor, top-p keeps an inter-class edge only when a
    center has fewer same-class neighbors than it keeps. Anomalies have only normal
    neighbors in 'camo', so the exact kept inter-class count is pinned, and it is
    well below the relation's base rate. Pins the selector without training."""
    g, y = camo_graph()
    camo = g.relation_names.index("camo")
    ei = g.edge_index[:, g.edge_type == camo]
    n = len(y)
    adj = _simple_undirected(sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), (n, n)))
    _, kept = CAREGNN._select_mean(adj, np.arange(n), y.astype(float), 0.5, np.zeros((n, 1)))
    expected = 0
    for v in range(n):
        nbrs = adj.indices[adj.indptr[v] : adj.indptr[v + 1]]
        k = int(np.ceil(0.5 * len(nbrs) - 1e-9))
        expected += max(0, k - int((y[nbrs] == y[v]).sum()))
    assert int((kept > 0).sum()) == expected
    upper = sp.triu(adj, k=1).tocoo()
    assert (kept > 0).mean() < 0.6 * (y[upper.row] != y[upper.col]).mean()


def test_bandit_rule_and_freeze():
    det = CAREGNN(rl_step=0.02)
    det.rl_stopped_epoch_ = np.full(1, -1)
    steps, frozen, rewards = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=bool), [[]]
    det._bandit_step(steps, frozen, rewards, np.array([0.5]), np.array([0.4]), 0)
    assert steps[0] == 1 and rewards[0] == [1]  # distance fell: keep more
    det._bandit_step(steps, frozen, rewards, np.array([0.4]), np.array([0.4]), 1)
    assert steps[0] == 2  # tie counts as not rising
    det._bandit_step(steps, frozen, rewards, np.array([0.4]), np.array([0.6]), 2)
    assert steps[0] == 1 and rewards[0][-1] == -1
    det._bandit_step(steps, frozen, rewards, np.array([np.nan]), np.array([0.6]), 3)
    assert len(rewards[0]) == 3  # undefined distance: no reward
    for e in range(4, 30):
        g_prev, g_cur = (0.5, 0.4) if e % 2 else (0.4, 0.5)
        det._bandit_step(steps, frozen, rewards, np.array([g_prev]), np.array([g_cur]), e)
        if frozen[0]:
            break
    assert frozen[0] and det.rl_stopped_epoch_[0] == e
    before = steps.copy()
    det._bandit_step(steps, frozen, rewards, np.array([0.9]), np.array([0.1]), e + 1)
    assert np.array_equal(steps, before)


def test_thresholds_are_exact_and_clipped():
    det = CAREGNN(rl_step=0.02)
    assert det._thresholds(np.array([5]))[0] == 0.6  # no 0.6000000000000001 drift
    assert det._thresholds(np.array([-100, 100])).tolist() == [0.0, 1.0]


# ------------------------------------------------------------------ fitting


def test_fit_contract_and_separation():
    g, y = camo_graph()
    det = CAREGNN(epochs=40, random_state=0).fit(g, y)
    assert det.decision_scores_.shape == (g.n_nodes,)
    assert auc(y, det.decision_scores_) > 0.95
    assert det.predict_proba().shape == (g.n_nodes, 2)
    assert det.relation_names_ == ["camo", "homo"]
    assert det.relation_p_.shape == (2,)
    assert ((det.relation_p_ >= 0) & (det.relation_p_ <= 1)).all()
    hist = det.relation_p_history_
    assert hist.shape == (40, 2) and (hist[0] == 0.5).all()
    assert np.isin(np.round(np.diff(hist, axis=0), 9), [-0.02, 0.0, 0.02]).all()


def test_rl_step_zero_keeps_half():
    g, y = camo_graph(n_normal=100, n_anom=20)
    det = CAREGNN(epochs=5, rl_step=0.0, random_state=0).fit(g, y)
    assert (det.relation_p_history_ == 0.5).all() and (det.relation_p_ == 0.5).all()


def test_fit_scores_equal_decision_function():
    g, y = camo_graph()
    det = CAREGNN(epochs=15, random_state=0).fit(g, y)
    assert np.array_equal(det.decision_scores_, det.decision_function(g))


def test_inductive_on_unseen_graph():
    g_a, y_a = camo_graph(seed=0)
    g_b, y_b = camo_graph(seed=7)
    det = CAREGNN(epochs=40, random_state=0).fit(g_a, y_a)
    scores = det.decision_function(g_b)
    assert scores.shape == (g_b.n_nodes,)
    assert auc(y_b, scores) > 0.9


def test_score_is_pure():
    g, y = camo_graph()
    det = CAREGNN(epochs=10, random_state=0).fit(g, y)
    before = pickle.dumps(det)
    s1 = det.decision_function(g)
    assert pickle.dumps(det) == before
    assert np.array_equal(s1, det.decision_function(g))


def test_seed_reproducibility():
    g, y = camo_graph()
    a = CAREGNN(epochs=10, random_state=5).fit(g, y).decision_scores_
    b = CAREGNN(epochs=10, random_state=5).fit(g, y).decision_scores_
    assert np.array_equal(a, b)


def test_semi_supervised_labels():
    """The label-scarce regime CARE-GNN is cited for: 20 positives, 80 negatives."""
    g, y = camo_graph()
    rng = np.random.default_rng(0)
    y_semi = np.full_like(y, -1)
    y_semi[rng.choice(np.flatnonzero(y == 1), 20, replace=False)] = 1
    y_semi[rng.choice(np.flatnonzero(y == 0), 80, replace=False)] = 0
    det = CAREGNN(epochs=40, random_state=0).fit(g, y_semi)
    hidden = y_semi == -1
    assert auc(y[hidden], det.decision_scores_[hidden]) > 0.8


def test_minibatches_cover_the_sample():
    g, y = camo_graph(n_normal=100, n_anom=20)
    det = CAREGNN(epochs=5, batch_size=7, random_state=0).fit(g, y)
    assert det.decision_scores_.shape == (g.n_nodes,)


# ---------------------------------------------------------------- relations


def test_relation_alignment_by_name():
    g, y = camo_graph()
    det = CAREGNN(epochs=5, random_state=0).fit(g, y)
    homo = g.edge_type == g.relation_names.index("homo")
    only_homo = Graph(
        adj=g.adj,
        x=g.x,
        edge_index=g.edge_index[:, homo],
        edge_type=np.zeros(homo.sum(), dtype=np.int64),
        relation_names=["homo"],
    )
    rels = det._score_relations(only_homo)
    assert rels[0].nnz == 0 and rels[1].nnz > 0  # mapped to fitted index 1
    assert det.decision_function(only_homo).shape == (g.n_nodes,)

    renamed = Graph(
        adj=g.adj,
        x=g.x,
        edge_index=g.edge_index[:, homo],
        edge_type=np.zeros(homo.sum(), dtype=np.int64),
        relation_names=["nope"],
    )
    with pytest.raises(ValueError, match="relation"):
        det.decision_function(renamed)
    with pytest.raises(ValueError, match="edge_type"):
        det.decision_function(Graph(adj=g.adj, x=g.x))


def test_single_relation_fallback():
    g, y = camo_graph(n_normal=100, n_anom=20)
    plain = Graph(adj=g.adj, x=g.x)
    det = CAREGNN(epochs=5, random_state=0).fit(plain, y)
    assert det.relation_p_.shape == (1,) and det.relation_names_ == []
    assert det.decision_function(g).shape == (g.n_nodes,)  # edge_type ignored: union


def test_temporal_split_keeps_relations():
    g, y = camo_graph()
    g.node_time = np.arange(g.n_nodes, dtype=float) % 10
    g.node_labels = y
    train, test = temporal_split(g, cutoff=6)
    assert train.relation_names == test.relation_names == ["camo", "homo"]
    assert train.edge_type is not None and test.edge_type is not None
    det = CAREGNN(epochs=20, random_state=0).fit(train, train.node_labels)
    assert det.decision_function(test).shape == (test.n_nodes,)


# ------------------------------------------------------------- validation


def test_params_roundtrip_and_bad_inputs():
    det = CAREGNN(hidden=8, rl_step=0.05, sim_weight=1.0)
    params = det.get_params()
    assert params["hidden"] == 8 and params["rl_step"] == 0.05 and params["sim_weight"] == 1.0
    assert det.set_params(batch_size=64).batch_size == 64
    for bad in (
        {"rl_step": 1.0},
        {"rl_step": -0.1},
        {"sim_weight": -1.0},
        {"hidden": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0.0},
        {"weight_decay": -1.0},
    ):
        with pytest.raises(ValueError, match=next(iter(bad))):
            CAREGNN(**bad)

    g, y = camo_graph(n_normal=60, n_anom=10)
    with pytest.raises(ValueError, match="node features"):
        CAREGNN(epochs=1).fit(Graph(adj=g.adj), y)
    det = CAREGNN(epochs=2, random_state=0).fit(g, y)
    with pytest.raises(ValueError, match="features"):
        det.decision_function(Graph(adj=g.adj, x=g.x[:, :3]))
    neg = g.edge_type.copy()
    neg[0] = -1
    bad_g = Graph(adj=g.adj, x=g.x, edge_index=g.edge_index, edge_type=neg)
    with pytest.raises(ValueError, match="negative"):
        CAREGNN(epochs=1).fit(bad_g, y)


# ------------------------------------------------- review regressions


def test_relation_at_zero_can_recover():
    """p_r = 0 used to keep no neighbor, leave the distance undefined and switch the
    bandit off for good. At least one neighbor is kept, so the distance stays
    defined and a +1 reward lifts p_r off the floor."""
    adj = _simple_undirected(sp.random(30, 30, density=0.2, format="csr", random_state=0))
    _, kept = CAREGNN._select_mean(adj, np.arange(30), np.linspace(0, 1, 30), 0.0, np.ones((30, 1)))
    assert len(kept) == int((np.diff(adj.indptr) > 0).sum())  # old rule: 0 kept

    det = CAREGNN(rl_step=0.5)
    det.rl_stopped_epoch_ = np.full(1, -1)
    steps = np.array([-1])  # 0.5 - 0.5 = p_r at the floor
    assert det._thresholds(steps)[0] == 0.0
    g_floor = np.array([kept.mean()])
    det._bandit_step(steps, np.zeros(1, dtype=bool), [[]], g_floor + 0.1, g_floor, 0)
    assert det._thresholds(steps)[0] == 0.5


def test_unnamed_single_relation_fit_matches_codes():
    """Fit with edge_type (all code 0, no names) matches the scoring graph by code;
    only a fit without edge_type uses the union."""
    g, y = camo_graph(n_normal=100, n_anom=20)
    one = Graph(adj=g.adj, x=g.x, edge_index=g.edge_index, edge_type=np.zeros(g.n_edges))
    det = CAREGNN(epochs=3, random_state=0).fit(one, y)
    two = Graph(adj=g.adj, x=g.x, edge_index=g.edge_index, edge_type=g.edge_type)
    with pytest.raises(ValueError, match="fit on 1"):
        det.decision_function(two)


def test_edgeless_graph_with_relation_names():
    g, y = camo_graph(n_normal=60, n_anom=10)
    empty = Graph(
        adj=sp.csr_matrix(g.adj.shape),
        x=g.x,
        edge_type=np.zeros(0, dtype=np.int64),
        relation_names=["camo", "homo"],
    )
    det = CAREGNN(epochs=3, random_state=0).fit(empty, y)
    assert det.relation_names_ == ["camo", "homo"]
    assert det.decision_function(empty).shape == (g.n_nodes,)
    assert det.decision_function(g).shape == (g.n_nodes,)


def test_declared_but_unused_unknown_relation_is_fine():
    g, y = camo_graph(n_normal=100, n_anom=20)
    det = CAREGNN(epochs=3, random_state=0).fit(g, y)
    extra = Graph(
        adj=g.adj,
        x=g.x,
        edge_index=g.edge_index,
        edge_type=g.edge_type,
        relation_names=[*g.relation_names, "never_used"],
    )
    assert np.array_equal(det.decision_function(extra), det.decision_scores_)


def test_mutated_names_that_miss_codes_are_rejected():
    g, y = camo_graph(n_normal=60, n_anom=10)
    det = CAREGNN(epochs=2, random_state=0).fit(g, y)
    g.relation_names = ["camo"]  # mutated after construction: code 1 has no name
    with pytest.raises(ValueError, match="no name"):
        det.decision_function(g)
    with pytest.raises(ValueError, match="no name"):
        CAREGNN(epochs=1).fit(g, y)


def test_rl_step_must_be_sane():
    with pytest.raises(ValueError, match="rl_step"):
        CAREGNN(rl_step=1e-300)
    assert CAREGNN(rl_step=0.0).rl_step == 0.0
