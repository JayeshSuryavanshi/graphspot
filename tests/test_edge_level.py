from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from graphspot import Graph, score_transactions
from graphspot.detectors import FlatBaseline, XGBGraph
from graphspot.transforms import NeighborAggregation, ensure_node_features


def tx_frame(n_normal=600, n_fraud=120, seed=0):
    """Transactions where fraud is only visible relationally: fraudulent edges funnel
    into a small set of mule sellers from many one-off buyers."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        rows.append((f"b{rng.integers(0, 200)}", f"s{rng.integers(0, 50)}", rng.lognormal(3, 1), 0))
    mules = [f"mule{k}" for k in range(3)]
    for i in range(n_fraud):
        rows.append((f"fb{i}", mules[i % 3], rng.lognormal(3, 1), 1))
    df = pd.DataFrame(rows, columns=["buyer", "seller", "amount", "is_fraud"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_graph(df):
    return Graph.from_pandas(df, source="buyer", target="seller", edge_features=["amount"])


def test_edge_level_fit_and_shapes():
    df = tx_frame()
    g = build_graph(df)
    y = df["is_fraud"].to_numpy()
    det = XGBGraph(level="edge", random_state=0).fit(g, y)
    assert det.decision_scores_.shape == (g.n_edges,)
    assert det.predict().shape == (g.n_edges,)
    names = det.explain(k=3)
    assert len(names) == 3


def test_edge_level_separates_relational_fraud():
    from sklearn.metrics import roc_auc_score

    df = tx_frame()
    g = build_graph(df)
    y = df["is_fraud"].to_numpy()
    hide = np.random.default_rng(1).random(len(y)) < 0.4
    y_train = np.where(hide, -1, y)
    det = XGBGraph(level="edge", random_state=0).fit(g, y_train)
    auc = roc_auc_score(y[hide], det.decision_scores_[hide])
    assert auc > 0.95


def test_edge_level_inductive_on_unseen_graph():
    df_a, df_b = tx_frame(seed=0), tx_frame(seed=7)
    g_a, g_b = build_graph(df_a), build_graph(df_b)
    det = XGBGraph(level="edge", random_state=0).fit(g_a, df_a["is_fraud"].to_numpy())
    scores = det.decision_function(g_b)
    assert scores.shape == (g_b.n_edges,)
    y_b = df_b["is_fraud"].to_numpy()
    assert scores[y_b == 1].mean() > scores[y_b == 0].mean()


def test_edge_level_score_is_pure():
    df = tx_frame()
    g = build_graph(df)
    det = XGBGraph(level="edge", random_state=0).fit(g, df["is_fraud"].to_numpy())
    before = pickle.dumps(det)
    det.decision_function(g)
    assert pickle.dumps(det) == before


def test_flat_baseline_edge_level():
    df = tx_frame()
    g = build_graph(df)
    det = FlatBaseline(level="edge", random_state=0).fit(g, df["is_fraud"].to_numpy())
    assert det.decision_scores_.shape == (g.n_edges,)


def test_ensure_node_features_synthesis():
    df = tx_frame(n_normal=50, n_fraud=10)
    g = build_graph(df)
    assert g.x is None
    g2 = ensure_node_features(g)
    assert g2.x is not None
    assert g2.x.shape == (g.n_nodes, 3)
    assert g2.feature_names == ["log1p_out_degree", "log1p_in_degree", "incident_mean(amount)"]
    assert g.x is None, "original graph must not be mutated"


def test_transform_edges_names_align():
    df = tx_frame(n_normal=80, n_fraud=10)
    g = ensure_node_features(build_graph(df))
    na = NeighborAggregation(hops=1, aggregators=("mean",))
    na.fit(g)
    feats = na.transform_edges(g)
    names = na.edge_feature_names(g)
    assert feats.shape == (g.n_edges, len(names))
    assert names[0].startswith("src_") and "amount" in names[-1]


def test_score_transactions_end_to_end():
    from sklearn.metrics import roc_auc_score

    df = tx_frame()
    y = df["is_fraud"].to_numpy().astype(float)
    hidden = np.random.default_rng(2).random(len(df)) < 0.5
    df = df.assign(label=np.where(hidden, np.nan, y))
    scores = score_transactions(
        df,
        source="buyer",
        target="seller",
        labels="label",
        edge_features=["amount"],
        detector=None,
    )
    assert scores.shape == (len(df),)
    assert roc_auc_score(y[hidden], scores[hidden]) > 0.9


def test_score_transactions_rejects_node_level_detector():
    df = tx_frame(n_normal=50, n_fraud=10)
    with pytest.raises(ValueError, match="edge-level"):
        score_transactions(
            df,
            source="buyer",
            target="seller",
            labels="is_fraud",
            detector=XGBGraph(level="node"),
        )
