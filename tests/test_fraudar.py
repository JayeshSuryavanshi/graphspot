from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from graphspot import Graph
from graphspot.detectors import Fraudar


def bipartite_with_ring(
    n_buyers=2000, n_sellers=800, n_edges=1500, ring_buyers=30, ring_sellers=5, seed=0
):
    """Sparse random marketplace plus a dense collusive block. Every ring buyer hits
    every ring seller, and no feature carries any signal at all."""
    rng = np.random.default_rng(seed)
    rows = [
        (f"b{rng.integers(0, n_buyers)}", f"s{rng.integers(0, n_sellers)}", 0)
        for _ in range(n_edges)
    ]
    ring_b = [f"rb{i}" for i in range(ring_buyers)]
    ring_s = [f"rs{j}" for j in range(ring_sellers)]
    rows += [(b, s, 1) for b in ring_b for s in ring_s]
    df = pd.DataFrame(rows, columns=["buyer", "seller", "in_ring"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    g = Graph.from_pandas(df, source="buyer", target="seller")
    return g, df, set(ring_b), set(ring_s)


# The ring must genuinely survive the greedy shave: each ring buyer's weighted
# priority (5 edges x 1/log(35) ~ 1.4) must exceed the degree-2 background buyers
# (~1.04), or the shave dismantles the ring before the background. A 30x3 ring sits
# right at that edge; 30x5 is safely inside it, matching the block densities the
# KDD'16 paper injects.


def test_recovers_planted_ring():
    g, df, ring_b, ring_s = bipartite_with_ring()
    det = Fraudar().fit(g)
    block = det.blocks_[0]
    found_rows = set(block.row_ids)
    found_cols = set(block.col_ids)
    row_recall = len(found_rows & ring_b) / len(ring_b)
    col_recall = len(found_cols & ring_s) / len(ring_s)
    row_precision = len(found_rows & ring_b) / max(len(found_rows), 1)
    assert row_recall >= 0.9
    assert col_recall == 1.0
    assert row_precision >= 0.9
    in_ring = df["in_ring"].to_numpy() == 1
    assert det.decision_scores_[in_ring].min() > 0
    assert det.decision_scores_[in_ring].mean() > det.decision_scores_[~in_ring].mean()


def test_camouflage_resistance():
    """A ring that also touches the most popular seller must still be found: the
    1/log(degree+5) weighting discounts edges into high-degree targets."""
    g, df, ring_b, ring_s = bipartite_with_ring(seed=3)
    camo = pd.DataFrame([(b, "s0", 0) for b in ring_b], columns=["buyer", "seller", "in_ring"])
    df2 = pd.concat([df, camo], ignore_index=True)
    g2 = Graph.from_pandas(df2, source="buyer", target="seller")
    det = Fraudar().fit(g2)
    found_rows = set(det.blocks_[0].row_ids)
    assert len(found_rows & ring_b) / len(ring_b) >= 0.9


def test_multiple_blocks():
    g, df, ring_b, ring_s = bipartite_with_ring(seed=5)
    extra_b = [f"xb{i}" for i in range(20)]
    extra_s = [f"xs{j}" for j in range(5)]
    df2 = pd.concat(
        [
            df,
            pd.DataFrame(
                [(b, s, 1) for b in extra_b for s in extra_s],
                columns=["buyer", "seller", "in_ring"],
            ),
        ],
        ignore_index=True,
    )
    g2 = Graph.from_pandas(df2, source="buyer", target="seller")
    det = Fraudar(n_blocks=2).fit(g2)
    assert len(det.blocks_) == 2
    all_rows = set(det.blocks_[0].row_ids) | set(det.blocks_[1].row_ids)
    assert len(all_rows & (ring_b | set(extra_b))) >= 0.9 * (len(ring_b) + len(extra_b))


def test_scores_new_graph_by_block_membership():
    g, df, ring_b, ring_s = bipartite_with_ring()
    det = Fraudar().fit(g)
    repeat = pd.DataFrame(
        [(next(iter(ring_b)), next(iter(ring_s)), 1), ("bfresh", "sfresh", 0)],
        columns=["buyer", "seller", "in_ring"],
    )
    g_new = Graph.from_pandas(repeat, source="buyer", target="seller")
    scores = det.decision_function(g_new)
    assert scores[repeat["in_ring"] == 1][0] > 0
    assert scores[repeat["in_ring"] == 0][0] == 0.0


def test_score_is_pure_and_deterministic():
    g, *_ = bipartite_with_ring()
    det = Fraudar().fit(g)
    before = pickle.dumps(det)
    s1 = det.decision_function(g)
    assert pickle.dumps(det) == before
    assert np.array_equal(s1, det.decision_function(g))
    det2 = Fraudar().fit(g)
    assert np.array_equal(det.decision_scores_, det2.decision_scores_)


def test_rejects_empty_and_bad_params():
    with pytest.raises(ValueError, match="edges"):
        Fraudar().fit(Graph(adj=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="n_blocks"):
        Fraudar(n_blocks=0)
    with pytest.raises(ValueError, match="supports levels"):
        Fraudar(level="node")
