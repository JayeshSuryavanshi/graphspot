"""Week-1 kill test: XGBGraph must land within +/-2 AUPRC of GADBench's published
fully-supervised numbers (YelpChi 91.11, Amazon 93.33) or the project stops and reassesses.

Protocol mirrors GADBench's actual selection procedure, extracted from their source:
the aggregator and hop count are hyperparameters selected on validation AUPRC, splits are
stratified over labeled nodes (Amazon's first 3305 nodes are unlabeled by convention and
carry y=-1 straight from the loader), and results average seeded trials. Their exact
train/val/test masks are frozen DGL artifacts, so a small gap is expected and documented.

Run: uv run python scripts/kill_test.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from graphspot.datasets import load_amazon, load_yelpchi
from graphspot.detectors import XGBGraph

PUBLISHED = {"yelpchi": 91.11, "amazon": 93.33}
GRID = [
    {"hops": h, "aggregators": a}
    for h in (1, 2, 3)
    for a in (("mean",), ("sum",), ("max",), ("mean", "max", "std"))
]
SEEDS = (0, 1, 2)


def one_trial(g, y, seed):
    labeled = np.flatnonzero(y >= 0)
    train_idx, rest = train_test_split(
        labeled, train_size=0.7, stratify=y[labeled], random_state=seed
    )
    val_idx, test_idx = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    y_train = np.full(g.n_nodes, -1, dtype=np.int64)
    y_train[train_idx] = y[train_idx]

    best = None
    for cfg in GRID:
        det = XGBGraph(random_state=seed, **cfg).fit(g, y_train)
        val_ap = average_precision_score(y[val_idx], det.decision_scores_[val_idx])
        if best is None or val_ap > best[0]:
            test_ap = average_precision_score(y[test_idx], det.decision_scores_[test_idx])
            best = (val_ap, cfg, test_ap)
    return best


def run(name, loader):
    t0 = time.perf_counter()
    g = loader()
    y = g.node_labels
    picks, tests = [], []
    for seed in SEEDS:
        _, cfg, test_ap = one_trial(g, y, seed)
        picks.append(cfg)
        tests.append(test_ap * 100)
    mean, std = float(np.mean(tests)), float(np.std(tests))
    target = PUBLISHED[name]
    ok = abs(mean - target) <= 2.0
    print(
        f"{name:8} AUPRC={mean:6.2f} +/- {std:4.2f} over {len(SEEDS)} trials "
        f"(published {target}, {'PASS' if ok else 'FAIL'})  "
        f"picked={picks[0]['hops']}hop/{'+'.join(picks[0]['aggregators'])}  "
        f"{time.perf_counter() - t0:5.1f}s"
    )
    return ok


if __name__ == "__main__":
    results = [run("yelpchi", load_yelpchi), run("amazon", load_amazon)]
    assert "torch" not in sys.modules, "kill test must run torch-free"
    print("KILL TEST:", "PASS" if all(results) else "FAIL, stop and reassess")
    sys.exit(0 if all(results) else 1)
