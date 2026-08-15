"""Week-5 demo: BWGNN on live torch, checked against GADBench's published Amazon
AUROC (98.27, fully supervised). Their masks are frozen DGL artifacts, so a seeded
stratified split stands in and a small gap is expected. Also reports the flat
baseline and wall clock.

Run: uv run python scripts/demo_bwgnn.py
"""

from __future__ import annotations

import resource
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from graphspot.datasets import load_amazon
from graphspot.detectors import BWGNN
from graphspot.splits import stratified_split, train_labels

PUBLISHED_AUROC = 98.27
SEEDS = (0, 1, 2)

g = load_amazon()
y = g.node_labels

aurocs, auprcs = [], []
t0 = time.perf_counter()
for seed in SEEDS:
    tr, _va, te = stratified_split(y, seed=seed)
    det = BWGNN(random_state=seed).fit(g, train_labels(y, tr))
    aurocs.append(roc_auc_score(y[te], det.decision_scores_[te]) * 100)
    auprcs.append(average_precision_score(y[te], det.decision_scores_[te]) * 100)
elapsed = (time.perf_counter() - t0) / len(SEEDS)

rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
gap = float(np.mean(aurocs)) - PUBLISHED_AUROC
print(
    f"amazon  BWGNN AUROC {np.mean(aurocs):5.2f} ± {np.std(aurocs):4.2f} "
    f"(published {PUBLISHED_AUROC}, gap {gap:+.2f})  AUPRC {np.mean(auprcs):5.2f}\n"
    f"        flat baseline: see `graphspot bench` (xgboost cannot share this process "
    f"on macOS, dual OpenMP runtimes)\n"
    f"        {elapsed:.0f}s per fit (100 epochs, full batch, cpu), peak rss {rss_gb:.2f}GB"
)
print("PASS" if abs(gap) <= 2.0 else "FAIL vs published (investigate before claiming)")
