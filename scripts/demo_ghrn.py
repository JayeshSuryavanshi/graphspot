"""v0.2 demo: GHRN against its own encoder (BWGNN) on Amazon and YelpChi, same
seeded stratified splits, paired per seed. GHRN only earns its slot if pruning
heterophilous edges beats the unpruned encoder, so the paired delta is the
headline, and the published numbers are the sanity check. GADBench's masks are
frozen DGL artifacts, so a seeded split stands in and a small gap is expected.

Run: uv run python scripts/demo_ghrn.py
"""

from __future__ import annotations

import resource
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from graphspot.datasets import load_amazon, load_yelpchi
from graphspot.detectors import BWGNN, GHRN
from graphspot.splits import stratified_split, train_labels

SEEDS = (0, 1, 2)
LOADERS = {"amazon": load_amazon, "yelpchi": load_yelpchi}


def run(det, g, y, tr, te):
    t0 = time.perf_counter()
    s = det.fit(g, train_labels(y, tr)).decision_scores_[te]
    return (
        roc_auc_score(y[te], s) * 100,
        average_precision_score(y[te], s) * 100,
        time.perf_counter() - t0,
    )


for name, load in LOADERS.items():
    g = load()
    y = g.node_labels
    res = {"BWGNN": [], "GHRN": []}
    for seed in SEEDS:
        tr, _va, te = stratified_split(y, seed=seed)
        res["BWGNN"].append(run(BWGNN(random_state=seed), g, y, tr, te))
        res["GHRN"].append(run(GHRN(random_state=seed), g, y, tr, te))
    for det, rows in res.items():
        auroc, auprc, secs = (np.array(c) for c in zip(*rows, strict=True))
        print(
            f"{name:8s} {det:6s} AUROC {auroc.mean():5.2f} ± {auroc.std():4.2f}  "
            f"AUPRC {auprc.mean():5.2f} ± {auprc.std():4.2f}  {secs.mean():.0f}s/fit"
        )
    d_auprc = np.array([b[1] - a[1] for a, b in zip(res["BWGNN"], res["GHRN"], strict=True)])
    wins = int((d_auprc > 0).sum())
    print(
        f"{name:8s} GHRN - BWGNN AUPRC {d_auprc.mean():+.2f} "
        f"(per seed {', '.join(f'{d:+.2f}' for d in d_auprc)}; {wins}/{len(SEEDS)} wins)"
    )

rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"peak rss {rss_gb:.2f}GB (full batch, cpu)")
