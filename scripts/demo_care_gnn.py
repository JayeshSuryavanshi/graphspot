"""v0.2 demo: CAREGNN on the 3-relation YelpChi and Amazon graphs, against BWGNN on
the union graph, in two regimes: a seeded 40% stratified split, and the
label-scarce GADBench regime (20 positive, 80 negative labels) that motivates
CARE-GNN in the spec. `rl_step=0` (thresholds fixed at 0.5) is the ablation that
shows whether the bandit earns its keep. Paired per seed, AUPRC primary.

No published CARE-GNN numbers are recorded here yet: add them (with the paper's
split protocol) once they are read from the paper, not from memory.

Run: uv run python scripts/demo_care_gnn.py
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from graphspot.datasets import load_amazon, load_yelpchi
from graphspot.detectors import BWGNN, CAREGNN
from graphspot.splits import semi_supervised_split, stratified_split, train_labels

SEEDS = (0, 1, 2)
LOADERS = {"amazon": load_amazon, "yelpchi": load_yelpchi}
REGIMES = {
    "40% train": lambda y, seed: stratified_split(y, train_size=0.4, val_size=0.2, seed=seed),
    "20+/80-": lambda y, seed: semi_supervised_split(y, seed=seed),
}
DETECTORS = {
    "BWGNN": lambda seed: BWGNN(random_state=seed),
    "CAREGNN": lambda seed: CAREGNN(random_state=seed),
    "CAREGNN p=.5": lambda seed: CAREGNN(rl_step=0.0, random_state=seed),
}

for name, load in LOADERS.items():
    g = load(relations=True)
    y = g.node_labels
    for regime, split in REGIMES.items():
        auprc: dict[str, list[float]] = {d: [] for d in DETECTORS}
        for seed in SEEDS:
            tr, _va, te = split(y, seed)
            for det_name, make in DETECTORS.items():
                t0 = time.perf_counter()
                s = make(seed).fit(g, train_labels(y, tr)).decision_scores_[te]
                secs = time.perf_counter() - t0
                auprc[det_name].append(average_precision_score(y[te], s) * 100)
                print(
                    f"{name:8s} {regime:9s} seed {seed} {det_name:12s} "
                    f"AUROC {roc_auc_score(y[te], s) * 100:5.2f}  "
                    f"AUPRC {auprc[det_name][-1]:5.2f}  {secs:.0f}s"
                )
        for det_name, vals in auprc.items():
            mean, std = np.mean(vals), np.std(vals)
            print(f"{name:8s} {regime:9s} {det_name:12s} AUPRC {mean:5.2f} ± {std:4.2f}")
        delta = np.subtract(auprc["CAREGNN"], auprc["BWGNN"])
        tag = f"{name:8s} {regime:9s}"
        print(f"{tag} CAREGNN - BWGNN AUPRC {delta.mean():+.2f} per seed {delta.round(2)}")

# ru_maxrss is bytes on macOS, kilobytes on Linux
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
print(f"peak rss {rss_gb:.2f}GB (cpu)")
