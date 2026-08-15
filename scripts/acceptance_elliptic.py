"""The v0 acceptance test: strict-inductive Elliptic on a laptop, one command.

Fit on time steps 1-34, score steps 35-49 as a disjoint graph the model has never seen,
report per-step AUPRC (never a single aggregate: the dark-market shutdown at step 43
collapses the base rate, and one number would average that away), the flat-baseline
lift, a base-rate sweep, and wall clock plus peak memory.

PyOD's graph detectors raise NotImplementedError on decision_function; PyGOD's flagship
OOMs on this graph on a 12 GB GPU (pygod issue #118). This script is the counterexample.

Run: uv run python scripts/acceptance_elliptic.py
"""

from __future__ import annotations

import resource
import sys
import time

import graphspot
from graphspot.datasets import load_elliptic
from graphspot.detectors import FlatBaseline, XGBGraph
from graphspot.metrics import base_rate_sweep, evaluate_temporal

CUTOFF = 34

t0 = time.perf_counter()
g = load_elliptic(accept_license=True)
t_load = time.perf_counter() - t0

train, test = graphspot.temporal_split(g, cutoff=CUTOFF)
print(
    f"elliptic: {g.n_nodes} nodes, {g.adj.nnz} edges, loaded in {t_load:.1f}s\n"
    f"train: steps <= {CUTOFF}, {train.n_nodes} nodes, "
    f"{(train.node_labels == 1).sum()} illicit / {(train.node_labels >= 0).sum()} labeled\n"
    f"test : steps > {CUTOFF}, {test.n_nodes} nodes, "
    f"{(test.node_labels == 1).sum()} illicit / {(test.node_labels >= 0).sum()} labeled\n"
)

t1 = time.perf_counter()
det = XGBGraph(random_state=0).fit(train, train.node_labels)
t_fit = time.perf_counter() - t1
t2 = time.perf_counter()
scores = det.decision_function(test)
t_score = time.perf_counter() - t2

flat = FlatBaseline(random_state=0).fit(train, train.node_labels)
flat_scores = flat.decision_function(test)

report = evaluate_temporal(test.node_labels, scores, test.node_time)
flat_report = evaluate_temporal(test.node_labels, flat_scores, test.node_time)

print("step  labeled  illicit  base%   XGBGraph  flat")
for row, frow in zip(report["steps"], flat_report["steps"], strict=False):
    auprc = f"{row['auprc'] * 100:6.1f}" if "auprc" in row else "     -"
    fap = f"{frow['auprc'] * 100:6.1f}" if "auprc" in frow else "     -"
    print(
        f"{row['step']:4.0f}  {row['n_labeled']:7.0f}  {row['n_pos']:7.0f}"
        f"  {row['base_rate'] * 100:5.1f}  {auprc}    {fap}"
    )
print(
    f"\nmean per-step AUPRC: XGBGraph {report['mean_auprc'] * 100:.1f} "
    f"vs flat {flat_report['mean_auprc'] * 100:.1f} "
    f"over {report['n_scored_steps']} scored steps"
)

print("\nbase-rate sweep on the test span (positives downsampled):")
for r in base_rate_sweep(test.node_labels, scores, rates=(0.05, 0.01, 0.003), seed=0):
    print(
        f"  base {r['base_rate'] * 100:4.1f}%  AUPRC {r['auprc'] * 100:5.1f}"
        f" ± {r['auprc_std'] * 100:4.1f}  rec@k {r['rec_at_k'] * 100:5.1f}"
    )

rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"\nfit {t_fit:.1f}s, score {t_score:.1f}s, peak rss {rss_gb:.2f}GB, torch-free")
assert "torch" not in sys.modules
