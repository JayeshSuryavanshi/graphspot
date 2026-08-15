# graphspot

Graph anomaly detection for people who have to score things they have never seen before.

- **Inductive by contract.** Every detector's `decision_function` scores nodes absent at fit
  time. Never `NotImplementedError`.
- **Installs without torch.** The core is numpy/scipy/sklearn/pandas/xgboost. Deep detectors
  live behind `pip install graphspot[deep]`.
- **Honest baselines.** Every evaluation can include a no-graph tabular baseline, and graphspot
  warns loudly when the graph model fails to beat it.

Status: pre-release, under active development. The v0 detector set is
`NeighborAggregation` (transform), `XGBGraph`, `RFGraph`, `FlatBaseline`, with
`FlatUnsupervised`, `OddBall`, `Fraudar` and `BWGNN` (behind `[deep]`) on the way.

```python
import graphspot
from graphspot.detectors import XGBGraph, FlatBaseline
from graphspot.datasets import load_yelpchi

g = load_yelpchi()

det = XGBGraph(random_state=0).fit(g, y=g.node_labels)
flat = FlatBaseline(random_state=0).fit(g, y=g.node_labels)

print(graphspot.evaluate(
    g.node_labels,
    det.decision_scores_,
    baseline_scores=flat.decision_scores_,
))

det.explain(k=5)   # e.g. [("2hop_mean(f21)", 0.14), ("1hop_max(f3)", 0.11), ...]
```

Works directly on transaction dataframes:

```python
g = graphspot.Graph.from_pandas(
    tx, source="buyer_id", target="seller_id",
    edge_features=["amount"], time="ts",
    node_features=accounts.set_index("account_id"),
)
```

## Benchmarks

Every number regenerates from one command: `graphspot bench --quick`. AUPRC x100,
mean over three seeded trials, on the four auto-download datasets, out-of-the-box
defaults, no torch installed. Tolokers and Questions use the split masks their
upstream ships; the flat baseline is the same XGBoost on raw features with no graph.

```
dataset         XGBGraph       RFGraph  FlatBaseline
----------------------------------------------------
yelpchi      89.32±0.42*   76.76±0.41    83.45±0.51
amazon       93.75±0.57*   89.70±1.00    90.25±1.22
tolokers     57.34±1.34    58.18±1.35*   38.61±0.70
questions    22.05±1.30*   16.09±1.49    16.77±1.50
```

Read the losses too: on YelpChi and Amazon the plain random forest does not beat
the no-graph baseline. Publishing where the graph does not help is the point.

macOS note: xgboost needs Homebrew's libomp (`brew install libomp`).

## The acceptance test

Strict-inductive Elliptic, one command, on a laptop, no torch:
`uv run python scripts/acceptance_elliptic.py` fits on time steps 1-34 and scores
steps 35-49 as a disjoint graph the model has never seen. 203,769 nodes; fit 10.1s,
score 0.8s, peak rss 3.05GB. PyOD's graph detectors raise `NotImplementedError` on
`decision_function`; PyGOD's flagship OOMs on this graph on a 12 GB GPU.

Two honest findings the per-step table makes visible. The dark-market shutdown at
step 43 collapses every model (AUPRC in the 90s drops to single digits), which is
why the script refuses to print a single aggregate number. And on this dataset the
graph model does not beat the flat baseline (mean per-step AUPRC 55.3 vs 56.4):
Elliptic's feature matrix already contains 72 neighborhood-aggregate columns
computed by the dataset authors, so the flat model is quietly graph-informed. The
loud baseline exists precisely to surface results like this.


License: BSD-3-Clause.
