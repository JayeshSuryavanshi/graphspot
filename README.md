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

License: BSD-3-Clause.
