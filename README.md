# graphspot

**Graph anomaly detection for things you haven't seen yet.**

[![PyPI](https://img.shields.io/pypi/v/graphspot)](https://pypi.org/project/graphspot/)
[![Python](https://img.shields.io/pypi/pyversions/graphspot)](https://pypi.org/project/graphspot/)
[![CI](https://github.com/JayeshSuryavanshi/graphspot/actions/workflows/ci.yml/badge.svg)](https://github.com/JayeshSuryavanshi/graphspot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

graphspot scores **nodes and edges that were absent at fit time** — new accounts, new
transactions, tomorrow's data. It installs in seconds without torch, takes a pandas
dataframe or a scipy sparse matrix, and every benchmark number in this README
regenerates from one command.

```bash
pip install graphspot            # tree, structural, and baseline detectors
pip install 'graphspot[deep]'    # + BWGNN (torch)
```

## Sixty seconds

```python
import pandas as pd, graphspot

tx = pd.read_parquet("transactions.parquet")

# one call on a bare edge list: one anomaly score per row
scores = graphspot.score_transactions(
    tx, source="buyer_id", target="seller_id",
    labels="chargeback",            # NaN / -1 = unlabeled
    edge_features=["amount"],
)
```

Or the full API:

```python
from graphspot.datasets import load_yelpchi
from graphspot.detectors import XGBGraph, FlatBaseline, Fraudar

g = load_yelpchi()

det = XGBGraph(random_state=0).fit(g, y=g.node_labels)
flat = FlatBaseline(random_state=0).fit(g, y=g.node_labels)

graphspot.evaluate(g.node_labels, det.decision_scores_,
                   baseline_scores=flat.decision_scores_)
# warns loudly if the graph model loses to the no-graph baseline

det.explain(k=3)
# [("2hop_mean(f21)", 0.14), ("1hop_max(f3)", 0.11), ...]

rings = Fraudar().fit(g)            # label-free, feature-free collusion blocks
rings.explain()
```

## Why graphspot

| | graphspot | PyGOD | PyOD graph module |
|---|---|---|---|
| Scores unseen nodes/edges | **yes, every detector** | no (transductive in practice) | no (`decision_function` raises) |
| Core install | numpy/scipy/sklearn/xgboost | requires torch (undeclared) | requires torch |
| Edge-level scoring | **yes** | node only | node only |
| Honest flat baselines | **automatic, warns when graph loses** | no | no |
| Label provenance metadata | **every dataset** | no | no |
| Maintained | yes | last release Feb 2024 | active |

Three design rules, each earned from a failure we measured elsewhere:

1. **`decision_function` never raises `NotImplementedError`.** Inductive scoring is the
   contract, not a feature.
2. **The core never imports torch.** CI installs the built wheel in a bare venv and
   asserts it. Deep detectors live behind `[deep]`.
3. **Every number is regenerable.** `scripts/kill_test.py` reproduces GADBench's
   published results under their protocol; `graphspot bench --quick` rebuilds the
   table below; a monthly CI job re-runs both from scratch.

## Detectors

| Detector | Level | Labels | Needs | Reference |
|---|---|---|---|---|
| `NeighborAggregation` | transform | — | core | makes any tabular model graph-aware |
| `XGBGraph` | node, edge | yes | core | GADBench rank 1 of 29 |
| `RFGraph` | node, edge | yes | core | GADBench rank 3 |
| `FlatBaseline` | node, edge | yes | core | the no-graph control, always available |
| `FlatUnsupervised` | node | no | core | IForest / LOF (BOND: each beats every deep model somewhere) |
| `OddBall` | node | no | core | Akoglu et al., PAKDD 2010 |
| `Fraudar` | edge, block | no | core | Hooi et al., KDD 2016 (clean-room, BSD-3) |
| `BWGNN` | node | yes | `[deep]` | Tang et al., ICML 2022 (clean-room, plain torch sparse) |

`graphspot.list_detectors()` reports what is usable in your environment. Any PyOD
detector runs through graphspot's graphs and benchmarks in one line:

```python
from pyod.models.ecod import ECOD
det = graphspot.compat.from_pyod(ECOD()).fit(g)
```

## Benchmarks

`graphspot bench --quick` — AUPRC ×100, mean over three seeded trials, out-of-the-box
defaults, no torch installed. Tolokers and Questions use the frozen split masks their
upstream ships.

```
dataset         XGBGraph       RFGraph  FlatBaseline
----------------------------------------------------
yelpchi      89.32±0.42*   76.76±0.41    83.45±0.51
amazon       93.75±0.57*   89.70±1.00    90.25±1.22
tolokers     57.34±1.34    58.18±1.35*   38.61±0.70
questions    22.05±1.30*   16.09±1.49    16.77±1.50
```

Read the losses too: on YelpChi and Amazon the plain random forest does not beat the
no-graph baseline. Publishing where the graph does not help is the point.

Under GADBench's own selection protocol (validation-selected hyperparameters, their
Amazon label conventions), `scripts/kill_test.py` reproduces their published numbers:
YelpChi 92.45±0.29 (published 91.11), Amazon 93.91±0.96 (published 93.33). BWGNN
reproduces its published Amazon AUROC within 0.74 points on CPU
(`scripts/demo_bwgnn.py`).

## The acceptance test

Strict-inductive Elliptic on a laptop, one command, no torch:

```bash
uv run python scripts/acceptance_elliptic.py
```

Fits on time steps 1–34 and scores steps 35–49 as a disjoint graph the model has never
seen: 203,769 nodes, **fit 10.1s, score 0.8s, peak rss 3.05GB**. The per-step table
shows why the script refuses to print one aggregate number: the dark-market shutdown
at step 43 collapses every model from 90s AUPRC to single digits. And on this dataset
the flat baseline edges out the graph model (56.4 vs 55.3 mean per-step AUPRC) —
Elliptic's features already embed 72 neighborhood aggregates, so the "flat" model is
quietly graph-informed. The loud baseline exists to surface exactly this.

## Datasets

Seven loaders with provenance as first-class metadata — every dataset carries
`label_type` (adjudicated / proxy / injected), `label_source`, license, and
redistribution status:

| Dataset | Nodes | Labels | Auto-download |
|---|---|---|---|
| YelpChi | 45,954 | proxy (filtered reviews) | yes |
| Amazon | 11,944 | proxy (helpful votes; first 3,305 unlabeled by convention) | yes |
| Tolokers | 11,758 | adjudicated (banned workers) | yes |
| Questions | 48,921 | adjudicated | yes |
| Elliptic | 203,769 | proxy (licit/illicit tags), 49 time steps | gated: `accept_license=True` (CC BY-NC-ND) |

Splits are utilities, not afterthoughts: seeded stratified, GADBench's 100-label
semi-supervised regime, and `temporal_split` for strict-inductive evaluation.
`evaluate_temporal` reports per-step metrics; `base_rate_sweep` re-evaluates as
positives thin toward production rarity.

## Platform notes

- macOS: xgboost needs Homebrew's libomp (`brew install libomp`).
- macOS: torch and xgboost cannot share one process (each bundles its own OpenMP
  runtime; the mix segfaults or deadlocks). graphspot raises a clear error instead of
  crashing; run `[deep]` and tree detectors in separate processes. Linux is
  unaffected, and CI proves coexistence there.

## Support

- Pre-1.0: minor releases may change APIs; anything removed gets a deprecation release
  first. Fitted-attribute names (`decision_scores_`, `labels_`, `threshold_`) are
  stable and PyOD-compatible.
- Scope: node- and edge-level anomaly detection on static graphs. Graph-level
  detection and streaming are out of scope for now.
- Small reproducible bug reports get priority. Issues and PRs welcome.

## Citation

```bibtex
@software{graphspot,
  author = {Suryavanshi, Jayesh},
  title  = {graphspot: inductive graph anomaly detection with honest baselines},
  url    = {https://github.com/JayeshSuryavanshi/graphspot},
  year   = {2026},
}
```

Third-party algorithm provenance is documented in
[LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY). License: BSD-3-Clause.
