"""Week-3 demo: collusion blocks that node-level detectors structurally cannot see.

Part 1 plants a 30x5 complete bipartite ring inside a sparse synthetic marketplace where
node features carry zero signal. A supervised node-level detector has nothing to learn
from; label-free Fraudar recovers the block from structure alone.

Part 2 runs Fraudar on the real YelpChi graph and reports the densest block plus wall
clock and peak memory, per the no-unmeasured-scale-claims rule.

Run: uv run python scripts/demo_rings.py
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
import pandas as pd

from graphspot import Graph
from graphspot.datasets import load_yelpchi
from graphspot.detectors import Fraudar

rng = np.random.default_rng(0)

rows = [(f"b{rng.integers(0, 2000)}", f"s{rng.integers(0, 800)}", 0) for _ in range(1500)]
ring_b = [f"rb{i}" for i in range(30)]
ring_s = [f"rs{j}" for j in range(5)]
rows += [(b, s, 1) for b in ring_b for s in ring_s]
df = pd.DataFrame(rows, columns=["buyer", "seller", "in_ring"])
g = Graph.from_pandas(df, source="buyer", target="seller")

det = Fraudar().fit(g)
block = det.blocks_[0]
found_b = set(block.row_ids) & set(ring_b)
found_s = set(block.col_ids) & set(ring_s)
print("part 1: synthetic marketplace, zero feature signal, no labels")
print(
    f"  planted 30x5 ring -> recovered {len(found_b)}/30 buyers, {len(found_s)}/5 sellers, "
    f"density {block.density:.3f}, precision "
    f"{len(found_b) / max(len(block.row_ids), 1):.2f}"
)

t0 = time.perf_counter()
gy = load_yelpchi()
det = Fraudar().fit(gy)
elapsed = time.perf_counter() - t0
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
block = det.blocks_[0]
flagged = det.decision_scores_ > 0
labels = gy.node_labels
member_rate = labels[block.nodes].mean()
base_rate = labels.mean()
print("\npart 2: YelpChi, real graph, label-free")
print(
    f"  densest block: {len(block.rows)} sources x {len(block.cols)} targets, "
    f"{block.n_edges} edges, density {block.density:.3f}"
)
print(
    f"  fraud label rate inside block {member_rate:.1%} vs base rate {base_rate:.1%} "
    f"({member_rate / base_rate:.1f}x), {flagged.sum()} edges flagged"
)
print(f"  {elapsed:.1f}s wall, peak rss {rss_gb:.2f}GB, {gy.adj.nnz} edges, pure python heap")
sys.exit(0)
