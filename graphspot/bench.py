from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from graphspot.graph import Graph
from graphspot.splits import semi_supervised_split, stratified_split, train_labels


def _default_detectors() -> dict[str, Callable[[int], Any]]:
    from graphspot.detectors import FlatBaseline, RFGraph, XGBGraph

    return {
        "XGBGraph": lambda seed: XGBGraph(random_state=seed),
        "RFGraph": lambda seed: RFGraph(random_state=seed),
        "FlatBaseline": lambda seed: FlatBaseline(random_state=seed),
    }


def bench_graph(
    name: str,
    g: Graph,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    regime: str = "supervised",
    detectors: dict[str, Callable[[int], Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run every detector on one graph across seeded trials. Uses the dataset's own
    frozen split masks when it ships them and the regime is supervised; otherwise
    seeded stratified (or GADBench's 100-label semi-supervised) splits.
    """
    detectors = detectors or _default_detectors()
    y = g.node_labels
    rows = []
    for det_name, factory in detectors.items():
        auprcs, times = [], []
        for seed in seeds:
            masks = getattr(g, "split_masks", None)
            if regime == "semi":
                tr, _va, te = semi_supervised_split(y, seed=seed)
            elif masks is not None and seed < len(masks["train"]):
                tr = np.flatnonzero(masks["train"][seed])
                te = np.flatnonzero(masks["test"][seed])
            else:
                tr, _va, te = stratified_split(y, seed=seed)
            t0 = time.perf_counter()
            det = factory(seed).fit(g, train_labels(y, tr))
            scores = det.decision_scores_
            times.append(time.perf_counter() - t0)
            auprcs.append(average_precision_score(y[te], scores[te]))
        rows.append(
            {
                "dataset": name,
                "detector": det_name,
                "auprc": 100 * float(np.mean(auprcs)),
                "auprc_std": 100 * float(np.std(auprcs)),
                "fit_seconds": float(np.mean(times)),
                "trials": len(seeds),
                "regime": regime,
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    detectors = list(dict.fromkeys(r["detector"] for r in rows))
    by = {(r["dataset"], r["detector"]): r for r in rows}
    widths = [max(10, *(len(d) for d in datasets))] + [max(14, len(d) + 2) for d in detectors]
    header = "dataset".ljust(widths[0]) + "".join(
        d.rjust(w) for d, w in zip(detectors, widths[1:], strict=False)
    )
    lines = [header, "-" * len(header)]
    for ds in datasets:
        cells = [ds.ljust(widths[0])]
        best = max(by[(ds, d)]["auprc"] for d in detectors if (ds, d) in by)
        flat = by.get((ds, "FlatBaseline"))
        for det, w in zip(detectors, widths[1:], strict=False):
            r = by.get((ds, det))
            if r is None:
                cells.append("-".rjust(w))
                continue
            mark = "*" if r["auprc"] == best else " "
            cells.append(f"{r['auprc']:5.2f}±{r['auprc_std']:4.2f}{mark}".rjust(w))
        lines.append("".join(cells))
        if flat is not None and flat["auprc"] >= best:
            lines.append(f"  ! {ds}: no graph detector beats the flat baseline here")
    lines.append("* best per dataset; AUPRC x100, mean ± std over seeded trials")
    return "\n".join(lines)


def run_bench(
    names: Sequence[str],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    regime: str = "supervised",
) -> list[dict[str, Any]]:
    from graphspot.datasets import QUICK_LOADERS

    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in QUICK_LOADERS:
            raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(QUICK_LOADERS)}")
        rows += bench_graph(name, QUICK_LOADERS[name](), seeds=seeds, regime=regime)
    return rows
