from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from graphspot.graph import Graph


def score_transactions(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    labels: str | np.ndarray,
    edge_features: Sequence[str] | None = None,
    time: str | None = None,
    node_features: pd.DataFrame | None = None,
    detector: Any | None = None,
) -> np.ndarray:
    """One-call scoring of a transaction dataframe: one anomaly score per row.

    Builds the graph with `Graph.from_pandas`, fits an edge-level detector on the
    labeled rows (`labels` may be a column name or an array; NaN or -1 means
    unlabeled), and returns scores aligned with `df`. Works on a bare edge list:
    when no account-level features exist, node features are synthesized from
    incident edge attributes and degrees.
    """
    g = Graph.from_pandas(
        df,
        source=source,
        target=target,
        edge_features=edge_features,
        time=time,
        node_features=node_features,
    )
    if isinstance(labels, str):
        y = df[labels].to_numpy()
    else:
        y = np.asarray(labels)
    y = np.where(pd.isna(y), -1, y).astype(np.int64)

    if detector is None:
        from graphspot.detectors import XGBGraph

        detector = XGBGraph(level="edge")
    elif getattr(detector, "level", "edge") != "edge":
        raise ValueError("score_transactions needs an edge-level detector")

    detector.fit(g, y)
    return detector.decision_scores_
