from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from graphspot.base import BaseDetector
from graphspot.graph import as_graph
from graphspot.transforms import NeighborAggregation


class PyODAdapter(BaseDetector):
    """Run any PyOD detector through graphspot's graphs, splits, and benchmarks.

    With `use_graph=True` the PyOD model sees neighbor-aggregated features, so a
    plain tabular detector becomes graph-aware in one line; with False it sees the
    raw node features. Works with PyOD's tabular detectors (IForest, ECOD, LOF,
    ...). PyOD's `pyg_*` graph detectors are transductive and raise on
    `decision_function` by design; the adapter surfaces that with a clear error
    instead of passing it through.
    """

    supported_levels = ("node",)

    def __init__(
        self,
        *,
        estimator: Any = None,
        level: Literal["node", "edge"] = "node",
        use_graph: bool = True,
        hops: int = 2,
        aggregators: Sequence[str] = ("mean", "max", "std"),
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(level=level, contamination=contamination, random_state=random_state)
        if estimator is None:
            raise ValueError("estimator is required: any fitted-API PyOD detector instance")
        self.estimator = estimator
        self.use_graph = use_graph
        self.hops = hops
        self.aggregators = tuple(aggregators)

    def _features(self, graph: Any) -> np.ndarray:
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("PyODAdapter needs node features (graph.x)")
        if self.use_graph:
            return self.aggregation_.transform(g)
        return g.x

    def fit(self, graph: Any, y: np.ndarray | None = None) -> PyODAdapter:
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("PyODAdapter needs node features (graph.x)")
        if self.use_graph:
            self.aggregation_ = NeighborAggregation(hops=self.hops, aggregators=self.aggregators)
            self.aggregation_.fit(g)
        self.estimator.fit(self._features(g))
        self._finalize_fit(np.asarray(self.estimator.decision_scores_, dtype=np.float64))
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        features = self._features(graph)
        try:
            return np.asarray(self.estimator.decision_function(features), dtype=np.float64)
        except NotImplementedError as err:
            raise RuntimeError(
                f"{type(self.estimator).__name__} is a transductive PyOD detector and cannot "
                "score unseen data; its decision_scores_ from fit time are available via "
                "decision_scores_. For inductive scoring use a graphspot detector."
            ) from err


def from_pyod(estimator: Any, **kwargs: Any) -> PyODAdapter:
    """Wrap a PyOD detector instance for graphspot graphs and evaluation.

    >>> from pyod.models.iforest import IForest          # doctest: +SKIP
    >>> det = from_pyod(IForest()).fit(g)                # doctest: +SKIP
    """
    return PyODAdapter(estimator=estimator, **kwargs)
