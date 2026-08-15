from __future__ import annotations

from typing import Any, Literal

import numpy as np

from graphspot.base import BaseDetector
from graphspot.graph import as_graph


class FlatUnsupervised(BaseDetector):
    """Unsupervised tabular baselines on raw node features, no graph at all.

    Thin adapters over scikit-learn's IsolationForest and LocalOutlierFactor
    (novelty mode, so scoring unseen nodes works). They earn their slot on
    evidence: in the BOND benchmark IsolationForest is the single best method on
    DGraph and LOF the best on Reddit, beating every deep graph model. When one
    of these wins, the graph carries no signal worth modelling.
    """

    supported_levels = ("node",)

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        estimator: Literal["iforest", "lof"] = "iforest",
        contamination: float = 0.01,
        random_state: int | None = None,
        params: dict[str, Any] | None = None,
    ):
        super().__init__(level=level, contamination=contamination, random_state=random_state)
        if estimator not in ("iforest", "lof"):
            raise ValueError(f"estimator must be 'iforest' or 'lof', got {estimator!r}")
        self.estimator = estimator
        self.params = params

    def fit(self, graph: Any, y: np.ndarray | None = None) -> FlatUnsupervised:
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("FlatUnsupervised needs node features (graph.x)")
        self.n_features_in_ = g.x.shape[1]
        if self.estimator == "iforest":
            from sklearn.ensemble import IsolationForest

            kw: dict[str, Any] = dict(random_state=self.random_state)
            kw.update(self.params or {})
            self.model_ = IsolationForest(**kw)
        else:
            from sklearn.neighbors import LocalOutlierFactor

            kw = dict(novelty=True)
            kw.update(self.params or {})
            self.model_ = LocalOutlierFactor(**kw)
        self.model_.fit(g.x)
        self._finalize_fit(-self.model_.score_samples(g.x))
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        g = as_graph(graph)
        if g.x is None or g.x.shape[1] != self.n_features_in_:
            raise ValueError(f"Graph must carry {self.n_features_in_} node features")
        return -self.model_.score_samples(g.x)
