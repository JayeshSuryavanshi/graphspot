from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from graphspot.base import BaseDetector
from graphspot.graph import as_graph
from graphspot.transforms import NeighborAggregation, ensure_node_features


def _guard_darwin_omp() -> None:
    """The dual-OpenMP conflict, from the xgboost side; see bwgnn._DARWIN_OMP_MSG."""
    import sys

    if sys.platform == "darwin" and "torch" in sys.modules:
        from graphspot.detectors.bwgnn import _DARWIN_OMP_MSG

        raise RuntimeError(_DARWIN_OMP_MSG)


class _SupervisedTreeDetector(BaseDetector):
    supported_levels = ("node", "edge")

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        contamination: float = 0.01,
        random_state: int | None = None,
        params: dict[str, Any] | None = None,
    ):
        super().__init__(level=level, contamination=contamination, random_state=random_state)
        self.params = params

    def _features(self, graph: Any) -> np.ndarray:
        raise NotImplementedError

    def _fit_features(self, graph: Any) -> np.ndarray:
        raise NotImplementedError

    def _make_estimator(self, y_labeled: np.ndarray) -> Any:
        raise NotImplementedError

    def fit(self, graph: Any, y: np.ndarray | None = None) -> _SupervisedTreeDetector:
        g = as_graph(graph)
        y = self._validate_labels(g, y, self.level)
        features = self._fit_features(g)
        labeled = y >= 0
        self.model_ = self._make_estimator(y[labeled])
        self.model_.fit(features[labeled], y[labeled])
        self._finalize_fit(self.model_.predict_proba(features)[:, 1])
        return self

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(self._features(graph))[:, 1].astype(np.float64)

    def explain(self, idx: int | None = None, k: int = 5) -> list[tuple[str, float]]:
        self._check_fitted()
        importances = np.asarray(self.model_.feature_importances_, dtype=np.float64)
        names = self._feature_names()
        if idx is not None:
            raise NotImplementedError(
                "Per-instance explanations land with the week-6 explain(); "
                "call explain() without idx for global importances"
            )
        order = np.argsort(importances)[::-1][:k]
        return [(names[i], float(importances[i])) for i in order]

    def _feature_names(self) -> list[str]:
        raise NotImplementedError


class _NeighborAggregationDetector(_SupervisedTreeDetector):
    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        hops: int = 2,
        aggregators: Sequence[str] = ("mean", "max", "std"),
        contamination: float = 0.01,
        random_state: int | None = None,
        params: dict[str, Any] | None = None,
    ):
        super().__init__(
            level=level, contamination=contamination, random_state=random_state, params=params
        )
        self.hops = hops
        self.aggregators = tuple(aggregators)

    def _fit_features(self, graph: Any) -> np.ndarray:
        g = ensure_node_features(as_graph(graph))
        self.aggregation_ = NeighborAggregation(hops=self.hops, aggregators=self.aggregators)
        self.aggregation_.fit(g)
        if self.level == "edge":
            self._names = self.aggregation_.edge_feature_names(g)
            return self.aggregation_.transform_edges(g)
        self._names = self.aggregation_.feature_names_
        return self.aggregation_.transform(g)

    def _features(self, graph: Any) -> np.ndarray:
        g = ensure_node_features(as_graph(graph))
        if self.level == "edge":
            return self.aggregation_.transform_edges(g)
        return self.aggregation_.transform(g)

    def _feature_names(self) -> list[str]:
        return self._names


class XGBGraph(_NeighborAggregationDetector):
    """XGBoost on k-hop neighbor-aggregated features. GADBench's rank-1 detector."""

    def _make_estimator(self, y_labeled: np.ndarray) -> Any:
        _guard_darwin_omp()
        from xgboost import XGBClassifier

        pos = int((y_labeled == 1).sum())
        neg = int((y_labeled == 0).sum())
        kw: dict[str, Any] = dict(
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=self.random_state,
        )
        kw.update(self.params or {})
        return XGBClassifier(**kw)


class RFGraph(_NeighborAggregationDetector):
    """Random forest on k-hop neighbor-aggregated features. GADBench's rank-3 detector."""

    def _make_estimator(self, y_labeled: np.ndarray) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        kw: dict[str, Any] = dict(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        kw.update(self.params or {})
        return RandomForestClassifier(**kw)


class FlatBaseline(_SupervisedTreeDetector):
    """The same tree model on raw node features with no graph. Runs in every benchmark;
    when a graph detector cannot beat this, graphspot says so loudly.
    """

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        estimator: Literal["xgb", "rf"] = "xgb",
        contamination: float = 0.01,
        random_state: int | None = None,
        params: dict[str, Any] | None = None,
    ):
        super().__init__(
            level=level, contamination=contamination, random_state=random_state, params=params
        )
        if estimator not in ("xgb", "rf"):
            raise ValueError(f"estimator must be 'xgb' or 'rf', got {estimator!r}")
        self.estimator = estimator

    def _fit_features(self, graph: Any) -> np.ndarray:
        g = ensure_node_features(as_graph(graph))
        self.n_features_in_ = g.x.shape[1]
        base = g.feature_names or [f"f{i}" for i in range(self.n_features_in_)]
        if self.level == "edge":
            self._names = [f"src_{n}" for n in base] + [f"dst_{n}" for n in base]
            if g.edge_attr is not None:
                self._names += g.edge_feature_names or [
                    f"e{i}" for i in range(g.edge_attr.shape[1])
                ]
        else:
            self._names = list(base)
        return self._flat_features(g)

    def _features(self, graph: Any) -> np.ndarray:
        g = ensure_node_features(as_graph(graph))
        if g.x.shape[1] != self.n_features_in_:
            raise ValueError(f"Graph must carry {self.n_features_in_} node features")
        return self._flat_features(g)

    def _flat_features(self, g: Any) -> np.ndarray:
        if self.level == "node":
            return g.x
        src, dst = g.edge_index[0], g.edge_index[1]
        blocks = [g.x[src], g.x[dst]]
        if g.edge_attr is not None:
            blocks.append(g.edge_attr)
        return np.hstack(blocks)

    def _feature_names(self) -> list[str]:
        return self._names

    def _make_estimator(self, y_labeled: np.ndarray) -> Any:
        if self.estimator == "xgb":
            _guard_darwin_omp()
            from xgboost import XGBClassifier

            pos = int((y_labeled == 1).sum())
            neg = int((y_labeled == 0).sum())
            kw: dict[str, Any] = dict(
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
                scale_pos_weight=neg / pos,
                random_state=self.random_state,
            )
            kw.update(self.params or {})
            return XGBClassifier(**kw)
        from sklearn.ensemble import RandomForestClassifier

        kw = dict(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        kw.update(self.params or {})
        return RandomForestClassifier(**kw)
