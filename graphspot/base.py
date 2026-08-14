from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import numpy as np

from graphspot.graph import Graph


class BaseDetector(ABC):
    """Contract: `decision_function` scores graphs unseen at fit time and never raises
    NotImplementedError. Fitted attributes match PyOD's plural names exactly.
    """

    supported_levels: ClassVar[tuple[str, ...]] = ("node",)
    requires: ClassVar[tuple[str, ...]] = ()
    inductive: ClassVar[bool] = True

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        if level not in self.supported_levels:
            raise ValueError(
                f"{type(self).__name__} supports levels {self.supported_levels}, got {level!r}"
            )
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")
        self.level = level
        self.contamination = contamination
        self.random_state = random_state

    decision_scores_: np.ndarray
    labels_: np.ndarray
    threshold_: float

    @abstractmethod
    def fit(self, graph: Any, y: np.ndarray | None = None) -> BaseDetector: ...

    @abstractmethod
    def decision_function(self, graph: Any) -> np.ndarray: ...

    def fit_predict(self, graph: Any, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(graph, y).labels_

    def predict(self, graph: Any | None = None) -> np.ndarray:
        self._check_fitted()
        if graph is None:
            return self.labels_
        scores = self.decision_function(graph)
        return (scores > self.threshold_).astype(np.int64)

    def predict_proba(self, graph: Any | None = None, *, method: str = "linear") -> np.ndarray:
        self._check_fitted()
        if method != "linear":
            raise ValueError(f"Unknown method {method!r}")
        scores = self.decision_scores_ if graph is None else self.decision_function(graph)
        lo, hi = float(self.decision_scores_.min()), float(self.decision_scores_.max())
        p = np.clip((scores - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(scores)
        return np.column_stack([1.0 - p, p])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for cls in type(self).__mro__:
            if cls is object:
                continue
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if name in ("self", "args", "kwargs") or p.kind is p.VAR_KEYWORD:
                    continue
                if hasattr(self, name):
                    out[name] = getattr(self, name)
        return out

    def set_params(self, **params: Any) -> BaseDetector:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params().items()))
        return f"{type(self).__name__}({args})"

    def _finalize_fit(self, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        self.decision_scores_ = scores
        self.threshold_ = float(np.percentile(scores, 100.0 * (1.0 - self.contamination)))
        self.labels_ = (scores > self.threshold_).astype(np.int64)

    def _check_fitted(self) -> None:
        if not hasattr(self, "decision_scores_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")

    @staticmethod
    def _validate_labels(graph: Graph, y: np.ndarray | None, level: str) -> np.ndarray:
        n = graph.n_nodes if level == "node" else graph.n_edges
        if y is None:
            raise ValueError("This detector is supervised; pass y (use -1 for unlabeled)")
        y = np.asarray(y)
        if y.shape != (n,):
            raise ValueError(f"y has shape {y.shape}, expected ({n},) for level={level!r}")
        labeled = y >= 0
        if not labeled.any() or np.unique(y[labeled]).size < 2:
            raise ValueError("y needs at least one labeled example of each class (0 and 1)")
        return y
