from __future__ import annotations

from math import comb
from typing import Any, Literal

import numpy as np

from graphspot.base import BaseDetector
from graphspot.graph import Graph, as_graph

_DARWIN_OMP_MSG = (
    "torch and xgboost cannot share one process on macOS: each loads its own OpenMP "
    "runtime, and the mix segfaults (torch first) or deadlocks (xgboost first). Run "
    "deep detectors and tree detectors in separate processes on macOS. Linux is "
    "unaffected; CI proves coexistence there."
)


def _require_torch():
    import sys

    if sys.platform == "darwin" and "xgboost" in sys.modules:
        raise RuntimeError(_DARWIN_OMP_MSG)
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            "graphspot's deep detectors (BWGNN, GHRN, CAREGNN) need torch: "
            "pip install 'graphspot[deep]'"
        ) from err
    return torch


def _simple_undirected(adj):
    """Symmetric, binary, self-loop free copy of `adj` (csr)."""
    import scipy.sparse as sp

    adj = sp.csr_matrix(adj + adj.T)
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


class BWGNN(BaseDetector):
    """Beta Wavelet GNN (Tang, Li, Li, Gao, Li: "Rethinking Graph Neural Networks for
    Anomaly Detection", ICML 2022), implemented clean-room from the paper.

    Anomalies shift a graph signal's spectral energy toward high frequencies, so a
    low-pass GNN erases exactly the evidence. BWGNN filters the (self-loop free,
    symmetrically normalized) Laplacian through the Beta wavelet bank
    W_{p,q} = (1 / (2 B(p+1, q+1))) (L/2)^p (I - L/2)^q with p + q = `order`,
    which gives `order`+1 band-pass views computed with sparse matmuls only, no
    eigendecomposition. Each view of the MLP-transformed features is concatenated and
    classified with class-weighted cross entropy.

    Inductive: `decision_function` runs a full forward pass on the new graph's own
    Laplacian. Full-batch by design; a polynomial filter of order C has an exact C-hop
    receptive field, so cutting neighborhoods (sampling) silently changes the filter.
    Torch only, no torch_geometric: the model is sparse matmuls end to end.
    """

    supported_levels = ("node",)
    requires = ("torch",)
    inductive = True

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        order: int = 2,
        hidden: int = 64,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(level=level, contamination=contamination, random_state=random_state)
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order = order
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def _laplacian(self, g: Graph):
        return self._laplacian_from_adj(_simple_undirected(g.adj))

    @staticmethod
    def _laplacian_from_adj(adj):
        """I - D^-1/2 A D^-1/2 for a symmetric, binary, self-loop free `adj`."""
        torch = _require_torch()
        import scipy.sparse as sp

        n = adj.shape[0]
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
        norm = sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)
        lap = (sp.eye(n) - norm).tocoo()
        idx = torch.tensor(np.vstack([lap.row, lap.col]), dtype=torch.long)
        val = torch.tensor(lap.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    def _filter_bank(self, lap, h):
        """Apply every Beta wavelet to h. (L/2)^p (I - L/2)^q h is computed by
        repeated sparse matmuls; the constants make the bank sum to (order+1)/2 * I,
        which the tests pin exactly.
        """
        torch = _require_torch()
        c = self.order
        half = lambda m: 0.5 * torch.sparse.mm(lap, m)  # noqa: E731

        powers = [h]
        for _ in range(c):
            powers.append(half(powers[-1]))
        outs = []
        for p in range(c + 1):
            q = c - p
            term = powers[p]
            for _ in range(q):
                term = term - half(term)
            beta_inv = (c + 1) * comb(c, p)
            outs.append(0.5 * beta_inv * term)
        return outs

    def _build_model(self, n_features: int) -> None:
        """Two plain Sequential blocks; the wavelet bank sits between them in
        `_forward`. No custom Module subclass, so the fitted detector pickles.
        """
        torch = _require_torch()
        self.pre_ = torch.nn.Sequential(
            torch.nn.Linear(n_features, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden),
        )
        self.post_ = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden * (self.order + 1), self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 2),
        )

    def _forward(self, lap, x):
        torch = _require_torch()
        h = self.pre_(x)
        return self.post_(torch.cat(self._filter_bank(lap, h), dim=1))

    def fit(self, graph: Any, y: np.ndarray | None = None) -> BWGNN:
        _require_torch()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError(f"{type(self).__name__} needs node features (graph.x)")
        y = self._validate_labels(g, y, self.level)
        self._train(self._laplacian(g), g.x, y)
        self._finalize_fit(self._forward_scores(g))
        return self

    def _train(self, lap, x_np: np.ndarray, y: np.ndarray) -> None:
        """Fresh model, seeded, trained full-batch on the labeled nodes of `y`.
        Sets `pre_` and `post_`.
        """
        torch = _require_torch()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        x = torch.tensor(x_np, dtype=torch.float32)
        labeled = np.flatnonzero(y >= 0)
        target = torch.tensor(y[labeled], dtype=torch.long)
        pos = int((y[labeled] == 1).sum())
        neg = len(labeled) - pos
        weight = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32)
        mask = torch.tensor(labeled, dtype=torch.long)

        self._build_model(x_np.shape[1])
        params = list(self.pre_.parameters()) + list(self.post_.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.pre_.train()
        self.post_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._forward(lap, x)
            loss = loss_fn(logits[mask], target)
            loss.backward()
            opt.step()

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        return self._forward_scores(as_graph(graph))

    def _forward_scores(self, g: Graph) -> np.ndarray:
        if g.x is None:
            raise ValueError(f"{type(self).__name__} needs node features (graph.x)")
        return self._probs(self._laplacian(g), g.x)

    def _probs(self, lap, x_np: np.ndarray) -> np.ndarray:
        """P(anomaly) per node under the current `pre_`/`post_`, in eval mode."""
        torch = _require_torch()
        x = torch.tensor(x_np, dtype=torch.float32)
        self.pre_.eval()
        self.post_.eval()
        with torch.no_grad():
            probs = torch.softmax(self._forward(lap, x), dim=1)[:, 1]
        return probs.numpy().astype(np.float64)
