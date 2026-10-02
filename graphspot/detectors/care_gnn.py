from __future__ import annotations

from math import ceil
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from graphspot.base import BaseDetector
from graphspot.detectors.bwgnn import _require_torch, _simple_undirected
from graphspot.graph import Graph, as_graph

_P_INIT = 0.5  # initial keep fraction p_r for every relation
_RL_WINDOW = 10  # the bandit freezes once |sum of the last 10 rewards| <= _RL_BAND
_RL_BAND = 2


class CAREGNN(BaseDetector):
    """CAmouflage-REsistant GNN (Dou, Liu, Sun, Deng, Peng, Yu: "Enhancing Graph
    Neural Network-based Fraud Detectors against Camouflaged Fraudsters", CIKM 2020),
    clean-room (no reference code consulted).

    Fraudsters camouflage by connecting to benign nodes and by mimicking benign
    features, so a GNN that averages every neighbor averages the camouflage in.
    CARE-GNN keeps, per relation, only the neighbors that look like the center
    under a label-aware similarity, and learns how many to keep:

    1. Label-aware similarity. A one-layer label predictor (softmax over a Linear
       on the raw features) gives each node P(fraud). The distance between v and u
       is |P_v - P_u| (half the L1 distance of the two class-probability vectors).
    2. Top-p selection. For each relation r, each center keeps its
       ceil(p_r * deg_r) nearest neighbors; ties go to the lower node id.
    3. Intra-relation: h_{v,r} = ReLU(W_r mean of the kept neighbors' features).
       Inter-relation: z_v = ReLU(W [x_v || sum_r p_r h_{v,r}]), then a linear
       head gives P(fraud). The thresholds p_r double as the relation weights.
    4. Loss = CE(head) + `sim_weight` * CE(label predictor), on a balanced sample
       of the labeled nodes redrawn each epoch (every minority-class node plus as
       many from the majority), in minibatches of `batch_size`. `weight_decay` is
       the L2 term.
    5. Bandit thresholds. After each epoch, each p_r moves by +`rl_step` if the
       mean distance of the kept neighbors did not rise since the previous epoch,
       else by -`rl_step`, starting from 0.5 and clipped to [0, 1]. A relation's
       p_r freezes once |sum of its last 10 rewards| <= 2.

    Relations come from `graph.edge_type` (e.g. `load_yelpchi(relations=True)`),
    each made symmetric and binary; without `edge_type` the graph is one relation.
    At scoring time, relations are matched to the fitted ones by
    `relation_names` when both graphs carry names, else by code.

    graphspot choices, where the paper's PDF could not be read when this was
    written (the method above follows its published description and needs checking
    against its equations): one layer (the paper's default and best setting), so
    similarity and aggregation act on raw features; the rounding and tie rules in
    step 2; summing relations in step 3; the reward tie (+1), clipping, the freeze
    rule taking no step, and no step after the last epoch, so the stored
    thresholds are the ones the weights were trained with.

    Inductive: the label predictor needs only features, the thresholds are fixed
    fractions and the weights are shared, so `decision_function` scores any graph
    with no labels. `decision_scores_` comes from the same function.
    """

    supported_levels = ("node",)
    requires = ("torch",)
    inductive = True

    def __init__(
        self,
        *,
        level: Literal["node", "edge"] = "node",
        hidden: int = 64,
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        weight_decay: float = 1e-3,
        sim_weight: float = 2.0,
        rl_step: float = 0.02,
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(level=level, contamination=contamination, random_state=random_state)
        for name, value in (("hidden", hidden), ("epochs", epochs), ("batch_size", batch_size)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if sim_weight < 0:
            raise ValueError(f"sim_weight must be >= 0, got {sim_weight}")
        if not 0.0 <= rl_step < 1.0:
            raise ValueError(f"rl_step must be in [0, 1), got {rl_step}")
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.sim_weight = sim_weight
        self.rl_step = rl_step

    # ------------------------------------------------------------------ fit

    def fit(self, graph: Any, y: np.ndarray | None = None) -> CAREGNN:
        torch = _require_torch()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("CAREGNN needs node features (graph.x)")
        y = self._validate_labels(g, y, self.level)
        rels = self._fit_relations(g)
        n_rel = len(rels)
        x = g.x
        self.n_features_in_ = x.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self._build_model(x.shape[1], n_rel)
        params = [p for m in (self.sim_, self.rel_, self.out_) for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        ce = torch.nn.CrossEntropyLoss()
        x_t = torch.tensor(x, dtype=torch.float32)

        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        minority, majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)

        steps = np.zeros(n_rel, dtype=np.int64)  # p_r = _P_INIT + steps * rl_step, exact
        frozen = np.zeros(n_rel, dtype=bool)
        rewards: list[list[int]] = [[] for _ in range(n_rel)]
        g_prev = np.full(n_rel, np.nan)
        self.relation_p_history_ = np.zeros((self.epochs, n_rel))
        self.rl_stopped_epoch_ = np.full(n_rel, -1, dtype=np.int64)

        for epoch in range(self.epochs):
            p = self._thresholds(steps)
            self.relation_p_history_[epoch] = p
            sample = np.concatenate([minority, rng.choice(majority, len(minority), replace=False)])
            sample = sample[rng.permutation(len(sample))]

            # selection uses the label predictor as of the start of the epoch
            p_fraud = self._fraud_prob(x_t)
            aggs, g_cur = [], np.full(n_rel, np.nan)
            for r, adj in enumerate(rels):
                agg, kept_dist = self._select_mean(adj, sample, p_fraud, p[r], x)
                aggs.append(torch.tensor(agg, dtype=torch.float32))
                if kept_dist.size:
                    g_cur[r] = kept_dist.mean()

            self._train_mode(True)
            for start in range(0, len(sample), self.batch_size):
                b = torch.arange(start, min(start + self.batch_size, len(sample)))
                nodes = torch.as_tensor(sample[start : start + self.batch_size])
                target = torch.as_tensor(y[nodes.numpy()], dtype=torch.long)
                logits = self._forward(x_t[nodes], [a[b] for a in aggs], p)
                loss = ce(logits, target) + self.sim_weight * ce(self.sim_(x_t[nodes]), target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self._train_mode(False)

            if epoch < self.epochs - 1 and self.rl_step > 0:
                self._bandit_step(steps, frozen, rewards, g_prev, g_cur, epoch)
            g_prev = np.where(np.isnan(g_cur), g_prev, g_cur)

        self.relation_p_ = self._thresholds(steps)
        self._finalize_fit(self._score(g, rels))
        return self

    def _bandit_step(self, steps, frozen, rewards, g_prev, g_cur, epoch) -> None:
        """One greedy Bernoulli-bandit move per relation; mutates its inputs."""
        lo, hi = -ceil(_P_INIT / self.rl_step), ceil((1.0 - _P_INIT) / self.rl_step)
        for r in range(len(steps)):
            if frozen[r] or not (np.isfinite(g_prev[r]) and np.isfinite(g_cur[r])):
                continue
            reward = 1 if g_prev[r] - g_cur[r] >= 0 else -1
            rewards[r].append(reward)
            recent = rewards[r][-_RL_WINDOW:]
            if len(recent) == _RL_WINDOW and abs(sum(recent)) <= _RL_BAND:
                frozen[r] = True
                self.rl_stopped_epoch_[r] = epoch
                continue
            steps[r] = min(max(steps[r] + reward, lo), hi)

    def _thresholds(self, steps: np.ndarray) -> np.ndarray:
        return np.clip(_P_INIT + steps * self.rl_step, 0.0, 1.0)

    # -------------------------------------------------------------- scoring

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        g = as_graph(graph)
        if g.x is None:
            raise ValueError("CAREGNN needs node features (graph.x)")
        if g.x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"graph has {g.x.shape[1]} features, CAREGNN was fit on {self.n_features_in_}"
            )
        return self._score(g, self._score_relations(g))

    def _score(self, g: Graph, rels: list[sp.csr_matrix]) -> np.ndarray:
        """The single scoring path, shared by fit and decision_function."""
        torch = _require_torch()
        x_t = torch.tensor(g.x, dtype=torch.float32)
        p_fraud = self._fraud_prob(x_t)
        rows = np.arange(g.n_nodes)
        aggs = [
            torch.tensor(self._select_mean(adj, rows, p_fraud, p, g.x)[0], dtype=torch.float32)
            for adj, p in zip(rels, self.relation_p_, strict=True)
        ]
        self._train_mode(False)
        with torch.no_grad():
            logits = self._forward(x_t, aggs, self.relation_p_)
            return torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float64)

    # --------------------------------------------------------------- model

    def _build_model(self, n_features: int, n_rel: int) -> None:
        """Built-in torch containers only, so the fitted detector pickles."""
        torch = _require_torch()
        self.sim_ = torch.nn.Linear(n_features, 2)
        self.rel_ = torch.nn.ModuleList(
            [torch.nn.Linear(n_features, self.hidden, bias=False) for _ in range(n_rel)]
        )
        self.out_ = torch.nn.Sequential(
            torch.nn.Linear(n_features + self.hidden, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 2),
        )

    def _train_mode(self, on: bool) -> None:
        for m in (self.sim_, self.rel_, self.out_):
            m.train(on)

    def _forward(self, x_c, aggs, p):
        torch = _require_torch()
        mixed = sum(
            float(p_r) * torch.relu(lin(a)) for p_r, lin, a in zip(p, self.rel_, aggs, strict=True)
        )
        return self.out_(torch.cat([x_c, mixed], dim=1))

    def _fraud_prob(self, x_t) -> np.ndarray:
        torch = _require_torch()
        with torch.no_grad():
            return torch.softmax(self.sim_(x_t), dim=1)[:, 1].numpy().astype(np.float64)

    @staticmethod
    def _select_mean(adj, rows, p_fraud, keep_p, x):
        """Mean of the features of each center's ceil(keep_p * deg) nearest
        neighbors under `adj`, nearest by |p_fraud| difference, ties to the lower
        node id. Returns the (len(rows), d) means (zero for a center with no kept
        neighbor) and the distances of every kept edge.
        """
        sub = adj[rows]
        deg = np.diff(sub.indptr)
        k = np.minimum(np.ceil(keep_p * deg - 1e-9), deg).astype(np.int64)
        center = np.repeat(np.arange(len(rows)), deg)
        dist = np.abs(p_fraud[rows][center] - p_fraud[sub.indices])
        order = np.lexsort((sub.indices, dist, center))
        center, nbr, dist = center[order], sub.indices[order], dist[order]
        keep = np.arange(len(order)) - sub.indptr[:-1][center] < k[center]
        center, nbr, dist = center[keep], nbr[keep], dist[keep]
        weights = sp.csr_matrix((1.0 / k[center], (center, nbr)), shape=(len(rows), adj.shape[0]))
        return np.asarray(weights @ x), dist

    # ----------------------------------------------------------- relations

    def _fit_relations(self, g: Graph) -> list[sp.csr_matrix]:
        if g.edge_type is None:
            self.relation_names_: list[str] = []
            return [_simple_undirected(g.adj)]
        if (g.edge_type < 0).any():
            raise ValueError("edge_type has negative codes (edges with no relation)")
        n_rel = max(int(g.edge_type.max()) + 1, len(g.relation_names))
        self.relation_names_ = list(g.relation_names)
        return self._split(g, g.edge_type, n_rel)

    def _score_relations(self, g: Graph) -> list[sp.csr_matrix]:
        n_rel = len(self.relation_p_)
        if n_rel == 1 and not self.relation_names_:
            return [_simple_undirected(g.adj)]  # fit as one relation: always the union
        if g.edge_type is None:
            if n_rel == 1:
                return [_simple_undirected(g.adj)]
            raise ValueError(
                f"CAREGNN was fit on {n_rel} relations {self.relation_names_}; "
                "this graph has no edge_type"
            )
        codes = g.edge_type
        if (codes < 0).any():
            raise ValueError("edge_type has negative codes (edges with no relation)")
        if self.relation_names_ and g.relation_names:
            unknown = sorted(set(g.relation_names) - set(self.relation_names_))
            if unknown:
                raise ValueError(f"unknown relation(s) {unknown}; fit on {self.relation_names_}")
            to_fit = np.array([self.relation_names_.index(r) for r in g.relation_names])
            codes = to_fit[codes]
        elif codes.size and codes.max() >= n_rel:
            raise ValueError(f"edge_type code {codes.max()} but CAREGNN was fit on {n_rel}")
        return self._split(g, codes, n_rel)

    @staticmethod
    def _split(g: Graph, codes: np.ndarray, n_rel: int) -> list[sp.csr_matrix]:
        n = g.n_nodes
        out = []
        for r in range(n_rel):
            ei = g.edge_index[:, codes == r]
            adj = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
            out.append(_simple_undirected(adj))
        return out
