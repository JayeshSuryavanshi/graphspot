from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from graphspot.base import BaseDetector
from graphspot.graph import Graph, as_graph


@dataclass
class Block:
    """One dense block: source-side and target-side node positions, and its density."""

    rows: np.ndarray
    cols: np.ndarray
    density: float
    n_edges: int
    row_ids: list | None = None
    col_ids: list | None = None
    nodes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = np.union1d(self.rows, self.cols)


def _densest_block(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Greedy shave from Hooi et al. (KDD 2016): repeatedly remove the entity with the
    smallest weighted degree, tracking the average-degree objective g = f/|S|; return the
    prefix that maximised g. Source and target roles of a node are separate entities, so
    entity ids are src_node for rows and n + dst_node for columns. Lazy heap entries are
    invalidated by comparing against the current priority.
    """
    entity = np.r_[src, n + dst]
    other = np.r_[n + dst, src]
    edge_w = np.r_[weights, weights]
    order = np.argsort(entity, kind="stable")
    entity, other, edge_w = entity[order], other[order], edge_w[order]
    edge_of = np.r_[np.arange(len(src)), np.arange(len(src))][order]

    starts = np.searchsorted(entity, np.arange(2 * n))
    ends = np.searchsorted(entity, np.arange(2 * n) + 1)
    priority = np.zeros(2 * n)
    np.add.at(priority, np.r_[src, n + dst], np.r_[weights, weights])

    alive_entity = np.zeros(2 * n, dtype=bool)
    alive_entity[np.r_[src, n + dst]] = True
    alive_edge = np.ones(len(src), dtype=bool)
    n_alive = int(alive_entity.sum())
    f = float(weights.sum())

    heap = [(priority[e], e) for e in np.flatnonzero(alive_entity)]
    heapq.heapify(heap)

    best_g = f / n_alive if n_alive else 0.0
    removal_order: list[int] = []
    best_step = 0

    while n_alive > 0:
        p, e = heapq.heappop(heap)
        if not alive_entity[e] or p != priority[e]:
            continue
        alive_entity[e] = False
        n_alive -= 1
        for k in range(starts[e], ends[e]):
            eid = edge_of[k]
            if not alive_edge[eid]:
                continue
            alive_edge[eid] = False
            f -= edge_w[k]
            o = other[k]
            if alive_entity[o]:
                priority[o] -= edge_w[k]
                heapq.heappush(heap, (priority[o], o))
        removal_order.append(e)
        if n_alive > 0 and f / n_alive > best_g:
            best_g = f / n_alive
            best_step = len(removal_order)

    survivors = np.ones(2 * n, dtype=bool)
    survivors[:] = False
    survivors[np.r_[src, n + dst]] = True
    survivors[removal_order[:best_step]] = False
    kept = np.flatnonzero(survivors)
    rows = kept[kept < n]
    cols = kept[kept >= n] - n
    inside = np.isin(src, rows) & np.isin(dst, cols)
    return rows, cols, best_g, int(inside.sum())


class Fraudar(BaseDetector):
    """FRAUDAR (Hooi et al., KDD 2016): camouflage-resistant dense-block detection.

    Label-free and feature-free. Edges are down-weighted by 1 / log(x + 5) of their
    target's degree, so a ring cannot hide by also touching popular targets. `fit`
    finds `n_blocks` dense blocks by greedy shaving with the 1/2-approximation
    guarantee; `decision_function` scores an edge by the density of the fitted block
    whose row set contains its source and whose column set contains its target
    (0 otherwise), so known rings are flagged when they act again in new data.

    Implemented from the paper alone; see LICENSE-THIRD-PARTY.
    """

    supported_levels = ("edge",)
    inductive = True

    def __init__(
        self,
        *,
        level: str = "edge",
        n_blocks: int = 1,
        contamination: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(
            level=level,  # type: ignore[arg-type]
            contamination=contamination,
            random_state=random_state,
        )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.n_blocks = n_blocks

    def fit(self, graph: Any, y: np.ndarray | None = None) -> Fraudar:
        g = as_graph(graph)
        if g.edge_index is None or g.n_edges == 0:
            raise ValueError("Fraudar needs a graph with edges")
        src = g.edge_index[0].copy()
        dst = g.edge_index[1].copy()
        n = g.n_nodes

        self.blocks_: list[Block] = []
        keep = np.ones(len(src), dtype=bool)
        self._edge_block = np.full(len(src), -1, dtype=np.int64)
        for _ in range(self.n_blocks):
            if not keep.any():
                break
            s, d = src[keep], dst[keep]
            col_deg = np.bincount(d, minlength=n).astype(np.float64)
            weights = 1.0 / np.log(col_deg[d] + 5.0)
            rows, cols, density, n_edges = _densest_block(s, d, weights, n)
            if len(rows) == 0 or len(cols) == 0:
                break
            block = Block(rows=rows, cols=cols, density=density, n_edges=n_edges)
            if g.node_index is not None:
                block.row_ids = list(g.node_index[rows])
                block.col_ids = list(g.node_index[cols])
            self.blocks_.append(block)
            inside = keep & np.isin(src, rows) & np.isin(dst, cols)
            self._edge_block[inside] = len(self.blocks_) - 1
            keep &= ~inside

        self._finalize_fit(self._score_edges(g))
        return self

    def explain(self, idx: int | None = None, k: int = 5) -> Any:
        """Without `idx`: the top-k fitted blocks as dicts. With `idx`: the block
        containing that fit-graph edge, or None if it belongs to no block."""
        self._check_fitted()
        if idx is not None:
            b = int(self._edge_block[idx])
            if b < 0:
                return None
            block = self.blocks_[b]
            return {
                "block": b,
                "density": block.density,
                "n_rows": len(block.rows),
                "n_cols": len(block.cols),
                "n_edges": block.n_edges,
                "row_ids": block.row_ids,
                "col_ids": block.col_ids,
            }
        ranked = sorted(self.blocks_, key=lambda blk: -blk.density)[:k]
        return [
            {
                "density": blk.density,
                "n_rows": len(blk.rows),
                "n_cols": len(blk.cols),
                "n_edges": blk.n_edges,
                "row_ids": blk.row_ids,
                "col_ids": blk.col_ids,
            }
            for blk in ranked
        ]

    def decision_function(self, graph: Any) -> np.ndarray:
        self._check_fitted()
        return self._score_edges(as_graph(graph))

    def _score_edges(self, g: Graph) -> np.ndarray:
        src, dst = g.edge_index[0], g.edge_index[1]
        scores = np.zeros(len(src))
        for block in self.blocks_:
            if g.node_index is not None and block.row_ids is not None:
                rows = g.node_index.get_indexer(block.row_ids)
                cols = g.node_index.get_indexer(block.col_ids)
                rows, cols = rows[rows >= 0], cols[cols >= 0]
            else:
                rows, cols = block.rows, block.cols
            inside = np.isin(src, rows) & np.isin(dst, cols)
            scores[inside] = np.maximum(scores[inside], block.density)
        return scores
