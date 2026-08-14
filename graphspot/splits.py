from __future__ import annotations

import numpy as np


def stratified_split(
    y: np.ndarray,
    *,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded stratified train/val/test indices over labeled nodes (``y >= 0``).

    Unlabeled nodes (``y == -1``) belong to no split; detectors still see them as
    graph structure, never as supervision or evaluation targets.
    """
    if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
        raise ValueError(f"invalid sizes: train={train_size}, val={val_size}")
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(y[y >= 0]):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_train = int(round(train_size * len(idx)))
        n_val = int(round(val_size * len(idx)))
        train.append(idx[:n_train])
        val.append(idx[n_train : n_train + n_val])
        test.append(idx[n_train + n_val :])
    rng2 = np.random.default_rng(seed + 1)
    out = []
    for part in (train, val, test):
        arr = np.concatenate(part)
        rng2.shuffle(arr)
        out.append(arr)
    return out[0], out[1], out[2]


def semi_supervised_split(
    y: np.ndarray,
    *,
    n_pos: int = 20,
    n_neg: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GADBench's label-scarce regime: 100 training labels (20 positive, 80 negative),
    remaining labeled nodes split evenly into val and test.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) < n_pos or len(neg) < n_neg:
        raise ValueError(
            f"need {n_pos} positives and {n_neg} negatives, have {len(pos)} and {len(neg)}"
        )
    rng.shuffle(pos)
    rng.shuffle(neg)
    train = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rest = np.concatenate([pos[n_pos:], neg[n_neg:]])
    rng.shuffle(rest)
    half = len(rest) // 2
    return train, rest[:half], rest[half:]


def train_labels(y: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Label vector exposing only the training split; everything else is -1."""
    out = np.full(len(y), -1, dtype=np.int64)
    out[train_idx] = np.asarray(y)[train_idx]
    return out
