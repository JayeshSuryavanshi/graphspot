from __future__ import annotations

import numpy as np


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k: int | None = None,
    baseline_scores: np.ndarray | None = None,
) -> dict[str, float | str]:
    """AUPRC-first evaluation. AUROC is reported but is a poor headline under class
    imbalance. `k` defaults to the number of positives (so prec@k == rec@k there).
    Pass `baseline_scores` from a flat (no-graph) model to get the graph lift.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    mask = y_true >= 0
    y, s = y_true[mask], scores[mask]
    if k is None:
        k = int((y == 1).sum())
    top = np.argsort(s)[::-1][:k]
    hits = float((y[top] == 1).sum())
    out: dict[str, float | str] = {
        "auprc": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        "prec_at_k": hits / k if k else 0.0,
        "rec_at_k": hits / max(int((y == 1).sum()), 1),
        "k": float(k),
    }
    if baseline_scores is not None:
        b = np.asarray(baseline_scores, dtype=np.float64)[mask]
        flat = float(average_precision_score(y, b))
        out["flat_baseline_auprc"] = flat
        lift = float(out["auprc"]) - flat
        out["graph_lift"] = f"{lift:+.3f} AUPRC"
        if lift <= 0:
            import warnings

            warnings.warn(
                f"Graph model does not beat the flat baseline "
                f"(AUPRC {out['auprc']:.3f} vs {flat:.3f}). "
                "The graph structure may carry no signal here; prefer the simpler model.",
                stacklevel=2,
            )
    return out


def evaluate_temporal(
    y_true: np.ndarray,
    scores: np.ndarray,
    times: np.ndarray,
    *,
    k_per_step: int | None = None,
) -> dict[str, object]:
    """Per-time-step evaluation for temporally ordered data. A single aggregate over a
    stream whose base rate shifts (Elliptic's step-43 market shutdown drops it 39x) is
    misleading, so the per-step table is the primary output and the summary is the mean
    over steps that contain both classes.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    times = np.asarray(times)
    rows: list[dict[str, float]] = []
    for t in np.unique(times):
        m = (times == t) & (y_true >= 0)
        y, s = y_true[m], scores[m]
        pos = int((y == 1).sum())
        row: dict[str, float] = {
            "step": float(t),
            "n_labeled": float(len(y)),
            "n_pos": float(pos),
            "base_rate": pos / len(y) if len(y) else 0.0,
        }
        if pos and pos < len(y):
            r = evaluate(y, s, k=k_per_step)
            row["auprc"] = float(r["auprc"])
            row["auroc"] = float(r["auroc"])
        rows.append(row)
    scored = [r for r in rows if "auprc" in r]
    return {
        "steps": rows,
        "mean_auprc": float(np.mean([r["auprc"] for r in scored])) if scored else float("nan"),
        "mean_auroc": float(np.mean([r["auroc"] for r in scored])) if scored else float("nan"),
        "n_scored_steps": len(scored),
    }


def base_rate_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    rates: tuple[float, ...] = (0.05, 0.01, 0.001),
    n_rep: int = 5,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Downsample positives to each target base rate and re-evaluate: how a detector
    holds up as anomalies get rarer, which is the regime production fraud lives in.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    mask = y_true >= 0
    y, s = y_true[mask], scores[mask]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    out = []
    for rate in rates:
        n_pos = int(round(rate * len(neg) / (1.0 - rate)))
        if n_pos < 1 or n_pos > len(pos):
            continue
        aps, recs = [], []
        for _ in range(n_rep):
            keep_pos = rng.choice(pos, size=n_pos, replace=False)
            idx = np.concatenate([keep_pos, neg])
            aps.append(average_precision_score(y[idx], s[idx]))
            top = idx[np.argsort(s[idx])[::-1][:n_pos]]
            recs.append(float((y[top] == 1).sum()) / n_pos)
        out.append(
            {
                "base_rate": rate,
                "n_pos": float(n_pos),
                "auprc": float(np.mean(aps)),
                "auprc_std": float(np.std(aps)),
                "rec_at_k": float(np.mean(recs)),
            }
        )
    return out
