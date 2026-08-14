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
