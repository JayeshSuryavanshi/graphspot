from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import Any

_DETECTORS: list[dict[str, Any]] = [
    {"name": "XGBGraph", "levels": ("node", "edge"), "supervised": True, "requires": ()},
    {"name": "RFGraph", "levels": ("node", "edge"), "supervised": True, "requires": ()},
    {"name": "FlatBaseline", "levels": ("node", "edge"), "supervised": True, "requires": ()},
    {"name": "FlatUnsupervised", "levels": ("node",), "supervised": False, "requires": ()},
    {"name": "OddBall", "levels": ("node",), "supervised": False, "requires": ()},
    {"name": "Fraudar", "levels": ("edge",), "supervised": False, "requires": ()},
    {"name": "BWGNN", "levels": ("node",), "supervised": True, "requires": ("torch",)},
]


def list_detectors() -> list[dict[str, Any]]:
    """Every detector, its levels, and whether it is usable in this environment.

    `available` reflects installed backends without importing them. `note` carries
    platform caveats, currently the macOS rule that torch and xgboost cannot share
    one process (dual OpenMP runtimes).
    """
    out = []
    for spec in _DETECTORS:
        entry = dict(spec)
        missing = [r for r in spec["requires"] if find_spec(r) is None]
        entry["available"] = not missing
        if missing:
            entry["note"] = f"pip install 'graphspot[deep]' (missing: {', '.join(missing)})"
        elif "torch" in spec["requires"] and sys.platform == "darwin":
            entry["note"] = (
                "macOS: run in a separate process from xgboost detectors (dual OpenMP runtimes)"
            )
        else:
            entry["note"] = ""
        out.append(entry)
    return out
