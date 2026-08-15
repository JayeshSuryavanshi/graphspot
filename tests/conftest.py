from __future__ import annotations

import sys


def pytest_ignore_collect(collection_path, config):
    """On macOS, torch and xgboost cannot coexist in one process (dual OpenMP
    runtimes: segfault or deadlock, see graphspot.detectors.bwgnn._DARWIN_OMP_MSG).
    Importing test_bwgnn at collection would load torch into the shared suite
    process and poison every xgboost test, so it is only collected when named
    explicitly: `pytest tests/test_bwgnn.py`. Linux collects everything, and the
    deep CI job proves coexistence there.
    """
    if sys.platform != "darwin":
        return None
    if collection_path.name != "test_bwgnn.py":
        return None
    if any("test_bwgnn" in str(arg) for arg in config.invocation_params.args):
        return None
    return True
