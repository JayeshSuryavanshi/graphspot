from __future__ import annotations

import sys

DEEP_TEST_FILES = ("test_bwgnn.py", "test_ghrn.py")


def pytest_ignore_collect(collection_path, config):
    """On macOS, torch and xgboost cannot coexist in one process (dual OpenMP
    runtimes: segfault or deadlock, see graphspot.detectors.bwgnn._DARWIN_OMP_MSG).
    Importing a torch test file at collection would load torch into the shared suite
    process and poison every xgboost test, so each is only collected when named
    explicitly: `pytest tests/test_bwgnn.py tests/test_ghrn.py`. Linux collects
    everything, and the deep CI job proves coexistence there.
    """
    if sys.platform != "darwin":
        return None
    if collection_path.name not in DEEP_TEST_FILES:
        return None
    if any(collection_path.stem in str(arg) for arg in config.invocation_params.args):
        return None
    return True
