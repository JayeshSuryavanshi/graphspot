from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from graphspot.graph import Graph

_CACHE = Path.home() / ".cache" / "graphspot"

# sha256 of the upstream zips, pinned after first verified download; None = trust-on-first-use
_SOURCES = {
    "yelpchi": {
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
        "mat": "YelpChi.mat",
        "sha256": None,
        "label_type": "proxy",
        "label_source": "Yelp filtered-review flag, Rayana & Akoglu 2015 via Dou et al. 2020",
    },
    "amazon": {
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
        "mat": "Amazon.mat",
        "sha256": None,
        "label_type": "proxy",
        "label_source": "helpful-vote ratio thresholding, Dou et al. 2020",
        "unlabeled_prefix": 3305,
    },
}


@dataclass(frozen=True)
class Provenance:
    name: str
    label_type: str
    label_source: str
    license: str
    redistributable: bool
    auto_download: bool


def _fetch(name: str, root: Path | None) -> Path:
    src = _SOURCES[name]
    root = Path(root) if root else _CACHE
    root.mkdir(parents=True, exist_ok=True)
    mat_path = root / src["mat"]
    if mat_path.exists():
        return mat_path
    with urllib.request.urlopen(src["url"]) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if src["sha256"] is not None and digest != src["sha256"]:
        raise RuntimeError(f"{src['url']} sha256 mismatch: expected {src['sha256']}, got {digest}")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(src["mat"], root)
    return mat_path


def _load_fraud_mat(name: str, root: Path | None) -> Graph:
    mat = scipy.io.loadmat(_fetch(name, root))
    adj = sp.csr_matrix(mat["homo"])
    adj.data[:] = 1.0
    features = mat["features"]
    x = features.toarray() if sp.issparse(features) else np.asarray(features)
    labels = np.asarray(mat["label"]).ravel().astype(np.int64)
    prefix = _SOURCES[name].get("unlabeled_prefix", 0)
    if prefix:
        # Standard protocol since Dou et al. 2020: these leading nodes carry no reliable
        # label (zero positives among them) and are excluded from training and evaluation.
        labels[:prefix] = -1
    g = Graph(adj=adj, x=x.astype(np.float64), node_labels=labels)
    g.provenance = Provenance(  # type: ignore[attr-defined]
        name=name,
        label_type=_SOURCES[name]["label_type"],
        label_source=_SOURCES[name]["label_source"],
        license="unverified upstream terms; linked, never redistributed",
        redistributable=False,
        auto_download=True,
    )
    return g


def load_yelpchi(root: Path | str | None = None) -> Graph:
    """YelpChi review-fraud graph (45,954 nodes). Homogeneous union of the 3 relations."""
    return _load_fraud_mat("yelpchi", None if root is None else Path(root))


def load_amazon(root: Path | str | None = None) -> Graph:
    """Amazon review-fraud graph (11,944 nodes). Homogeneous union of the 3 relations."""
    return _load_fraud_mat("amazon", None if root is None else Path(root))
