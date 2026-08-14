from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from graphspot.graph import Graph

_CACHE = Path.home() / ".cache" / "graphspot"

ELLIPTIC_LICENSE_NOTE = (
    "Elliptic is CC BY-NC-ND 4.0: NonCommercial (unusable for employer work) and "
    "NoDerivatives (graphspot downloads from the upstream mirror and never redistributes, "
    "including converted files). Pass accept_license=True to acknowledge these terms."
)


@dataclass(frozen=True)
class Provenance:
    name: str
    label_type: str
    label_source: str
    license: str
    redistributable: bool
    auto_download: bool


def _download(url: str, sha256: str | None = None) -> bytes:
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if sha256 is not None and digest != sha256:
        raise RuntimeError(f"{url} sha256 mismatch: expected {sha256}, got {digest}")
    return payload


def _cache_root(root: Path | str | None) -> Path:
    out = Path(root) if root else _CACHE
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------- DGL fraud .mat pair

_MAT_SOURCES = {
    "yelpchi": {
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
        "mat": "YelpChi.mat",
        "label_type": "proxy",
        "label_source": "Yelp filtered-review flag, Rayana & Akoglu 2015 via Dou et al. 2020",
        "unlabeled_prefix": 0,
    },
    "amazon": {
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
        "mat": "Amazon.mat",
        "label_type": "proxy",
        "label_source": "helpful-vote ratio thresholding, Dou et al. 2020",
        "unlabeled_prefix": 3305,
    },
}


def _load_fraud_mat(name: str, root: Path | str | None) -> Graph:
    src = _MAT_SOURCES[name]
    cache = _cache_root(root)
    mat_path = cache / src["mat"]
    if not mat_path.exists():
        payload = _download(src["url"])
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extract(src["mat"], cache)
    mat = scipy.io.loadmat(mat_path)
    adj = sp.csr_matrix(mat["homo"])
    adj.data[:] = 1.0
    features = mat["features"]
    x = features.toarray() if sp.issparse(features) else np.asarray(features)
    labels = np.asarray(mat["label"]).ravel().astype(np.int64)
    if src["unlabeled_prefix"]:
        # Standard protocol since Dou et al. 2020: these leading nodes carry no reliable
        # label (zero positives among them) and are excluded from training and evaluation.
        labels[: src["unlabeled_prefix"]] = -1
    g = Graph(adj=adj, x=x.astype(np.float64), node_labels=labels)
    g.provenance = Provenance(  # type: ignore[attr-defined]
        name=name,
        label_type=src["label_type"],
        label_source=src["label_source"],
        license="unverified upstream terms; linked, never redistributed",
        redistributable=False,
        auto_download=True,
    )
    return g


def load_yelpchi(root: Path | str | None = None) -> Graph:
    """YelpChi review-fraud graph (45,954 nodes). Homogeneous union of the 3 relations."""
    return _load_fraud_mat("yelpchi", root)


def load_amazon(root: Path | str | None = None) -> Graph:
    """Amazon review-fraud graph (11,944 nodes). Homogeneous union of the 3 relations.

    The first 3,305 nodes are unlabeled by convention and carry ``node_labels == -1``.
    """
    return _load_fraud_mat("amazon", root)


# ------------------------------------------------- yandex heterophilous .npz pair (MIT)

_NPZ_SOURCES = {
    "tolokers": {
        "url": "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz",
        "label_source": "Toloka crowdworkers banned in a project, Platonov et al. 2023",
    },
    "questions": {
        "url": "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/questions.npz",
        "label_source": "Yandex Q users who remained active, Platonov et al. 2023",
    },
}


def _graph_from_heterophilous_npz(path: Path, name: str, label_source: str) -> Graph:
    data = np.load(path)
    edges = data["edges"].astype(np.int64)
    n = data["node_features"].shape[0]
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0
    g = Graph(
        adj=adj,
        x=data["node_features"].astype(np.float64),
        node_labels=data["node_labels"].astype(np.int64),
    )
    g.split_masks = {  # type: ignore[attr-defined]
        "train": data["train_masks"],
        "val": data["val_masks"],
        "test": data["test_masks"],
    }
    g.provenance = Provenance(  # type: ignore[attr-defined]
        name=name,
        label_type="adjudicated",
        label_source=label_source,
        license="MIT (yandex-research/heterophilous-graphs)",
        redistributable=True,
        auto_download=True,
    )
    return g


def _load_heterophilous(name: str, root: Path | str | None) -> Graph:
    src = _NPZ_SOURCES[name]
    cache = _cache_root(root)
    path = cache / f"{name}.npz"
    if not path.exists():
        path.write_bytes(_download(src["url"]))
    return _graph_from_heterophilous_npz(path, name, src["label_source"])


def load_tolokers(root: Path | str | None = None) -> Graph:
    """Tolokers crowdworker graph (11,758 nodes, 21.8% positives). Ships 10 mask trials
    in ``graph.split_masks``."""
    return _load_heterophilous("tolokers", root)


def load_questions(root: Path | str | None = None) -> Graph:
    """Yandex Q user graph (48,921 nodes, 3.0% positives). Ships 10 mask trials in
    ``graph.split_masks``."""
    return _load_heterophilous("questions", root)


# ------------------------------------------------------------ Elliptic (CC BY-NC-ND)

_ELLIPTIC_BASE = "https://data.pyg.org/datasets/elliptic/"
_ELLIPTIC_FILES = (
    "elliptic_txs_features.csv.zip",
    "elliptic_txs_edgelist.csv.zip",
    "elliptic_txs_classes.csv.zip",
)


def load_elliptic(root: Path | str | None = None, *, accept_license: bool = False) -> Graph:
    """Elliptic Bitcoin transaction graph (203,769 nodes over 49 time steps).

    ``node_time`` carries each transaction's time step, enabling strict-inductive
    temporal evaluation. Labels: 1 illicit, 0 licit, -1 unknown.
    """
    if not accept_license:
        raise PermissionError(ELLIPTIC_LICENSE_NOTE)
    cache = _cache_root(root)
    parsed = cache / "elliptic_parsed.npz"
    if not parsed.exists():
        for fname in _ELLIPTIC_FILES:
            dest = cache / fname
            if not dest.exists():
                dest.write_bytes(_download(_ELLIPTIC_BASE + fname))
        feats = pd.read_csv(cache / _ELLIPTIC_FILES[0], header=None)
        classes = pd.read_csv(cache / _ELLIPTIC_FILES[2])
        edges = pd.read_csv(cache / _ELLIPTIC_FILES[1])
        tx_ids = feats[0].to_numpy(dtype=np.int64)
        pos = pd.Series(np.arange(len(tx_ids)), index=tx_ids)
        label_map = {"unknown": -1, "1": 1, "2": 0}
        labels = (
            classes.set_index("txId")["class"]
            .map(label_map)
            .reindex(tx_ids)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        np.savez_compressed(
            parsed,
            x=feats.iloc[:, 2:].to_numpy(dtype=np.float64),
            node_time=feats[1].to_numpy(dtype=np.float64),
            labels=labels,
            src=pos.loc[edges["txId1"]].to_numpy(dtype=np.int64),
            dst=pos.loc[edges["txId2"]].to_numpy(dtype=np.int64),
        )
    data = np.load(parsed)
    n = data["x"].shape[0]
    adj = sp.csr_matrix((np.ones(len(data["src"])), (data["src"], data["dst"])), shape=(n, n))
    g = Graph(
        adj=adj,
        x=data["x"],
        node_labels=data["labels"],
        node_time=data["node_time"],
    )
    g.provenance = Provenance(  # type: ignore[attr-defined]
        name="elliptic",
        label_type="proxy",
        label_source="Elliptic Ltd licit/illicit tags; 77% of nodes unknown",
        license="CC BY-NC-ND 4.0: NonCommercial, NoDerivatives; never redistribute",
        redistributable=False,
        auto_download=True,
    )
    return g


QUICK_LOADERS = {
    "yelpchi": load_yelpchi,
    "amazon": load_amazon,
    "tolokers": load_tolokers,
    "questions": load_questions,
}
