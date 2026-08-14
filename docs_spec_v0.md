# graphspot v0 Build Spec

## 1. NAME

**`graphspot`**. Verified live 2026-08-14: `pypi.org/pypi/graphspot/json` returns 404, `api.github.com/users/graphspot` returns 404, `gh search repos "graphspot"` returns zero results. Clean on all three namespaces.

Runner-up: **`gadscan`** (also verified free: PyPI 404, GitHub user 404, single irrelevant collision `bumie-e/GADSCandyStoreApp`, 0 stars). Reject `gadx` (GitHub user namespace is taken) and `anograph` (collides with Stream-AD/AnoGraph, AAAI'22, same field).

Claim the GitHub org `graphspot` and the PyPI name today, before writing code.

## 2. POSITIONING

graphspot is a graph anomaly detection library for people who have to score things they have never seen before. It installs in five seconds with no PyTorch, takes a pandas transaction dataframe or a scipy sparse matrix, and returns calibrated anomaly scores for nodes **and edges**, including nodes and edges that were absent at fit time. It is for practitioners on marketplace, payments and platform-abuse teams who have partial labels (chargebacks, manual review, confirmed takeovers) and a fixed-size review queue. The single thing it does better than a revived PyGOD: **inductive scoring with an honest baseline**. PyOD's graph layer, shipped 2026-04-11, raises `NotImplementedError` on `decision_function`, `predict`, `predict_proba` and `predict_confidence` for all eight of its graph detectors, and its own README says so ("transductive in v1, no out-of-sample predict"). PyGOD is node-only, transductive in practice, and its flagship detector OOMs on the Elliptic Bitcoin graph on a 12 GB GPU (open issue #118, unanswered since 2025-04-19). graphspot's v0 acceptance test is the inverse: fit on Elliptic time steps 1-34, score steps 35-49 it never saw, on a laptop, in one command, with a number that reproduces byte-for-byte from a locked script. Every benchmark run automatically includes a no-graph tabular baseline and prints a loud warning when the graph model fails to beat it.

## 3. SCOPE OF v0

Seven detectors plus one first-class transform. All are inductive by construction except where noted.

| # | Component | Level | Deps | Why it earns the slot |
|---|---|---|---|---|
| 1 | `NeighborAggregation` (transform) | node, edge | core | The differentiator, not a detail. k-hop sparse aggregation (mean/max/min/std/sum) as a sklearn-style `fit`/`transform`. Makes any tabular fraud model the user already owns graph-aware in one line. Scores 83.86 avg AUROC standalone in GADBench, ahead of GCN (77.61), GIN (80.27), GAS (81.58), DCI (80.53). TFM4GAD (WWW'26) shows the field converging on exactly this shape. Nobody ships it. |
| 2 | `XGBGraph` | node, edge | core | GADBench rank #1 of 29: 86.53 avg AUROC, 58.95 avg AUPRC. YelpChi AUPRC 91.11 vs GCN 20.88. Gains only +1.34 AUPRC from a 100-trial search, so the default path works out of the box. |
| 3 | `RFGraph` | node, edge | core | GADBench rank #3 AUROC (84.63), #2 Rec@K, best on Elliptic (93.21 / 78.86) and Weibo (99.43). Shares ~90% of its code with #2, so marginal cost is near zero, and it disagrees usefully with XGBGraph. |
| 4 | `FlatBaseline` (XGB / RF on raw or NA features, no graph) | node, edge | core | Credibility, and it runs automatically in every benchmark. Plain XGBoost (83.53 / 54.95) beats GCN, SGC, PNA, GIN, GAS and DCI in GADBench. On strict-inductive Elliptic, RF on raw features scores F1 0.821 against the best GNN's 0.689. |
| 5 | `FlatUnsupervised` (IsolationForest / LOF adapters) | node, edge | core | BOND: IsolationForest is the single best method on DGraph (60.9 AUROC), beating every deep GNN; LOF is best on Reddit. Thin adapters over sklearn, near-zero cost, huge trust payoff. |
| 6 | `OddBall` | node | core | 545 citations, the canonical structural baseline. No features, no labels, no GPU. Essential for graphs like DGraph-Fin where "nearly all anomalous nodes share identical features with normal nodes." |
| 7 | `Fraudar` | edge, block | core | Highest-differentiation item on the list. Clean-room from KDD'16. The only maintained Python implementation is GPL-3.0 and therefore unusable by anyone building a permissive stack; the CMU page 404s. Label-free, feature-free collusion detection on bipartite account-to-item graphs, which is precisely seller rings and card-testing fanout. A BSD-3 version becomes the reference implementation by default. |
| 8 | `BWGNN` | node | `[deep]` | The one neural model. GADBench rank #2 overall (85.10), best neural on Reddit, Amazon, T-Finance, DGraph-Fin. Canonical baseline in 2026 papers. No maintained PyG implementation exists: the reference repo is DGL-only, DGL's last release was 2024-09-03 with zero macOS wheels for torch 2.7+. A clean, tested, inductive PyG BWGNN is a scarce artifact and buys credibility with the research audience. |

**Explicitly OUT of v0, with reasons:**

- **DOMINANT and the whole reconstruction family** (CoLA, CONAD, AnomalyDAE, GUIDE, Radar, ANOMALOUS, SCAN). PyOD v2.2.0 shipped all eight natively on PyG under BSD-2 on 2026-04-11. Reimplementing them is dead on arrival, and DOMINANT would drag torch into the critical path for a commodity model. Instead ship `graphspot.compat.from_pyod(estimator)`, a documented adapter that runs any PyOD detector through graphspot's evaluation harness. This converts the biggest competitive threat into a distribution channel and is consistent with already having a merged PR in PyOD.
- **GHRN.** Genuinely good (GADBench rank #4, composable) but it is a wrapper over a neural encoder, so it only pays off after BWGNN is solid. First item in v0.2.
- **CARE-GNN.** The evidence for it is real (91.19 vs XGBGraph's 64.03 on label-scarce YelpChi) but it requires the multi-relation data model, which is L effort. v0 stores relation IDs on every edge so the data model is ready, and v0.2 ships the detector.
- **PC-GNN, ConsisGAD, SpaceGNN, GAD-NR, TAM, GGAD.** Weak or unreplicated benchmark evidence, unlicensed repos, or too new. PC-GNN ranks 81.88 in GADBench, below plain XGBoost.
- **Graph-level detection.** Thin fraud demand.
- **Streaming / river-style online scoring.** Real need, wrong shape for v0's API.
- **`sklearn.Pipeline` compatibility.** sklearn assumes `X` is `(n_samples, n_features)`. A graph is not. Implement `get_params`/`set_params` by hand, do not inherit `BaseEstimator`, and say so in the docs.

## 4. PUBLIC API

**Dependency decision: torch-free core.** Core deps are `numpy`, `scipy`, `scikit-learn`, `pandas`, `xgboost`. `torch` and `torch_geometric` sit behind `[deep]` with imports deferred to call time, never at module scope.

Justification, all measured: torch's manylinux wheel is 526.6 MB. PyGOD's failure mode is fully explained by dependency handling: `pip install pygod` succeeds and `import pygod` raises `ModuleNotFoundError: No module named 'torch'`, because it declares five pure-Python deps and no torch. It then charges that 527 MB to run SCAN, a set-intersection algorithm. Seven of graphspot's eight components need no autodiff at all. xgboost is the one heavy-ish core dep and it is defensible: 57.6 MB on manylinux x86_64, 2.4 MB on macOS arm64 (verified on PyPI 2026-08-14), one tenth of torch and the library's #1 detector. `requires-python = ">=3.10,<3.14"`, `xgboost>=2.0` so 3.10/3.11 users resolve cleanly (xgboost 3.4.0 requires 3.12+).

```python
# graphspot/graph.py  -- zero torch imports anywhere in this module
GraphLike = Union["Graph", sp.spmatrix, np.ndarray, "nx.Graph", "Data", pd.DataFrame]

@dataclass
class Graph:
    adj: sp.csr_matrix                     # (n_nodes, n_nodes)
    x: np.ndarray | None = None            # (n_nodes, n_node_feats)
    edge_index: np.ndarray | None = None   # (2, n_edges), CSR-consistent order
    edge_attr: np.ndarray | None = None    # (n_edges, n_edge_feats)
    edge_type: np.ndarray | None = None    # (n_edges,) relation id; multi-relation ready
    edge_time: np.ndarray | None = None    # (n_edges,) unix seconds
    node_ids: np.ndarray | None = None     # join key back to the warehouse
    node_type: np.ndarray | None = None    # (n_nodes,) for bipartite / typed graphs

    @classmethod
    def from_pandas(cls, df, *, source: str, target: str,
                    edge_features: Sequence[str] | None = None,
                    time: str | None = None,
                    node_features: pd.DataFrame | None = None,
                    relation: str | None = None,
                    directed: bool = True) -> "Graph": ...
    @classmethod
    def from_scipy(cls, adj, *, x=None) -> "Graph": ...
    @classmethod
    def from_networkx(cls, g, *, node_features=None) -> "Graph": ...
    @classmethod
    def from_pyg(cls, data) -> "Graph": ...     # lazy import
    def to_pyg(self) -> "Data": ...             # lazy import, ImportError names the extra
    def subgraph(self, nodes) -> "Graph": ...
    def before(self, t: float) -> "Graph": ...  # temporal slice, for inductive splits

def as_graph(g: GraphLike, **kw) -> Graph: ...  # single normalization funnel
```

```python
# graphspot/base.py
class BaseDetector(ABC):
    supported_levels: ClassVar[tuple[str, ...]]      # ("node",) / ("node","edge") / ("edge","block")
    requires: ClassVar[tuple[str, ...]] = ()         # ("torch","pyg")
    inductive: ClassVar[bool] = True

    def __init__(self, *, level: Literal["node","edge"] = "node",
                 contamination: float = 0.01, random_state: int | None = None): ...

    @abstractmethod
    def fit(self, graph: GraphLike, y: np.ndarray | None = None) -> Self: ...
    @abstractmethod
    def decision_function(self, graph: GraphLike) -> np.ndarray: ...   # NEVER NotImplementedError

    def fit_predict(self, graph, y=None) -> np.ndarray: ...
    def predict(self, graph: GraphLike | None = None) -> np.ndarray: ...   # one array, always
    def predict_proba(self, graph=None, *, method="linear") -> np.ndarray: ...
    def explain(self, idx: int | np.ndarray, k: int = 5) -> Explanation: ...
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **params) -> Self: ...

    # fitted attributes: PyOD-identical plural names, always numpy
    decision_scores_: np.ndarray
    labels_: np.ndarray
    threshold_: float
```

`predict()` returns one array. No `return_pred`/`return_score`/`return_prob`/`return_conf` flag soup whose return shape mutates at runtime, which is the worst part of PyGOD's surface. Names are `decision_scores_`, `labels_`, `threshold_`, matching PyOD exactly; PyGOD's singular `decision_score_`/`label_` divergence was filed as a bug (#80) and closed unfixed, and it costs every cross-library user an `AttributeError`.

`explain()` ships in v0 for the tree detectors and Fraudar only, because there it is nearly free and genuinely honest: every NA feature is a `(hop, aggregator, original_feature)` triple, so importance maps directly to "driven by the 2-hop mean of `chargeback_rate` across this seller's buyers." Fraudar returns block membership. Neural explanation is out of scope.

Realistic usage:

```python
import pandas as pd, graphspot
from graphspot.detectors import XGBGraph, Fraudar

tx = pd.read_parquet("transactions_2026q2.parquet")
g = graphspot.Graph.from_pandas(
    tx, source="buyer_id", target="seller_id", time="ts",
    edge_features=["amount", "hours_to_ship", "is_new_device"],
    node_features=accounts.set_index("account_id"),
)

train, test = graphspot.temporal_split(g, cutoff="2026-06-01")   # strict inductive by default

det = XGBGraph(level="edge", hops=2, aggregators=("mean", "max", "std"))
det.fit(train, y=train.edge_labels)          # chargeback labels on the transaction
scores = det.decision_function(test)          # edges never seen at fit time

print(graphspot.evaluate(test.edge_labels, scores, k=2000))
# {'auprc': 0.412, 'rec_at_k': 0.331, 'auroc': 0.938, 'prec_at_k': 0.089,
#  'flat_baseline_auprc': 0.287, 'graph_lift': '+0.125 AUPRC'}

det.explain(scores.argmax(), k=5)
# top drivers: 2hop_mean(chargeback_rate)=0.41, 1hop_max(is_new_device)=1.0, ...

rings = Fraudar().fit(train)                  # no labels, no features required
rings.blocks_[0].nodes                        # 43 buyers, 2 sellers, density 0.94
```

## 5. DATASETS

All URLs re-verified HTTP 200 on 2026-08-14.

**Tier A, auto-download, ships in v0 (7 loaders):**

| Dataset | Source | Torch needed? |
|---|---|---|
| Amazon | `data.dgl.ai/dataset/FraudAmazon.zip` (26.1 MB, `.mat`, `scipy.io.loadmat`) | no |
| YelpChi | `data.dgl.ai/dataset/FraudYelp.zip` (18.0 MB, `.mat`) | no |
| Tolokers | `yandex-research/heterophilous-graphs` `.npz` | no |
| Questions | same | no |
| Elliptic | `data.pyg.org/datasets/elliptic/*` raw CSVs, pandas | no |
| Weibo | `pygod-team/data/weibo.pt.zip` (13.3 MB) | **yes, one-time convert** |
| Reddit | `pygod-team/data/reddit.pt.zip` (2.4 MB) | **yes, one-time convert** |

Weibo and Reddit are pickled PyG `Data` objects, so the first load needs `graphspot[deep]`; graphspot converts to a `.npz` cache once and every subsequent load is torch-free. Never call bare `torch.load`; pass `weights_only=False` explicitly, verify sha256 before unpickling, and say in the docstring that this is a pickle execution surface. PyGOD's loader has been broken on torch >= 2.6 for exactly this reason for 21 months.

Amazon and YelpChi are 3-relation graphs. v0 loads the union as a homogeneous graph and populates `edge_type`, so CARE-GNN in v0.2 needs no data-model change.

**Licensing, cleared:** Elliptic is CC BY-NC-ND 4.0. Download at runtime from PyG's mirror, cache on the user's machine, **never re-host and never redistribute a converted file** (NoDerivatives). Gate it behind an explicit one-time `accept_license=True`, and state plainly in the docs that NonCommercial makes it unusable for employer work. Tolokers, Questions, Weibo and Reddit are the only safe mirror candidates if you ever build a Hugging Face mirror (currently HF has nothing: searches for "yelpchi", "tfinance", "amazon-fraud" and "graph anomaly" return zero datasets, which is an ownable wedge for later).

**Tier B, manual fetch with a checksum verifier and a `graphspot datasets fetch --help` that prints URL, license and drop path:** DGraph-Fin (registration-gated, my probe of the zip returned a 2,383-byte HTML gate page), T-Finance, T-Social (single Google Drive folder, no license stated). Do not auto-scrape. T-Social is 1.40 GB and stays out of CI.

**Tier C, opt-in only with a warning:** `inj_cora`, `inj_amazon`, `inj_flickr`, `gen_*`. Importing one emits a warning citing the ICDE 2023 leakage result (arXiv:2210.12941: a trivial method exploiting the injection artifact reaches state of the art). Never included in a default sweep. Present only so users can reproduce BOND and PyGOD numbers.

**Provenance is first-class metadata, not README prose.** Every `Dataset` carries `label_type` (adjudicated / proxy / injected / synthetic), `label_source` (one sentence, e.g. "helpful-vote ratio thresholding, Dou et al. 2020"), `license`, `redistributable: bool`, `auto_download: bool`. Amazon and YelpChi are labelled `proxy`, not `adjudicated`. No library does this and it is the first thing a bank or marketplace risk team asks for.

## 6. BUILD PLAN

Each week ends in something you can show someone.

**Week 1. Core plus the two winners.** `Graph`, `as_graph`, `from_pandas`, `BaseDetector`, `NeighborAggregation`, `XGBGraph`, `RFGraph`, `FlatBaseline`. Set up release automation and the PyPI-wheel smoke test on day one, not in week 6. Demo: a notebook reproducing GADBench's YelpChi AUPRC (target 91.11 for XGBGraph, tolerance +/- 2) and Amazon (93.33), on scipy sparse, with no torch installed. **If these numbers do not land in week 1, stop and reassess.**

**Week 2. Datasets and the evaluation harness.** Five torch-free Tier A loaders, provenance metadata, `graphspot.evaluate` (AUPRC primary and the model-selection metric, Rec@K, AUROC de-emphasized, Precision@k), strict-inductive splits as the default, the GADBench semi-supervised regime (100 labels: 20 positive, 80 negative), and automatic flat-baseline comparison with a loud warning when the graph loses. Demo: `graphspot bench --quick` prints a five-dataset table from one command, regenerable from a locked seed script.

**Week 3. The edge axis and Fraudar.** Edge-level NA (aggregate over both endpoint neighborhoods, concat edge features), `score_transactions(df, ...)` convenience entry point, and a clean-room Fraudar from the KDD'16 paper only. Write the clean-room provenance note in `LICENSE-THIRD-PARTY` as you go, and do not open the GPL-3.0 `rgmining/fraudar` source at any point. Demo: seller-ring detection on YelpChi edges plus a synthetic bipartite marketplace graph, showing blocks that node-level detectors miss entirely.

**Week 4. Scale, time, and the acceptance test.** OddBall, the Elliptic loader with `EllipticTemporal` support, per-timestep metrics (refuse to emit a single aggregate score on a temporally ordered dataset; the step-43 shutdown drops the base rate 39x from 11.6% to 0.3%), and a base-rate sweep utility that subsamples anomalies to 0.1% and reports recall. Demo: **the acceptance test.** Elliptic, strict inductive, on the M-series laptop, one command, with a memory and wall-clock number and a per-timestep table. This is the thing PyGOD's 1,497 stars cannot do.

**Week 5. BWGNN on PyG.** Clean-room from ICML'22 (Beta(p,q) polynomial filters over the normalized Laplacian, fully specified in the paper), behind `[deep]`, with neighbor sampling and a real inductive `decision_function`. Demo: BWGNN matching published AUROC on Amazon (98.27) and Reddit (70.82) on live PyG 2.8, which no maintained repo currently offers.

**Week 6. Ship.** `explain()` for trees and Fraudar, `graphspot.list_detectors()` reporting what is usable given installed backends, actionable `ImportError` messages that name the extra, `graphspot.compat.from_pyod`, docs, the reproduction script in CI, the written support contract, and PyPI 0.1.0.

**Non-negotiable engineering rules, each earned from a verified PyGOD failure:**
1. No torch or PyG import at module scope. A CI job installs the bare core and runs `python -c "import graphspot; graphspot.list_detectors()"`.
2. A CI job installs the published wheel from PyPI in a clean env and runs the README quickstart verbatim. PyGOD's fatal gap was main-versus-PyPI divergence: CARD merged in Nov 2024, never released, `__version__` never bumped, docs advertising a model pip could not install.
3. Release fires on tag. Test the version matrix you claim to support, including Windows, or do not claim it.
4. One command regenerates every number in the README. Make "our benchmark table reproduces from this command" the headline claim.
5. Never claim scale you have not measured. Publish a memory and runtime table per detector on named hardware, and fail loudly with a clear message rather than OOMing.

## 7. ADOPTION PLAN, FIRST THREE NON-CODE ACTIONS

**1. Ask PyOD about the v2 inductive graph roadmap, this week, before writing code.** Open a polite issue on `yzhao062/pyod` referencing your merged PR #708: "I'm building an inductive/heterogeneous graph AD library and want to be complementary, not duplicative. Is out-of-sample `predict` for the `pyg_*` detectors on the v2 roadmap?" The answer is free, arrives in days, and either confirms the wedge or saves you six weeks. It also opens the door to the interop bridge.

**2. Answer the seven abandoned PyGOD issues with real help.** #118 (mini-batch OOM on Elliptic), #115 (process killed on inj_flickr), #121 (cannot reproduce BOND numbers), #116 and #117 (version-bisect hell), #114 (dynamic graphs), #122 (all scores low). Post one substantive comment each: explain the actual cause (dense adjacency built in `process_graph` before the loader runs; `torch.load` without `weights_only`), and after week 4, add a short graphspot snippet that does the thing they were blocked on. These are pre-qualified users who have been blocked for one to two years with zero maintainer response. Do it as genuine help, not as a drive-by ad.

**3. PR to `safe-graph/graph-fraud-detection-papers`.** 1,883 stars, pushed 2026-06-29, the one channel in this space that is demonstrably alive, while GADBench (149 stars) and PyGOD are both dormant. Add graphspot under a tools or implementations section. Pair it with a single write-up whose headline is the reproducibility table plus the negative results (the datasets where the graph does not beat the flat baseline). Publishing where you lose is what earns trust from practitioners; nobody else in this space does it.

Also, in the README, state plainly and factually that PyGOD is unmaintained (last commit 2024-11-14, last release 2024-02-04, CI red for 216 consecutive days) and provide an API migration table. Its BSD-2 license and its attribute names make switching cost near zero, and no fork has diverged by a single meaningful commit, so the succession slot is open.

## 8. RISKS

**Risk 1: the clean-room numbers do not reproduce.** Every reference implementation for XGBGraph, RFGraph, BWGNN and GHRN is unlicensed (GADBench, Rethinking-Anomaly-Detection, GHRN all have no LICENSE file, meaning all rights reserved), so you must reimplement from papers. If your XGBGraph scores 78 AUPRC on YelpChi against the published 91.11, the entire "reproducible benchmarks" positioning collapses. *Cheapest test: week 1, day 1. Implement NA plus XGBGraph, run YelpChi and Amazon, compare to GADBench Table 13. Two days of work decides whether the whole project is viable.* Note GADBench's splits are frozen precomputed masks, not recomputable, so expect and document a small gap.

**Risk 2: there is no audience.** PyGOD does ~1,259 downloads in the last 30 days against PyOD's 5,152,411, a ratio near 4000:1. The graph AD packaging market may already have been destroyed by PyGOD's dormancy. The upside case rests entirely on torch-geometric's 1,373,253 monthly downloads having no maintained AD option, which is a distribution bet, not a demand fact. *Cheapest test: adoption action #2, executed in week 2 before the library exists. Post substantive answers in the six unanswered PyGOD threads and count replies. If pre-qualified, actively blocked users do not engage, the audience is gone and you should reposition as a PyOD contribution instead of a standalone library.*

**Risk 3: you become PyGOD.** Solo maintainer, one author with 53% of commits, project stops the week life gets busy. PyGOD's release pipeline died nine months before its code did. *Cheapest test: build the release automation and the PyPI-wheel smoke test in week 1, then check on week 6 whether they are still green with zero manual intervention. If you cannot keep a two-job CI matrix green for six weeks while working full-time at eBay, you will not keep a library alive for two years, and you should scope down to a single well-maintained package of NA plus the tree detectors rather than eight components.*

Watch item, not a top-three risk: PyOD ships 12 releases in 4 months and its docs say "transductive **in v1**." Assume 6 to 12 months before it attempts inductive support. Ship v0 well inside that window, and treat interoperability as a deliberate strategy rather than waiting to be absorbed.

## 9. UNVERIFIED

Flagged plainly.

- **T-Finance and T-Social have no stated license anywhere.** The BWGNN repo's license field is null and the data lives in a single Google Drive folder whose metadata returns 403 without an API key. File sizes were computed from the GADBench paper table, not from a download. Tier B only, and do not mirror them.
- **The `FraudAmazon.zip` / `FraudYelp.zip` terms on `data.dgl.ai` are unverified.** The URLs return 200 and graphspot links rather than redistributes, but the underlying redistribution terms were not checked.
- **Elliptic's CC BY-NC-ND status comes from the Kaggle API license field.** Confirm against Elliptic's own terms before publishing anything that touches it, and treat the NonCommercial clause as blocking for employer work.
- **Two of the strongest evaluation-critique sources are 2026 arXiv preprints, not peer reviewed.** "GAD in the Wild" (arXiv:2605.07133) and the Elliptic transductive-leakage re-evaluation (arXiv:2604.19514, single author) are directionally consistent with peer-reviewed ICDE 2023 and LoG 2023 leakage results, but cite the peer-reviewed pair as primary support in any paper.
- **"No maintained PyG BWGNN exists" is absence of evidence.** One unvetted 0-star community port was found. Re-search before claiming novelty publicly.
- **PyGOD's `torch.load` breakage was inferred from code plus the PyTorch 2.6 changelog, not reproduced at runtime.** Do not assert it as a tested fact in the README.
- **BWGNN's citation count is undercounted.** OpenAlex covers only the arXiv preprint (52) and Semantic Scholar was rate-limiting. Its real influence is far higher.
- **Whether PyOD accepts an interop bridge is unknown** until adoption action #1 gets a reply.
- **All star counts, download stats, release dates and namespace availability are point-in-time as of 2026-08-14.** Re-check `graphspot` on PyPI and GitHub immediately before claiming it, and claim it the same day.