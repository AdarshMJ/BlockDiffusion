# CLAUDE.md — BlockDiffusion Project

Persistent project tracker + codebase-understanding notes. Read this first every
session so we don't re-explore the codebase. Update the **Progress Log** at the
bottom whenever a feature lands.

---

## 1. Project goal (the "what" and "why")

Scale **discrete graph diffusion** to **very large sparse graphs** (target:
**~40k-node planar graphs**). Plain DiGress saturates at ~100–200 nodes; the
current record (ETH-2024 iterative local expansion) is ~5037 nodes. We aim for
~8× that.

**Core idea — coarsen → diffuse → decode (factorized block diffusion):**
1. **Coarsen** each training graph `G` into a small super-node graph `G_c` with
   spectrum-preserving **Loukas** coarsening (the `coarsen_mini` package).
2. **Latent diffusion** — a standard DiGress models `p_θ(G_c)` on the small
   coarse graph (deliberately not novel).
3. **Decode back to full resolution** (the heart of the contribution):
   - **(C) size predictor** — predict `v_i` = #original nodes per supernode (for
     padding the intra diffusion).
   - **(D) intra-cluster generation** — a shared discrete diffusion expands each
     supernode `i` into its subgraph `B_i`. Parallel over clusters, conditionally
     independent given `G_c`.
   - **(E) inter-cluster (bipartite) generation** — generate cross-edges **only
     for pairs `(i,j) ∈ E_c`** (sparsity ⇒ scalability). This is the main
     contribution; design is multi-pass, variable-grouping refinement.

Full design rationale, related work, and open questions live in
[BlockDiffusion/scalable_block_diffusion_brainstorm(1).md](scalable_block_diffusion_brainstorm(1).md).
Key competitors to beat/cite: **HiGen** (closest), **LGDC** (closest framing),
**PARD** (block diffusion on graphs), **SparseDiff** (sparse single-model).

**Decoder design options (from brainstorm §4):** Option 0 (shared denoiser on
induced pair-subgraph `B_i∪B_j`, intra frozen, cross-edges masked) for the
engine + Option 1 (full conditional independence) for v1 parallelism; then
Option 2 (multi-pass variable-grouping refinement) as the contribution; Option 3
(coarse-level light autoregression) as a fallback/ablation. Global-degree
coordination via predicted target degree per node (no chemical/valence priors).

---

## 2. Repository layout

Paths in this file are relative to `BlockDiffusion/` (where this CLAUDE.md lives).

```
MemGen/                         ← git root, primary working dir
├── BlockDiffusion/             ← the codebase we work in (a DiGress port)
│   ├── CLAUDE.md               ← you are here
│   ├── main.py                 ← training/test/sample entry point
│   ├── diffusion_model.py      ← DiscreteDenoisingDiffusion (the diffusion core)
│   ├── utils.py                ← PlaceHolder, to_dense, masking, logging
│   ├── models/
│   │   ├── transformer_model.py← GraphTransformer (denoiser network)
│   │   └── layers.py           ← Xtoy, Etoy, masked_softmax
│   ├── diffusion/
│   │   ├── noise_schedule.py   ← cosine schedule, uniform/marginal transitions
│   │   ├── diffusion_utils.py  ← posterior, sampling, masking helpers
│   │   ├── distributions.py    ← DistributionNodes (samples #nodes)
│   │   └── extra_features.py   ← cycles/eigenvalues/degree extra features
│   ├── datasets/
│   │   ├── spectre_dataset.py  ← ALL data modules (see §4)
│   │   └── generate_planar.py  ← synthetic planar dataset generator (Delaunay)
│   ├── metrics/                ← train_metrics, abstract_metrics
│   ├── analysis/               ← spectre_utils (MMD sampling metrics), viz, orca
│   ├── configs/                ← YAML configs (one per experiment)
│   └── coarsen_mini/           ← standalone graph coarsening (see §5)
├── DiGress/                    ← reference original DiGress (not our working copy)
├── scripts/
└── gridout/
```

---

## 3. DiGress diffusion — how it works (data flow)

DiGress = **discrete denoising diffusion** over graphs. A graph is `(X, E)`:
- `X`: node features, one-hot, shape `(bs, n, dx_classes)`.
- `E`: edge features, one-hot, shape `(bs, n, n, de_classes)`. **Class 0 = "no
  edge"** (see `utils.encode_no_edge`). For our current datasets `de=2`
  (no-edge / edge), `dx=1` (single constant node type).
- `y`: graph-global features, here empty `(bs, 0)`; time `t` is appended as the
  conditioning signal.

**Dense representation:** sparse PyG batches → dense `(X, E, node_mask)` via
`utils.to_dense` (uses `to_dense_batch` + `to_dense_adj`). Everything downstream
is dense `n×n`, masked by `node_mask`. *This dense `n×n` is exactly the O(n²)
bottleneck the project removes by factorizing.*

**Key file: [diffusion_model.py](diffusion_model.py) —
`DiscreteDenoisingDiffusion(nn.Module)`**
- `apply_noise(X,E,y,node_mask)` — samples a timestep `t`, builds transition
  `Q̄_t`, computes `probX = X @ Q̄t.X`, `probE = E @ Q̄t.E`, samples a noisy
  one-hot `(X_t, E_t)`. Returns `noisy_data` dict.
- `forward(noisy_data, extra_data, node_mask)` — concatenates extra features onto
  `X_t/E_t/y_t` and calls the `GraphTransformer`. Predicts clean-graph logits.
- `training_step` — noise → extra features → forward → `TrainLossDiscrete`
  (cross-entropy on X and E; weighted by `cfg.model.lambda_train = [λX, λE, λy]`).
- `compute_val_loss` — variational lower bound (NLL): `kl_prior + Σ Lt - log p(N)
  - reconstruction`.
- `sample_batch(...)` — reverse process. Samples #nodes from `node_dist`, starts
  from `limit_dist` noise `z_T`, loops `t=T…1` calling `sample_p_zs_given_zt`.
  Returns a list of `[atom_types, edge_types]` (dense, collapsed to class ids).
- `sample_p_zs_given_zt(s,t,...)` — one reverse step. Uses the analytic posterior
  `compute_batched_over0_posterior_distribution` weighted by the network's
  softmax prediction. Has an **inference-time γ-tempering** knob
  (`cfg.model.gamma`, default 1.0 = no-op) on the forward posterior `q`.

**Transitions** (`diffusion/noise_schedule.py`):
- `PredefinedNoiseScheduleDiscrete` — cosine (default) β schedule, `ᾱ_t`.
- `DiscreteUniformTransition` vs `MarginalUniformTransition`
  (`cfg.model.transition`). **Marginal** uses dataset node/edge marginals as the
  limit distribution (recommended; `limit_dist` = data marginals).

**The denoiser: [transformer_model.py](models/transformer_model.py)
`GraphTransformer`** — DiGress's graph transformer. Stack of `XEyTransformerLayer`
(node/edge/global blocks with FiLM cross-conditioning, `NodeEdgeBlock` self-
attention that also updates edge reps). Input/output MLPs per stream. Symmetrizes
`E`, zeroes diagonal, residual skip on `X`. Hidden dims set in config
(`dx, de, dy, n_head, dim_ffX, dim_ffE`).

**Extra features** (`diffusion/extra_features.py`, selected by
`cfg.model.extra_features`): `'none'` (DummyExtraFeatures), `'degree'`, `'all' |
'cycles' | 'eigenvalues' | 'norm_*'` (ExtraFeatures), or `'sc_*'`
(SelfConditionedDiGressFeatures). They append structural info to `X/E/y` and
change `input_dims` (computed in `SpectreDatasetInfos.compute_input_output_dims`).

**Training loop** (`main.py`): plain PyTorch (no Lightning). AdamW +
ReduceLROnPlateau, grad clip 1.0. Per epoch: `train_epoch` → `validate` (NLL) →
checkpoint best → periodic `sample_batch` + `sample_and_evaluate` (MMD metrics vs
train/test via `analysis/spectre_utils.py`, plus graph-stat tables). Experiments
write to `experiments/<name>_<timestamp>/{checkpoints,graphs,chains,logs}`.

CLI: `python main.py --config configs/<x>.yaml --mode {train,test,sample}
[--checkpoint ...] [--experiment_dir ...] [--device cuda]`.

---

## 4. Data — format and loaders (IMPORTANT)

All datamodules live in
[datasets/spectre_dataset.py](datasets/spectre_dataset.py).
Selected by `cfg.dataset.name` via `create_datamodule` in `main.py`.

**The canonical per-graph PyG `Data` object** (see `_nx_to_pyg`, and `process`):
- `x`: `(n, 1)` float, **all ones** — a single constant node type (these are
  structure-only graphs, no real node labels). `node_types = [1]`.
- `edge_index`: `(2, num_edges)` — **undirected (both directions present)**.
- `edge_attr`: `(num_edges, 2)` float one-hot, `[:, 1] = 1` (all edges are the
  single "edge" type; class 0 is reserved for "no edge").
- `y`: `(1, 0)` empty.
- `n_nodes`: `(1,)` long.

**Datamodules** (all expose `train/val/test_dataloader()` returning PyG
`DataLoader`, plus `node_counts()`, `node_types()`, `edge_counts()`,
`get_gt_graphs()`):
- `SpectreGraphDataModule` + subclasses **SBM / Planar / Comm20** — download
  `.pt` from the SPECTRE repo, 200 (or 100) graphs, split 64%/16%/20% (seed 0).
  **Planar is the project's headline target dataset.**
- `CoraChameleonDataModule` — loads `data/CoraChameleon/node_<size>/<split>.pt`,
  sliced graphs of fixed `node_size` (10…1000). Drops the 100-dim real features,
  replaces with constant `x=1`.
- `TwoDensityMixtureDataModule`, `DCSBMDataModule` — synthetic ER / SBM
  generators (cached `.pt`), built for CRM-prior experiments.
- `CommunityDataModule`, `EgoDataModule` — load EDGE-style `.pkl` lists of
  networkx graphs, 60/20/20 split.

**`SpectreDatasetInfos`** holds `n_nodes` (node-count distribution), `node_types`,
`edge_types` (marginals), `max_n_nodes`, `nodes_dist` (`DistributionNodes`), and
computes `input_dims`/`output_dims` from one example batch + the extra-feature
modules. The diffusion model reads dims and marginals from here.

`single_graph: true` in a config collapses train/val/test to one graph (overfit
sanity test).

---

## 5. Coarsening — `coarsen_mini` (how it works)

Standalone, torch-only, PyG-based. Full docs:
[coarsen_mini/README.md](coarsen_mini/README.md). Works on a
single PyG `Data` (one graph at a time).

**Two-step API:**
```python
from coarsen_mini.algorithms.loukas.main import coarsen_loukas
from coarsen_mini.core.builders import build_coarsening

# 1. Partition the graph (Loukas spectral coarsening)
partition, info = coarsen_loukas(
    data, r=0.5,                          # target reduction ratio (or n_clusters=...)
    laplacian_kind="normalized_self_loop",# default; or "combinatorial" / "normalized"
    method="edges",                       # "edges" (default) | "neighborhood"
    K=100,                                # # preserved eigenvectors
)
# 2. Assemble Q, P, A_c, X_c
coars = build_coarsening(data, partition, method="loukas", r_requested=0.5)
```

**Key objects:**
- `Partition` (`core/partition.py`) — `assignment: (N,) int64` (contiguous
  cluster ids), `n_clusters`, `cluster_sizes` (= our `v_i`!), `to_Q_combi()`
  (binary lifting, torch sparse CSR). `cluster_sizes` is exactly the size-predictor
  target for Stage (C).
- `Coarsening` dataclass (`core/coarsening.py`) — the full artifact:
  - `original`, `coarsened` (PyG `Data` of `G_c`: `x=X_c`, `edge_index`,
    `edge_weight`, `num_nodes=n_clusters`).
  - `Q`: `(N, n)` sparse CSR **degree-normalized** lifting `Q[i,c]=sqrt(D_N[i]/D_n[c])`.
  - `P`: `(n, N)` sparse CSR `= pinv(Q)` (cluster-mean projector for binary Q).
  - `P_loukas`: optional multilevel chain (Loukas only).
  - `partition`, `r_requested`, `r_actual`, `laplacian_kind`.
  - helpers `.N`, `.n`.
- `X_c = P @ X` (cluster-mean features). `A_c = Q_binary^T A Q_binary`, diagonal
  dropped, symmetrized (`core/q.py::coarsened_adjacency`).

**`info` dict** from `coarsen_loukas`: `n_clusters`, `r_actual`, `n_levels`,
`R` (eigenvectors), `P_loukas`, timings, `stalled` flag (warns if Loukas can't
reach target reduction — relevant at high `r` / large graphs; absorption fallback
mitigates).

**Laplacian modes:** `combinatorial` (L=D−A, community/cut structure),
`normalized_self_loop` (default, matches GCN propagation `I−L̂`), `normalized`.
For our generative use the choice affects the RSA fidelity story (brainstorm §8).

**File map:** `algorithms/loukas/{main,cost,matching}.py` (multilevel loop,
per-edge spectral cost, greedy non-overlapping matching), `algorithms/graclus.py`
(needs torch-cluster), `core/{builders,coarsening,partition,q,laplacian,eig,
conversions}.py`. Demo: `python -m coarsen_mini.demo` (Cora at r=0.5).

**Conventions to remember:** all matrices are torch **sparse CSR** (never COO);
default dtype float32 (eig promotes to float64 internally). `Q` is the
degree-normalized lifting; `Q_binary` (from `partition.to_Q_combi()`) is only for
building `A_c`.

---

## 6. How DiGress and coarsening connect (the plan, not yet built)

The two halves are currently **independent**: DiGress trains/samples graphs;
`coarsen_mini` reduces a single PyG graph. The project glues them:

1. **Build `(G, G_c)` pairs** — run `coarsen_loukas` + `build_coarsening` over
   each training graph; store `G_c` (for Stage B latent diffusion), the partition
   / `cluster_sizes` (Stage C targets), the intra subgraphs `B_i` and the
   inter pairs `B_i∪B_j` for `(i,j)∈E_c` (Stage D/E training examples).
2. **Latent diffusion (B)** — train the existing `DiscreteDenoisingDiffusion` on
   the collection of small `G_c`. Mostly reuse current code (note: `G_c` has
   weighted edges; current pipeline uses 2-class one-hot edges — needs a
   decision on edge-weight handling).
3. **Size predictor (C)** — new small GCN/transformer → `v_i`.
4. **Shared intra/inter denoiser (D/E)** — a diffusion denoiser run on small
   subproblems (single cluster, or cluster pair with intra frozen + cross-edges
   masked). One 40k graph fragments into thousands of subproblem training
   examples.

Open implementation questions are tracked in the brainstorm §9 and in §6b below.

### 6b. Pending design decisions (distilled from the brainstorm)

**Decided so far:**
- **Edge-weight handling (Stage B): Option A — discretize/bucketize.** Coarse
  super-edge weights are integer counts of contracted cross-edges (`A_c[i,j]`).
  We model them as categorical edge classes, no transformer-architecture change;
  only the edge encoding + `edge_types` marginals change.
- **`K` = 11 edge classes** (`de = 11`): `{0 = no-edge, 1…9 = exact weight,
  10 = "≥10" clamp}`. Set from the coarsening diagnostic at `r=0.9`
  (`w_p99=9`, `w_max=13` → buckets 1–9 cover >99%, heavy tail clamps to class 10).
  The clamp bucket makes this robust to the full 5000-graph set having a heavier
  tail than the diagnostic's 50-graph sample. Encoding rule for weight `w`:
  `0→0`, `1≤w≤9→w`, `w≥10→10`.
- **Reduction ratio `r = 0.90`** (Stage A, dev/500-node). Gives `n_c≈50`
  (latent diffusion solidly inside DiGress's comfort zone), `v̄≈10`,
  `v_max≈36`, `|E_c|≈137`. No Loukas stalls on planar-500. **Fallback `r=0.80`**
  (`n_c≈99`, `v̄≈5`, `v_max≈12`, `K≈7`) if the intra diffusion struggles with the
  `v_max=36` cluster-size tail / padding waste — one-line change + re-coarsen.
  Diagnostic tool: `coarsen_diagnostic.py`; raw numbers in
  `coarsen_diagnostic.json`.

**Must decide to run a naive v1 (blockers for an end-to-end run):**
1. **Size predictor (Stage C)** — for v1 can sample `v_i` from the empirical
   per-cluster-size distribution (or use ground-truth sizes); defer the real
   GCN/transformer predictor.
2. **Intra/inter engine** — v1 = **Option 0** (one shared denoiser on the induced
   subgraph, intra frozen + cross-edges masked) + **Option 1** (bipartites
   conditionally independent given `G_c` + intra → fully parallel).

**Can defer (do NOT block naive v1; these are the research contributions):**
- **Option 2** multi-pass variable-grouping refinement (the main contribution)
  and its convergence framing (Gibbs/fixed-point) — brainstorm §4.2, §9 Q1.
- **Global-degree coordination** (§4.2bis): expose consumed degree or predict
  per-node target degree up front; hard-mask validity net. Affects *quality*, not
  whether a naive run executes.
- **Diverse coarsening forest** (§7): multiple randomized coarsenings per graph
  (ETH-style) vs one fixed coarsening. Start with one fixed coarsening.
- **`laplacian_kind` / `method`** (combinatorial vs nsl; edges vs neighborhood).
- **RSA ε → fidelity bound** (§8) — the theory "cherry," explicitly not a blocker.
- **Dedicated bipartite diffusion module** vs Option 0 shared denoiser (§4.5).

**Known engineering kinks to iron out (not research, just plumbing):**
- A datamodule that serves coarse graphs `G_c` with bucketed-weight edge classes
  to the existing DiGress (currently hardcoded `dx=1`, `de=2`).
- Variable-size intra subproblem batching (DiGress already supports variable `n`
  via `node_mask` + `DistributionNodes`, so this is mostly batching/padding code).
- A reconstruction/"uncoarsening" routine that stitches generated `B_i` + the
  per-`E_c` bipartites back into one full graph.
- Conditioning channel: how the coarse-graph context reaches each intra/inter
  subproblem (cache the `G_c` embedding once, §4.4).

**Bottom line:** a naive pipeline (Option 0 + Option 1, fixed coarsening, sampled
sizes) is buildable now and is exactly what the brainstorm prescribes for v1 —
its purpose is to *measure the inter-cluster incoherence baseline* (§9 Q2) that
Option 2 later fixes. The degree/coherence issues are quality kinks, not run
blockers. Dev scale is 500-node planar (this is small enough to also sanity-check
against full DiGress), before pushing toward the 40k headline target.

**Open scaling tension (40k target, NOT the dev pipeline):** single-level
coarsening obeys `n_c · v̄ ≈ N`. At `N=40k` with `v̄=10`, `n_c=4000` — but plain
DiGress (the "deliberately standard" latent stage) saturates ~200 nodes. You
cannot keep both `n_c` and `v̄` small with one coarsening level at 40k. The 40k
story will need one of: aggressive coarsening (`v̄≈200, n_c≈200`, bigger intra
problems), **recursive/multi-level** coarsening of `G_c`, or a scalable latent
diffusion (SparseDiff-style). Irrelevant at 500-node dev scale; flagged so it's
on the radar before we scale up. Also note coarsening cost: ~0.45 s/graph at 500
nodes with `K=100` (so ~40 min one-time preprocessing for 5000 graphs); the eig
will be markedly slower at 40k.

---

## 7. Conventions & environment

- Pure PyTorch (no pytorch-lightning). Config = YAML → `DotDict` (dot access).
- Configs in `BlockDiffusion/configs/`. `config_*.yaml` per dataset; `dignode*`
  for CoraChameleon node-sliced runs.
- Run GPU smoke tests by handing the user runnable code (see workflow notes).
- coarsen_mini import root: `PYTHONPATH=<parent of coarsen_mini>` i.e. run from
  inside `BlockDiffusion/`.

---

## 8. Progress Log

- **2026-06-21** — Initial codebase exploration. Read brainstorm, DiGress core
  (`main.py`, `diffusion_model.py`, `transformer_model.py`, `utils.py`,
  `spectre_dataset.py`, a config) and `coarsen_mini` (README, `loukas/main.py`,
  `core/{builders,coarsening,partition,q}.py`). Wrote this CLAUDE.md.
- **2026-06-21** — Added `datasets/generate_planar.py` (Delaunay planar-graph
  generator; default 5000 train / 500 val / 500 test × 500 nodes, saved as
  per-split nx pickles + meta.json). Decided **Option A** (bucketized categorical
  edge weights) for Stage-B latent diffusion. Catalogued pending design decisions
  in §6b. Still no changes to the diffusion/coarsening code itself.
- **2026-06-21** — Ran `coarsen_diagnostic.py`; locked `r=0.9` and `de=11` weight
  buckets (see §6b). Fixed a Python<3.10 compat bug in
  `coarsen_mini/core/partition.py` (missing `from __future__ import annotations`).
- **2026-06-21** — Built the **Stage-A→B latent-diffusion pipeline**:
  - `build_coarse_dataset.py` — coarsens planar graphs → `G_c` records
    (`coarse` Data with `de=11` bucketed edges + stashed `assignment`/
    `cluster_sizes` for Stages C/D/E). Coarsens ONCE; saves `data/coarse_planar/
    {train,val,test}.pt`.
    Edge bucketing: `clamp(round(w),1,10)` → classes 1..10; class 0 = no-edge.
  - `datasets/spectre_dataset.py` — added `CoarseGraphDataset` +
    `CoarseGraphDataModule` (`name='coarse_planar'`), with a de-generic
    `edge_counts` override (base `_SyntheticDataModuleBase` hardcodes de=2).
  - `main.py` — registered `coarse_planar` in `create_datamodule` and
    `create_sampling_metrics` (generic structural MMD; no planar/SBM validity).
  - `configs/config_coarse_planar.yaml` — DiGress on `G_c` (dx=1, de=11, marginal
    transition, extra_features='all', λE=5).
  - Verified: `ExtraFeatures` ('all') is de-agnostic (`E_t[...,1:].sum(-1)`), so
    de=11 is safe; all files py_compile-clean. **Not yet run** (no ML deps in the
    sandbox — user runs on GPU). Run order in the config header.
- **2026-06-21** — **Unified into one command + one config.** Coarsening logic
  moved to `datasets/coarsen_pipeline.py` (`build_coarse_cache` = lazy, cached,
  idempotent; cache subdir keyed by `r/K/laplacian/method`). `CoarseGraphDataModule`
  now builds the coarse cache on first run from a new `coarsening:` config section,
  then trains — so `python main.py --config configs/config_coarse_planar.yaml
  --mode train` does coarsen+train in one shot (planar dataset assumed already
  generated). `build_coarse_dataset.py` kept as an OPTIONAL CPU-node prebuild
  wrapper over the same `build_coarse_cache`. Aux features stay `'all'` (required).

## 9. Files I have NOT yet read in depth
(ask the user for these if/when needed, per workflow preference — don't grep)
`diffusion/{noise_schedule,diffusion_utils,distributions,extra_features}.py`
(understood via usage), `metrics/*`, `analysis/spectre_utils.py` (MMD metrics),
`analysis/visualization.py`, `coarsen_mini/algorithms/loukas/{cost,matching}.py`,
`coarsen_mini/core/{laplacian,eig,conversions}.py`, `coarsen_mini/algorithms/graclus.py`.
