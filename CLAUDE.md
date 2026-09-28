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
[BlockDiffusion/scalable_block_diffusion_brainstorm.md](scalable_block_diffusion_brainstorm.md).
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

> **CURRENT STATE (2026-06-21): only Stage B (latent diffusion on `G_c`) is
> built. There is NO decoder/uncoarsening yet (Stages C/D/E unbuilt).**
> Consequences when reading any `coarse_planar` experiment:
> - Generated outputs and the `graphs/comparisons/*.png` are **coarse graphs**
>   (~50 nodes), NOT full 500-node planar graphs. The "Ground Truth" panel is the
>   ground-truth *coarse* `G_c`, not the planar graph — the label is misleading.
> - All MMD metrics in those logs are **coarse-vs-coarse**. The huge `r_*` ratios
>   are largely an artifact: coarse graphs of planar graphs are near-identical, so
>   the train↔test MMD denominator is tiny and any deviation explodes the ratio.
>   Judge by **absolute** stats (PLE/CPL/triangles/CL-overlap were all close in
>   the first 100-epoch run) and `clustering MMD` (was 0.34 — the real gap, shows
>   up as "holes"/uneven triangulation in the generated coarse graph).
> - First run (100 epochs) had **non-converged val NLL** (bounced 1800–2200) →
>   undertrained. Next step: longer training pass (n_epochs bumped to 1000).
> - Coarse `G_c` is weighted and denser (~0.11), but the earlier claim that
>   coarsening breaks planarity was wrong. Contracting connected clusters in a
>   planar graph produces a planar minor, and the decoder extractor verified all
>   Loukas clusters are connected. Oracle `G_c` must therefore remain planar;
>   generated Stage-B samples should be scored for planarity. Holes in generated
>   `G_c` will still **propagate** into any decoded full graph.

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
- **Reduction ratio `r = 0.80`** (current, switched from 0.90). `n_c≈99`,
  `v̄≈5`, `v_max≈12`, `K≈7` — tighter/more-equitable clusters and sparser,
  more planar-like coarse graphs than `r=0.9` (`n_c≈50`, `v_max≈36`). No Loukas
  stalls on planar-500. Diagnostic tool: `coarsen_diagnostic.py`.
- **`laplacian_kind = 'normalized'`** (current, switched from
  `normalized_self_loop`). Planar graphs have no self-loops, so there's nothing
  to preserve from the `+I` shift; plain `D^{-1/2}(D−A)D^{-1/2}` is the right
  spectrum to preserve. **NB:** the only valid strings are `combinatorial` /
  `normalized` / `normalized_self_loop` (see `coarsen_mini/core/laplacian.py:29`);
  anything else raises `ValueError` at coarsen time.

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

### 6c. Decoder design (agreed 2026-07-10)

**The coarse graph is a *blueprint* carrying a size/edge budget:**

| Symbol | Meaning | Status |
|---|---|---|
| `v_i` | #original nodes in supernode `i` | stored (`cluster_sizes`) |
| `e_i` | #edges *inside* cluster `i` | **not in `G_c`** — it's the dropped diagonal of `A_c` (`delete_diagonal=True`). Recomputable at train time from `assignment` + the planar graph. |
| `w_ij` | #original edges between clusters `i,j` | the super-edge weight (our `de=11` classes) |

Budget identities: `Σv_i = N`, `Σe_i + Σw_ij = |E|`.
**Consistency constraints nothing currently enforces:** (1) `Σv_i = N`;
(2) `w_ij ≤ v_i·v_j` (can't fit 10 cross-edges between two 2-node clusters);
(3) `v_i ≥ 1`, `w_ij ≥ 1` for `(i,j) ∈ E_c`.

**`w_ij` has two jobs:** something to *generate* in Stage B, and a *per-bipartite
edge budget* conditioning Stage E ("place exactly `w_ij` cross-edges between
`B_i` and `B_j`"). The second job partly dissolves brainstorm §4.2bis: the
per-*pair* count is pinned; only the per-*node* split stays free.

**Don't predict `e_i`.** Condition the intra diffusion on `v_i` + the supernode's
coarse embedding and let it learn `p(B_i | v_i, ctx)`; the edge-count distribution
comes free. A third quantity to keep consistent buys little.

**`N` is derived, not generated:** `N' = Σ_{i=1..n_c} v_i`. Independent `v_i`
sampling gives `σ_N' ≈ σ_v·√n_c`, so **relative** error *shrinks* with scale
(≈4% at N=500/n_c=99; ≈0.4% at N=40k). A cheap repair (redistribute the residual
`N − Σv_i` across the least-confident clusters) makes it exact. Settled: do the
repair; exact `N` is not a design risk at any scale.

**DECODER = alternating intra/inter block-Gibbs (agreed, supersedes two-phase).**
Partition every fine-graph edge variable by the clustering: intra blocks `A_ii`,
inter blocks `A_ij` for `(i,j)∈E_c`, and everything else pinned to 0 (the
sparsity that buys scalability). **Every edge belongs to exactly one block** — no
subproblem writes another's variables — so alternating denoising over
{intra blocks} and {inter blocks} is a well-defined block-Gibbs / Gauss–Seidel
scheme on the joint, not a hack. One shared denoiser, one shared diffusion clock:
> at reverse step `t`: denoise all intra blocks in parallel conditioned on the
> current inter state → then all inter blocks in parallel conditioned on the
> just-updated intra state.

- The old two-phase plan (all `T` intra steps, freeze, all `T` inter steps) is the
  *degenerate case* and bakes in exactly the independence assumption we want to
  drop. **Keep it as the ablation baseline** to quantify what alternation buys.
- This is the concrete instantiation of brainstorm **Option 2** (multi-pass
  refinement), with the associative-states (§6) cover.
- **Degree coordination falls out:** feed each node's *current total degree*
  (intra + all its cross blocks), recomputed each pass, as a node feature into
  every call. That is §4.2bis's "consumed degree" channel, obtained by iteration
  instead of an up-front budget.
- **Training:** sample one `t`, noise all fine-graph edges (non-`E_c` pairs stay
  0), then train the shared denoiser on every cluster subproblem and every pair
  subproblem in parallel. One network, one clock, one loss.
- **Honest caveat:** a reverse step factorizes the true joint kernel into a
  product of block kernels; exact only as per-step noise → 0 (large `T`). Measure,
  don't assume.

**Open architecture fork — resolved by the `de=2` ablation (running):**

| | Stage B models | Predictor predicts |
|---|---|---|
| **A. Joint** | topology + `w_ij` (E) + `v_i` as **node class** (X) | nothing |
| **B. Split** | **binary** topology (`de=2`) | `v_i` **and** `w_ij` |
| **C. Current** | topology + `w_ij` (`de=11`) | `v_i` only |

Note `X` is currently *wasted* (`dx=1` constant ⇒ zero gradient from `λX`).
If de=2 clearly beats de=11 on `clustering MMD` / holes → **B**. If holes persist
→ weights aren't the cause (diffusion capacity / intrinsic coarse-graph
difficulty) → **A** is fine.

**Predictor network spec (if B or C wins):** inputs = generated `G_c` (topology,
weights if present, degree/weighted-degree/Laplacian eigvec features); 3–5 layers
GINE or a small graph transformer (`n_c≈99`, tiny); **classification heads, not
regression** (`v_i ∈ {1..12}` → softmax + CE, PARD Algo. 2 style) so we can
*sample* and keep diversity. In **B**, sample `v` first, then condition the `w`
head on sampled `v` so `w_ij ≤ v_i·v_j` holds by construction rather than by
clamping.

**Structural prior to verify (not assume):** Loukas contracts edges/neighborhoods,
so every cluster `B_i` *should* be a connected subgraph. If true, Stage D should
generate connected subgraphs. Assert it in the extractor.

**RISK ON THE RECORD:** nothing in this decoder guarantees **planarity**. Oracle
`G_c` is planar (it is a minor when clusters are connected), but factorized
intra+inter expansion can realize its coarse edges with mutually incompatible
fine endpoints and create a non-planar expansion.
Planar validity is the headline metric. ETH-2024 gets it from local expansion +
refinement; we have no equivalent. Think about a planarity-aware sampling mask
(à la DiGress's valency mask) *before* eval, not after.

---

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
- **Local environment has no PyTorch or PyTorch Geometric.** Only run quick,
  dependency-free checks locally. For anything requiring torch/PyG, a GPU, full
  datasets, or appreciable runtime, give the user explicit runnable commands;
  do not launch it locally.
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
- **2026-06-21** — First coarse_planar run (100 epochs) reviewed: Stage-B only,
  coarse-vs-coarse eval, undertrained (val NLL not converged), generated coarse
  graphs show holes / clustering MMD 0.34. Added the CURRENT STATE callout in §6.
  Bumped `n_epochs`→1000 (sample_every_val 50, save_every_epochs 50) for a longer
  pass; added `run_coarse_train.sh` (resumable) for the oarsub job. Cache at
  `data/coarse_planar/r0.9_normalized_self_loop_edges_K100/` is reused (no
  re-coarsen).
- **2026-07-10** — Design session for the decoder. Agreed: **alternating
  intra/inter block-Gibbs** decoder (user's idea; supersedes the two-phase plan),
  the *blueprint* framing (`v_i`, `e_i`, `w_ij` + consistency constraints), and
  that exact `N` is a cheap repair (not a scaling risk). All recorded in §6c.
  Added the **`de=2` binary-edge ablation**: `dataset.binary_edges` flag on
  `CoarseGraphDataset` (collapses the 11 weight classes to no-edge/edge on the
  fly, **reuses the same coarse cache** — no re-coarsening) +
  `configs/config_coarse_planar_de2.yaml`, which is byte-identical to
  `config_coarse_planar.yaml` except `name`, `binary_edges: true`, and
  `force_rebuild: false`. **NEXT:** compare `de=11` (running, r=0.8) vs `de=2` on
  `clustering MMD` + comparison PNGs → picks Architecture A vs B (§6c). Then
  build the Stage-D/E training-data extractor (`B_i`, `(B_i,B_j,w_ij)`, `e_i`,
  assert cluster connectivity) — needed regardless of which architecture wins.
- **2026-09-09** — Added `datasets/decoder_blocks.py`, the first Stage-D/E
  implementation layer. It losslessly partitions every fine edge into exact
  intra blocks `A_ii` and inter blocks `A_ij`, computes unbucketed `e_i/w_ij`,
  checks cluster connectivity and all node/edge/coarse-support invariants, and
  reconstructs the original graph as a round-trip assertion. New coarse caches
  store `source_index` so skipped graphs cannot misalign fine/coarse records;
  legacy caches remain usable when split lengths match. Added CPU-only stdlib
  tests in `tests/test_decoder_blocks.py`. Added `build_decoder_dataset.py` to
  materialize plain-dictionary `{train,val,test}.pt` decoder caches plus a
  `summary.json` containing cluster-size and exact `e_i/w_ij` distributions.
- **2026-09-09** — Real r=0.8 extraction passed on all 6000 planar graphs:
  594k intra blocks, ~1.69M inter blocks, mean/max cluster size 5.05/17,
  inter budget mean/max 2.57/10, and **zero disconnected clusters**. Added
  `datasets/decoder_regions.py`, defining the oracle local training contract:
  intra regions update all internal pairs; inter regions expose both fixed
  intra subgraphs and update only the bipartite rectangle; both carry exact
  edge budgets, global-node maps, cluster-side labels, and total fine degrees.
  Added `datasets/decoder_dataset.py`, an on-demand torch Dataset with compact
  numpy indices (rather than duplicating ~1.9M payloads) and a dense-local padded
  collator. Singleton intra clusters are skipped by default because they contain
  no edge variable; inter regions are all retained. Real torch/PyG validation is
  provided by the bounded `smoke_decoder_dataset.py` command for the user to run
  in the ML environment.
- **2026-09-09** — Added `diffusion/masked_edge_diffusion.py`. Its forward
  process noises only upper-triangular variables selected by each region's
  `update_mask`, preserves frozen context exactly, supports separate intra/inter
  limiting marginals, and computes per-region-balanced x0 edge CE. The bounded
  smoke command now checks symmetry, frozen context, finite loss, and backward.
- **2026-09-09** — Added `models/block_denoiser.py`, a shared local
  GraphTransformer for intra/inter regions. Inputs are all inference-available:
  noisy edges, update-vs-frozen mask, block kind, cluster side/size, current
  (not target) degree, time, and oracle edge budget. Ground-truth final degree is
  deliberately excluded to prevent leakage. `OracleRegionDataset` now computes
  exact separate intra/inter no-edge/edge marginals, and the bounded smoke test
  performs a full denoiser forward/loss/backward pass.
- **2026-09-09** — Added `train_block_decoder.py` and
  `configs/config_oracle_block_decoder.yaml`. Training is bounded by optimizer
  steps (not misleading 1.9M-region epochs), uses 50/50 intra/inter sampling,
  bounded infrequent validation with separate kind losses, no in-loop graph
  sampling/MMD, atomic periodic/best checkpoints, and resume support.
- **2026-09-09** — CUDA trainer smoke passed on the real cache (10 steps,
  batch 64): val loss 0.3745, intra/inter loss 0.4687/0.2803, edge accuracy
  0.7896; checkpointing completed. Treat these only as systems-test numbers.
  First observed throughput was ~0.59 step/s, so run a longer warm benchmark
  before committing to the default 20k-step budget. Mirrored the full current
  implementation status into `scalable_block_diffusion_brainstorm.md` §0.1.
- **2026-09-09** — 500-step calibration passed at batch 256: val loss 0.2801
  (intra/inter 0.3452/0.2150), accuracy 0.8849, stable ~6.8 steps/s. Added the
  analytic mask-preserving reverse posterior and full local reverse chain to
  `masked_edge_diffusion.py`; next validate it in the bounded ML smoke command,
  then evaluate sampled oracle blocks before extending training.
- **2026-09-09** — Reverse-chain ML smoke passed. Added
  `evaluate_block_decoder.py` for held-out oracle-region generation from a
  checkpoint. It reports intra/inter edge-budget MAE and exact-match rate,
  edge-count histogram TV, sample density, edge accuracy/Jaccard, generated
  intra connectivity, and inter non-emptiness while the sampler asserts frozen
  context and symmetry throughout.
- **2026-09-09** — v2 step-500 sampling beats step-200 on intra count MAE
  (0.70 vs 0.93) and count-histogram TV (0.195 vs 0.219), with 100% generated
  intra connectivity. Inter projection gives exact budgets/nonempty blocks by
  construction; endpoint Jaccard remains low (~0.085), so the next decisive test
  is whole-graph structure. Added `generate_oracle_graphs.py`: the complete
  two-phase oracle baseline samples all intra blocks, injects those *generated*
  interiors as frozen context for inter sampling, assembles the 500-node graph,
  asserts global cross-edge budgets, and reports connectivity/planarity/basic
  structural statistics.
- **2026-09-10** — Added `generate_alternating_oracle_graphs.py`, an
  inference-only Option-2 diagnostic over a single coherent fine-graph state.
  Each reverse step updates all intra blocks against one pre-phase global-degree
  vector, recomputes degrees, then updates all inter blocks against one
  post-intra vector. Chunking does not introduce autoregression. It deliberately
  uses the existing v2 local checkpoint first; results are diagnostic because
  that checkpoint was not trained on global alternating states.
- **2026-09-10** — Inference-only alternation on records 89/425 confirmed the
  expected train/test state mismatch: still 0/2 planar, with edges 1472→1207,
  clustering 0.248→0.138, transitivity 0.219→0.129 versus two-phase. The v2
  network interpreted global degrees using weights trained on local degrees and
  deleted intra edges. Do not treat this as an Option-2 result. Next build a
  graph-centric trainer that noises all blocks at a shared t, computes global
  state degrees, and trains the alternating kernels on their inference-time
  conditioning distribution.
- **2026-09-10** — Added graph-contextual alternating training:
  `datasets/alternating_training.py`, `train_alternating_decoder.py`, and
  `configs/config_alternating_decoder.yaml`. For each graph/shared t it samples
  blockwise `intra_t + inter_t` for the intra kernel and teacher-forced
  `intra_{t-1} + inter_t` for the inter kernel, aggregates full-graph degrees
  without an N² tensor, and selects balanced local losses from that coherent
  state. Supports v2 weight initialization, bounded steps/validation, resume,
  and checkpoints. Needs the bounded CUDA smoke run next.
- **2026-09-10** — The five-step graph-contextual CUDA smoke passed from the v2
  step-500 initialization: validation loss 0.3702 (intra 0.4033, inter 0.3372)
  with successful forward/backward, validation, and checkpointing. Treat these
  strictly as systems-test values. Next run a 500-step calibration, then use
  the alternating whole-graph generator on the same validation records 89/425.
- **2026-09-10** — The 500-step graph-contextual calibration completed at a
  stable ~5.27 steps/s. Validation was best at step 250: total 0.2570, intra
  0.3220, inter 0.1920. Step 500 was slightly worse at 0.2633/0.3257/0.2010, so
  select `experiments/alternating_decoder_v3_calibration/best.pt` (step 250)
  for matched whole-graph evaluation on records 89/425 before doing more
  training.
- **2026-09-10** — Matched step-250 alternation partially recovered the v2
  mismatch collapse but did not beat two-phase: mean edges 1,374, clustering
  0.198, transitivity 0.190, connectivity 1/2, planarity 0/2. Decomposition
  proved both failure sources independently: generated-intra + oracle-inter and
  oracle-intra + generated-inter were each non-planar for every run. Context v3
  had mean 4.5 non-planar and 4 disconnected intra blocks per graph. Added
  `planar_block_projection.py` plus `diagnose_block_failures.py --planar-project`:
  it repairs each intra block to connected-planar, realizes a coarse spanning
  tree first, then fills inter budgets greedily under global planarity while
  measuring edge retention and any infeasible budget deficit. This is the next
  feasibility diagnostic; global degree alone is insufficient.
- **2026-09-10** — Projection ablations validate the diagnosis. Oracle inputs
  pass unchanged with zero deficit. Generated-intra+oracle-inter loses 193.5
  of 720 inter edges; oracle-intra+generated-inter loses 277. Projection is
  therefore not an acceptable final decoder. The benchmark graphs are uniform
  2D point sets followed by Delaunay triangulation, but generation discarded
  positions. Added `generate_planar_graph_and_positions` and
  `recover_planar_positions.py`, which deterministically replays the original
  SeedSequence streams and refuses to save unless every regenerated edge set
  exactly matches the existing pickle. Next verify/recover validation positions,
  then prototype the geometry-preserving decoder.
- **2026-09-10** — Float64 position replay exactly reconstructs records 89/425
  via Delaunay (~4.6 ms/graph). Fresh 500-node samples match the dataset family
  (1,481 edges, clustering 0.436), and a 40k-node connected planar Delaunay graph
  with 119,975 edges took 6.09 s without dense adjacency. This is explicitly a
  provenance-aware ceiling, not a learned result. Added the lossless
  centroid/local-offset representation in `datasets/coordinate_blocks.py`, 3
  dependency-free tests (13 total pass), and `build_coordinate_dataset.py`.
  This is a planar-dataset diagnostic only, not the main model: the target method
  must support citation, ego, and arbitrary sparse graphs without coordinates.
- **2026-09-10** — Implemented generic sparse-global-context v4. At each shared
  reverse time, `datasets/alternating_training.py` packs two coherent full-graph
  phase states using only intra/coarse-supported candidate pairs. A new
  `SparseGlobalContextEncoder` performs index-add message passing on that sparse
  support; each local block gathers its fine-node embeddings and injects them
  into the existing shared denoiser. Complexity is linear in graph size for
  bounded blocks/coarse degree. The local v3 step-250 weights remain compatible;
  new context parameters initialize separately. Training, validation, and
  alternating inference now understand this context. Config:
  `configs/config_sparse_context_decoder.yaml`. Static parsing, diff checks, and
  all 13 dependency-free tests pass. The five-step CUDA smoke also passed:
  validation loss 0.3238 (intra 0.3757, inter 0.2720). The 500-step calibration
  will log the sparse encoder's gradient norm explicitly.
- **2026-09-10** — V4's 500-step calibration passed. Sparse-context gradients
  remained nonzero (roughly 0.0014–0.0398), confirming the new pathway learns;
  throughput stabilized near 2.96 steps/s. Best is step 250: validation 0.2535,
  intra 0.3179, inter 0.1891, a modest improvement over degree-only v3 at
  0.2570/0.3220/0.1920. Step 500 regressed to 0.2625. Select
  `experiments/sparse_context_decoder_v4_calibration/best.pt` for full-graph
  records 89/425 before doing any longer training.
- **2026-09-10** — V4 full graphs improve v3 modestly: mean edges 1,374→1,399,
  clustering 0.198→0.214, transitivity 0.190→0.196, connectivity 1/2→2/2;
  still 0/2 planar and below independent v2 clustering 0.248. Do not extend v4
  blindly. The historical weighted and binary coarse runs are not controlled:
  de=11 used r=0.9/normalized-self-loop/~50 nodes/5 layers, while de=2 used
  r=0.8/normalized/~100 nodes/10 layers. Decoder tests still use oracle exact
  `w_ij`; learned weights are unresolved. Next decouple them: binary diffusion
  for coarse topology plus a separate sparse positive-integer edge-weight head.
- **2026-09-10** — Matched step-250 alternating generation on records 89/425
  only partially repaired the mismatch: mean edges/clustering recovered from
  1,207/0.138 to 1,374/0.198, but stayed below two-phase 1,472/0.248 and target
  1,481.5/0.442. Connectivity was 1/2 and planarity 0/2. Added decomposition
  diagnostics and `diagnose_block_failures.py` to test saved graphs as
  generated-intra+oracle-inter and oracle-intra+generated-inter hybrids before
  choosing the next model change.
- **2026-09-10** — Static validation of the alternating trainer passed: all
  changed Python files parse, the 10 dependency-free decoder tests pass, and
  `git diff --check` is clean. Removed an unnecessary flat index over ~1.9M
  regions from alternating-trainer startup by computing exact marginals
  directly from graph records. Alternating generation metadata now identifies
  graph-context-trained checkpoints instead of labeling every run as the old
  v2 distribution-shift diagnostic. Next gate is a 5-step CUDA smoke run.
- **2026-09-09** — First two complete oracle two-phase graphs exposed the main
  independence failure: both have 500 nodes, near-correct edge count, and are
  connected, but **0/2 are planar**. Mean clustering fell 0.442→0.248 and
  transitivity 0.383→0.219. This is not an oracle-coarsening failure: connected
  cluster contraction preserves planarity, so oracle `G_c` is a planar minor.
  The arbitrary mutually independent fine endpoint choices make the expansion
  non-planar. Corrected `main.py` to include planarity in coarse-planar metrics.
  Next priority is contextual/alternating inter refinement, not longer v2
  training.
- **2026-09-09** — First step-500 sampling diagnostic: intra connectivity 100%,
  but intra/inter edge counts overshot (8.83 vs 8.30 and 3.19 vs 2.68); inter
  exact-budget rate 32.8%, nonempty 96.1%. Found and fixed oracle `e_i` leakage:
  the denoiser now receives a budget only for inter regions. Added stochastic
  Gumbel top-k final projection so inter samples obey the legitimate oracle
  `w_ij` blueprint exactly without making edge placement deterministic. The old
  checkpoint is diagnostic only; retrain the corrected model from scratch.
- **2026-09-10** — Implemented topology/weight separation for replacing oracle
  coarse-edge budgets. `models/coarse_weight_predictor.py` is a sparse
  message-passing encoder over binary `E_c` and cluster sizes with a symmetric
  categorical edge head. Dynamic masking guarantees positive integer outputs
  no larger than `v_i*v_j`; out-of-range training targets raise instead of
  silently clipping. Added graph-batch packing, a bounded trainer/checkpointer,
  and a v1 config. Static parsing, all 13 dependency-free tests, and
  `git diff --check` pass. Next gate is a fail-fast 5-step CUDA smoke followed
  by a 500-step calibration.
- **2026-09-10** — Coarse-weight smoke and 500-step calibration passed. Best
  validation is step 400: loss 1.4683, exact multiplicity 35.7%, MAE 0.993;
  step 500 is similar at 1.4686/35.5%/0.984, and throughput reached 52 steps/s.
  Since the curve plateaued, added `evaluate_coarse_weight_predictor.py` to
  score the full validation split against constant-mode and endpoint-size-mode
  baselines and export decoder-ready budgets. Alternating generation now accepts
  `--edge-budget-predictions`, validates every predicted capacity, and records
  budget provenance. `run_coarse_weight_evaluation.sh` performs both the full
  comparison and a v4 learned-budget decode on records 89/425 in one command.
- **2026-09-10** — Weight decision-rule calibration resolved the apparent mass
  deficit. Rounded posterior mean has full-val per-edge MAE 0.938 and matches
  mean weight (2.579 predicted vs 2.569 target), reducing mean absolute total
  inter-budget error to 6.16 edges (mode: 102.74). Two v4 decodes then retained
  734 inter edges and 1,422 total edges with 2/2 connectivity, but remained 0/2
  planar and low-clustering (0.203). Lock rounded mean for graph budgets; weight
  prediction is not the source of the remaining structural failure.
- **2026-09-10** — Corrected the scaling plan: one r=0.8 transition only expands
  ~100→500 nodes and cannot directly reach 5k/10k without making dense Stage B
  large again. Production scaling must recurse bounded transitions, e.g.
  80→400→2000→10000. Added machine-readable saving and `--num-samples` to
  coarse DiGress sampling plus `run_sample_binary_coarse.sh`; sample the existing
  binary checkpoint next, then build the per-level size head and multilevel
  transition cache.
- **2026-09-10** — The existing binary coarse DiGress checkpoint sampled poorly
  at epoch 70 (16 samples: planar accuracy 0%, clustering/triangle MMD
  0.542/0.729), but the run stopped at epoch 94 of its configured 5000. This is
  evidence only that the checkpoint is undertrained, not that DiGress should be
  rejected. Retracted the premature EDGE-switch proposal: DiGress remains the
  Stage-B backbone. Finish the size-prediction and generated-coarse adapter first,
  then resume Stage B with sampling-based monitoring and fully train the decoder.
- **2026-09-10** — Implemented Stage C and the first non-oracle end-to-end
  adapter. `CoarseClusterSizePredictor` sparsely message-passes over binary
  coarse topology, conditions on requested final N, and likelihood-projects
  positive integer sizes to sum exactly to N. `generated_blueprint_record`
  constructs the decoder contract from generated topology/sizes/weights with
  no fine oracle edges. `generate_end_to_end.py` composes saved DiGress samples,
  size prediction, rounded-mean weight prediction, and v4 alternating decoding.
  Added a fail-fast smoke→500-step size calibration→two-graph E2E runner. All
  Python parses, 17 dependency-free tests pass, shell parses, and diff check is
  clean. This first run is a systems diagnostic because Stage B and v4 are not
  fully trained.
- **2026-09-10** — Full coarse-weight validation (500 graphs/140,960 edges)
  beats both non-neural baselines: learned exact/MAE/RMSE is
  35.54%/0.991/1.373, constant-mode is 28.17%/1.569/2.074, and endpoint-size
  mode is 28.79%/1.411/1.890. Learned mean weight is biased low (2.205 vs 2.569),
  but whole-graph inter-budget MAE improves to 102.7 edges from 442.4/267.5.
  The first downstream run exposed that `make_inter_region` correctly rejected
  mutating cached oracle `w_ij`. Fixed the interface by retaining validation of
  oracle count and adding a distinct capacity-checked sampling-budget override;
  15 dependency-free tests now pass. Rerun only the two-graph decode.
- **2026-09-14** — Sampled-size end-to-end gate passed. Two generated-topology,
  non-oracle 500-node graphs were connected with 1421/1431 edges. Sampled and
  exact-N-projected cluster sizes restored the data distribution (ranges 1--12
  and 1--11; std 1.725/1.623 vs train 1.618), fixing rounded-mean collapse to
  5--6. Outputs remain non-planar and low-clustering because Stage B is epoch 70
  and v4 is step 250. Added resumable full-training runners: decoder to step
  5000 first, then binary DiGress from best epoch 70 to epoch 500. DiGress now
  records cheap connectivity/planarity/clustering sample monitors every 100
  epochs and resume starts at checkpoint epoch + 1. All 17 local tests pass.
- **2026-09-14** — Full v4 decoder run completed through step 5000; best is step
  4000 with validation 0.2348 (intra/inter 0.2958/0.1738), improving step 250's
  0.2535. Oracle-blueprint clustering rose 0.214→0.318. Fully non-oracle sampled-
  size outputs have 1430/1433 edges, 2/2 connectivity, and clustering
  0.322/0.338 (about 0.330 mean), but remain non-planar. This establishes useful
  learning from longer decoder training while confirming planarity is a global
  coherence issue, not merely undertraining. The DiGress resume runner now goes
  to epoch 500, evaluates 32 final coarse samples, and automatically decodes two
  through the full pipeline.
- **2026-09-28** — Reconstructed the completed Stage-B resume. It reached epoch
  499 (best validation checkpoint epoch 361, NLL 1705.29); there is no unfinished
  epoch-500 tranche to resume. Epoch 100/200/300/400 monitors were all connected
  but 0% planar. Final 32-sample test clustering MMD improved only 0.542→0.501
  from the old checkpoint, triangle MMD worsened 0.729→0.763, and planarity
  stayed 0%. Full-pipeline outputs with the step-4000 decoder had 1430/1419
  edges, clustering 0.320/0.341, 2/2 connectivity, and 0/2 planarity.
- **2026-09-28** — Reopened the coarse-weight design. V1 independently rounds
  per-edge categorical posterior means after node message passing. Although it
  matches marginal mean weight and total inter mass, it does not model incident
  weight sums `s_i=sum_j w_ij`, correlations among edges sharing a supernode,
  compatibility with generated intra blocks, or topology/size distribution
  shift. Next experiment is a four-way decoder ablation: oracle weights; no
  weight conditioning/projection; current independent head; then (only if
  explicit budgets help) joint sparse diffusion over weights on `E_c` with node-
  strength and total-mass objectives. Do not resume long training before this
  interface choice is resolved.
- **2026-09-28** — Proposed the next benchmark and inference design. Added
  `datasets/generate_sbm_ks.py`, a reproducible mixed-size balanced two-block
  SBM generator (64/128 nodes, 5,000/500/500 total graphs by default, expected
  degree 16, KS margin 7 > 1, randomized membership, split-leakage checks).
  Planarity is demoted from the primary development gate. Target node count must
  condition every expansion; the present size head's exact-N projection does
  not imply 500-to-10k extrapolation. Train bounded recursive levels such as
  80→400→2000→10000. CRM's current latent `u` is log-sociability, not community
  identity; the LSP/SBM extension should retain `u` for degree and add a
  diffused membership latent plus block-affinity prior for coarse connectivity.
- **2026-09-28** — Generated and validated the SBM collection: 5,000/500/500
  mixed 64/128-node graphs, realized split mean degrees 16.000/15.983/16.006,
  all connected. Added `config_coarse_sbm_ks_de2.yaml` and generalized the
  coarse loader/cache naming and metrics so SBM does not inherit planarity
  checks. Stage B now accepts a log-scaled requested fine-N graph condition,
  retains it through reverse diffusion, and pairs 64/128 targets with 13/26
  coarse nodes under r=0.8. Added bounded train/validation-batch and skip-final-
  evaluation CLI flags so the first CUDA check can be five batches, not epochs.

## 9. Files I have NOT yet read in depth
(ask the user for these if/when needed, per workflow preference — don't grep)
`diffusion/{noise_schedule,diffusion_utils,distributions,extra_features}.py`
(understood via usage), `metrics/*`, `analysis/spectre_utils.py` (MMD metrics),
`analysis/visualization.py`, `coarsen_mini/algorithms/loukas/{cost,matching}.py`,
`coarsen_mini/core/{laplacian,eig,conversions}.py`, `coarsen_mini/algorithms/graclus.py`.
