# BlockDiffusion Uncoarsener Tutorial

This document explains the uncoarsener used in this repository: what information
it receives, what its three learned components do, how the alternating diffusion
decoder is trained, and how a generated coarse graph becomes a full graph.

The short version is:

```text
binary coarse graph Gc + requested final size N
              |
              +--> predict one size vi for every coarse node
              |
              +--> predict one edge count wij for every coarse edge
              |
              +--> diffuse edges inside every cluster (intra)
              |
              +--> diffuse edges between adjacent clusters (inter)
              |
              +--> refine inter edges while keeping each wij fixed
              v
        final N-node graph G
```

The uncoarsener is not one monolithic network. It is a factorization into two
small graph predictors and one shared local diffusion model.

---

## 1. The most important distinction: SBM communities are not coarsening clusters

Our current data are balanced two-community SBM graphs. Each original node has
an SBM community label, but the Loukas coarsener does not use that label as the
definition of a cluster.

There are therefore two different partitions:

1. **SBM communities:** the two large latent groups that define the data
   distribution.
2. **Coarsening clusters:** small groups of nearby original nodes contracted
   into individual supernodes.

For an (N=64) graph, the current coarsener normally produces 13 supernodes.
Those 13 supernodes are not the two SBM communities. Each supernode represents
roughly five original nodes. The two-community structure should instead appear
as organization among the 13 supernodes and should survive when they are
expanded.

This distinction matters because "intra" and "inter" in the decoder mean:

- **intra:** inside one coarsening cluster (B_i);
- **inter:** between two coarsening clusters (B_i) and (B_j).

They do not directly mean inside/across the two SBM communities.

---

## 2. What coarsening gives us during training

Start with a real fine graph (G=(V,E)). Coarsening produces:

- a binary coarse graph (G_c=(V_c,E_c));
- an assignment (a:V\rightarrow V_c), saying which fine node belongs to
  each supernode;
- a cluster size
  \[
  v_i = |B_i|;
  \]
- an exact coarse-edge multiplicity
  \[
  w_{ij}=|\{(u,v)\in E:u\in B_i,v\in B_j\}|.
  \]

The binary coarse adjacency answers only:

> Is there at least one fine edge between clusters (B_i) and (B_j)?

The weight (w_{ij}) answers:

> Exactly how many fine edges cross that cluster pair?

For each fine training graph, `build_decoder_dataset.py` stores a decoder
record containing:

```text
n_orig             final number of fine nodes
n_c                number of coarse nodes
assignment         fine-node -> coarse-node mapping
cluster_sizes      [v0, v1, ..., v(nc-1)]
intra_blocks       original edges inside each Bi
inter_blocks       original edges between Bi and Bj
w_ij               exact cross-edge count for each coarse edge
```

For the current SBM cache:

- 5,000 training graphs produce 95,000 intra blocks;
- they produce 556,897 inter blocks;
- cluster sizes range from 2 to 11, with mean about 5.05;
- inter weights range from 1 to 33, with mean about 5.59;
- every stored coarsening cluster is connected.

The implementation is in:

- `build_decoder_dataset.py`
- `datasets/decoder_blocks.py`
- `datasets/decoder_regions.py`

---

## 3. A small worked example

Suppose a coarse graph has three supernodes:

```text
      B0 -------- B1 -------- B2
```

Assume the requested final graph has eight nodes, and the true/predicted sizes
are:

```text
v = [3, 2, 3]
```

The fine-node sets are then:

```text
B0 = {0,1,2}
B1 = {3,4}
B2 = {5,6,7}
```

Suppose the predicted weights are:

```text
w01 = 2
w12 = 3
```

The decoder must solve five local problems:

```text
intra B0: choose edges among 3 nodes
intra B1: choose edges among 2 nodes
intra B2: choose edges among 3 nodes
inter B0-B1: choose exactly 2 of the 3*2 possible cross edges
inter B1-B2: choose exactly 3 of the 2*3 possible cross edges
```

There is no (B_0\)-(B_2) inter problem because ((0,2)\notin E_c). This is
where the scalability comes from: the decoder never considers cross-cluster
pairs unsupported by the coarse graph.

---

## 4. Component A: predicting cluster sizes

At inference, DiGress produces only binary (G_c). It does not tell us how many
fine nodes each supernode represents. The size predictor models

\[
p_\phi(v_i\mid G_c,N).
\]

Its inputs include coarse degree, requested (N), number of coarse nodes, and
graph-level degree normalization. A sparse message-passing network produces a
categorical distribution over positive sizes (1,\ldots,16).

Naively sampling every (v_i) independently would usually give the wrong total
number of nodes. After prediction, we therefore project the sizes so that

\[
v_i\ge 1,\qquad \sum_i v_i=N.
\]

The projection changes one node at a time, choosing the least costly categorical
change under the model probabilities. Thus the final graph always has exactly
the requested number of nodes.

Current result on SBM validation:

- raw MAE: about 0.97 node per cluster;
- projected MAE: about 0.93;
- total-size error after projection: exactly zero.

Implementation:

- `models/coarse_size_predictor.py`
- `datasets/coarse_size_dataset.py`
- `train_coarse_size_predictor.py`

Checkpoint:

```text
experiments/coarse_size_predictor_sbm_v1/best.pt
```

---

## 5. Component B: predicting coarse-edge multiplicities

A binary edge ((i,j)\in E_c) only says that at least one fine edge should exist
between (B_i) and (B_j). The weight predictor models

\[
p_\omega(w_{ij}\mid G_c,v_i,v_j).
\]

It message-passes over the sparse coarse graph and predicts a categorical value
from 1 to 40 for every present coarse edge. Classes larger than the endpoint
capacity (v_i v_j) are masked out, so impossible weights cannot be selected.

The current end-to-end default is the categorical mode because, on the SBM
validation cache, it gave the best whole-graph inter-edge-budget calibration.

The weight is used twice:

1. as a conditioning feature for the inter diffusion model;
2. as a final exact cardinality constraint.

At the final inter step, the decoder selects exactly (w_{ij}) cross edges using
Gumbel-perturbed model scores and top-(k) projection. The model chooses *which*
edges exist, while (w_{ij}) fixes *how many* exist.

Implementation:

- `models/coarse_weight_predictor.py`
- `datasets/coarse_weight_dataset.py`
- `train_coarse_weight_predictor.py`

Checkpoint:

```text
experiments/coarse_weight_predictor_sbm_v1/best.pt
```

---

## 6. Component C: the shared masked diffusion decoder

The edge generator is a shared graph-transformer denoiser. "Shared" means the
same network handles both intra and inter regions. A region-kind feature tells
it which task it is solving.

### 6.1 Intra region

For one cluster (B_i) with (v_i) nodes, the local adjacency is
(v_i\times v_i). Every off-diagonal pair is an update variable:

```text
update_mask = all off-diagonal pairs
```

The model learns the number and arrangement of intra edges. The cached oracle
intra edge count (e_i) is deliberately not supplied as a generation budget,
because it would not be available for a newly generated coarse graph.

### 6.2 Inter region

For a coarse edge ((i,j)), concatenate the two node sets:

```text
[nodes of Bi | nodes of Bj]
```

The local adjacency has four blocks:

```text
             Bi             Bj
       +-------------+-------------+
 Bi    | frozen A_ii | update A_ij |
       +-------------+-------------+
 Bj    | update A_ji | frozen A_jj |
       +-------------+-------------+
```

Only the bipartite cross rectangle is noised and predicted. The already
generated intra edges are visible conditioning context and cannot be changed by
the inter update.

This behavior is implemented by `update_mask` in
`diffusion/masked_edge_diffusion.py`.

### 6.3 Model features

For each local node, the denoiser receives:

- a constant active-node feature;
- a one-hot left/right side indicator;
- current noisy total degree;
- size of its cluster side;
- sparse whole-graph context from the current assembled graph.

For each local edge, it receives:

- noisy binary edge state;
- update-versus-frozen indicator.

Graph-level conditioning includes:

- diffusion time;
- intra/inter kind;
- (w_{ij}) for inter regions (zero for intra regions);
- region size;
- fraction of nodes on the left side.

The core network is `models/block_denoiser.py`. The sparse context encoder is
`models/sparse_global_context.py`.

---

## 7. Why training alternates intra and inter

If every local block were trained independently, different blocks incident to
the same fine node could make incompatible degree decisions. The alternating
trainer exposes each local prediction to the current state of the whole graph
without constructing a dense (N\times N) global adjacency.

For each sampled training graph and diffusion time (t):

### Phase A: train the intra update

1. Noise all intra blocks to time (t).
2. Noise all inter blocks to time (t).
3. Assemble their current global degrees and sparse graph context.
4. Predict clean intra edges.

All selected intra regions see the same pre-update global state, so they remain
parallel rather than becoming secretly autoregressive.

### Phase B: train the inter update

1. Move the intra context to (t-1) using the known forward process
   (teacher forcing).
2. Keep inter blocks at time (t).
3. Recompute global degrees and sparse context.
4. Predict clean inter edges while seeing the newer intra state.

This teaches the inference order:

```text
intra update -> recompute context -> inter update
```

The loss is cross-entropy only over entries selected by `update_mask`. Each
region is averaged separately before regions are averaged together, so a large
cluster pair does not dominate merely because it contains more candidate edges.

Implementation:

- `datasets/alternating_training.py`
- `train_alternating_decoder.py`

Current SBM checkpoint:

```text
experiments/sparse_context_decoder_sbm_v1/best.pt
```

It was selected at step 4,500 with validation loss 0.333612.

---

## 8. Inference: assembling a graph from local reverse processes

Given generated (G_c), predicted (v), and predicted (w):

1. Create disjoint fine-node ranges for all clusters.
2. Initialize every intra block from the learned intra edge marginal.
3. Initialize every supported inter block from the learned inter edge marginal.
4. At each reverse level:
   - update every intra block in parallel;
   - assemble the new global degrees/context;
   - update every inter block in parallel.
5. At the final inter level, enforce every exact (w_{ij}).
6. Copy all local blocks into one global adjacency matrix.

No inter region is created for a non-edge of (G_c). Chunking regions into GPU
batches is only a memory optimization; context is not updated between chunks,
so chunks within a phase stay conditionally parallel.

The inference driver is `generate_alternating_oracle_graphs.py`. Despite its
historical filename, the same `generate_alternating` function is used by the
fully predicted end-to-end pipeline.

---

## 9. Respacing and the extra inter-refinement pass

The decoder was trained with 200 diffusion times, but inference does not need to
evaluate every adjacent step. We implemented the exact discrete posterior for a
jump (t\rightarrow s<t), allowing a valid respaced chain.

The provisional schedule is:

```text
50 respaced levels: alternate intra then inter
10 extra levels:   re-noise only inter edges to t=40 and refine them
```

Configuration:

```text
configs/config_refinement_schedule_sbm.yaml
```

The refinement keeps intra edges frozen and preserves every (w_{ij}). It only
rearranges which cross-edge positions are active. On 16 paired validation
graphs, it improved degree-spread, clustering, and transitivity errors while the
50+10 sampler remained about 3.5 times faster than the original 200-level chain.

The schedule is configurable through:

- phases: `intra`, `inter`, or both;
- noise depth;
- number of reverse evaluations;
- uniform, quadratic, or explicit time grids;
- number of repeats.

Each run saves edge flips, predicted-(x_0) entropy, confidence, and pre/post
pass graph statistics in `sampling_diagnostics.json`.

---

## 10. Oracle evaluation versus real inference

These two modes answer different questions.

### Oracle blueprint diagnostic

```text
held-out real G
  -> true coarsening Gc
  -> true vi and wij
  -> decoder
```

This isolates whether the block diffusion decoder can reconstruct realistic
fine structure when its blueprint is correct.

### Fully predicted inference

```text
requested N
  -> noisy coarse adjacency
  -> DiGress-generated Gc
  -> predicted vi
  -> predicted wij
  -> decoder
```

This measures the complete system and includes error propagation from every
stage. The epoch-50 experiments are in this second category; they are not oracle
decodes.

The downstream models were trained on real coarsened graphs, so generated
coarse graphs create a possible distribution shift. We deliberately keep the
uncoarsener fixed while comparing Stage-B checkpoints so that improvements can
be attributed to the coarse generator rather than to a moving downstream model.

---

## 11. Is the system jointly end-to-end trained?

No. It is an end-to-end inference pipeline assembled from separately trained
components:

```text
Stage B DiGress             trained on real coarse graphs
size predictor              trained on true cluster sizes
weight predictor            trained on true wij values
alternating decoder         trained on true local fine blocks
```

At inference, their predictions are composed from noise to a final fine graph.
There is no gradient propagated through discrete (G_c\), (v), and (w) from
the final graph back into Stage B.

This modular baseline is intentional. It lets us identify whether an error came
from coarse topology, sizes, weights, or local decoding. Robust or joint
fine-tuning can be considered after that controlled diagnosis.

---

## 12. Why this design can scale

A dense fine-graph diffusion considers (O(N^2)) possible edges at every step.
The uncoarsener considers only:

\[
\sum_i O(v_i^2)
\quad+\quad
\sum_{(i,j)\in E_c} O(v_i v_j).
\]

When cluster sizes remain bounded and (G_c) is sparse, this grows roughly with
the number of supported coarse edges rather than with all fine-node pairs.

Local tensors are dense because their regions are small. Global scalability
comes from never instantiating unsupported cluster pairs and from using a sparse
context encoder—not from making every small local operation sparse.

---

## 13. Running the complete pipeline

The single entrypoint is:

```bash
python run_full_pipeline.py \
  --target-n 64 \
  --num-graphs 8 \
  --coarse-checkpoint experiments/coarse_sbm_ks_de2_full/checkpoints/epoch_50.pt \
  --seed 2050 \
  --device cuda \
  --output-dir experiments/full_pipeline_sbm_epoch50_n64
```

It performs:

1. conditioned coarse DiGress sampling;
2. exact-total size prediction;
3. capacity-masked mode weight prediction;
4. 50+10 alternating decoding;
5. graph-statistic calculation;
6. PNG visualization;
7. manifest and diagnostic saving.

Important outputs:

```text
coarse/graphs/generated_samples.pt     generated Gc samples
fine/generated_graphs.pt              final fine graphs
fine/metrics.json                     graph summaries
fine/sampling_diagnostics.json        reverse/refinement traces
fine/visualizations/graph_*.png       viewable final graphs
pipeline_manifest.json                exact checkpoints/configuration used
```

---

## 14. What is learned, predicted, and enforced?

| Quantity | Source at inference | Hard constraint? |
|---|---|---|
| requested (N) | user | yes |
| coarse node count (n_c) | empirical (p(n_c\mid N)) | sampled from trained support |
| coarse topology (G_c) | DiGress | binary/simple graph representation |
| cluster sizes (v_i) | size predictor | positive and sum exactly to (N) |
| coarse weights (w_{ij}) | weight predictor | (1\le w_{ij}\le v_i v_j) |
| intra-edge arrangement | block diffusion | simple undirected only |
| inter-edge arrangement | block diffusion | exactly (w_{ij}) edges per coarse edge |
| unsupported cross edges | not instantiated | always absent |

---

## 15. Current limitations

1. The current learned size support is only (N\in\{64,128\}). This does not
   demonstrate extrapolation to 5k or 10k nodes.
2. The current size and weight heads make mostly local/sparse predictions; a
   future joint weight model may coordinate incident (w_{ij}) values better.
3. The decoder is trained on real coarse blueprints and evaluated on generated
   ones, creating distribution shift.
4. Exact (w_{ij}) projection guarantees counts, not that those counts are
   globally optimal.
5. The current pipeline is one-level. Scaling to 10k will likely require
   bounded recursive expansions rather than one enormous expansion ratio.
6. The future structured-prior/LSP idea concerns Stage B: replace independent
   marginal coarse noise with a structurally informed coarse prior. It does not
   replace the uncoarsener described here.

---

## 16. A useful mental model

Think of the uncoarsener as an architect working from a blueprint:

- (G_c) says which rooms are allowed to have doors between them.
- (v_i) says how large each room is.
- (w_{ij}) says how many doors connect two adjacent rooms.
- intra diffusion designs each room's internal structure.
- inter diffusion chooses the precise door locations.
- alternating context lets each local decision see the evolving whole building.
- refinement relocates doors that are locally valid but globally awkward.

The blueprint controls global support and budgets; diffusion supplies the
fine-grained combinatorial arrangement.
