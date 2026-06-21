# coarsen_mini

Minimal, standalone implementations of two **graph coarsening** algorithms (Loukas spectral coarsening + Graclus), with helpers to build the lifting matrix `Q`, the projection `P = pinv(Q)`, and the coarsened graph `G_c = (A_c, X_c)`.

Lifted out of a larger benchmark codebase, this package is self-contained, torch-only (no DGL), and works on top of [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) `Data` objects.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=<parent_dir_of_coarsen_mini> python coarsen_mini/demo.py
```

The demo loads Cora and runs Loukas (3 variants) + Graclus at 50% reduction.

## What's inside

- **Loukas spectral coarsening** (`coarsen_loukas`) — multilevel variation-edges / variation-neighborhood
- **Graclus heavy-edge matching** (`coarsen_graclus`) — needs `torch-cluster`
- **Coarsening builder** (`build_coarsening`) — assembles `(Q, P, A_c, X_c)` from a Partition

## API summary

```python
from coarsen_mini.algorithms.loukas.main import coarsen_loukas
from coarsen_mini.algorithms.graclus import coarsen_graclus
from coarsen_mini.core.builders import build_coarsening

# Loukas — spectral approximation guarantees (Loukas 2019, JMLR)
partition, info = coarsen_loukas(
    data,                                # PyG Data
    r=0.5,                                # target reduction ratio (0..1)
    laplacian_kind="normalized_self_loop",   # see "Laplacian modes" below
    method="edges",                       # "edges" (default) | "neighborhood"
    K=100,                                # number of preserved eigenvectors
    progress=False,
)

# Graclus — fast O(N) heavy-edge matching (Dhillon et al. 2007)
partition, info = coarsen_graclus(data, r=0.5)

# Assemble the full coarsening (lifting / projection matrices + coarsened graph)
coars = build_coarsening(data, partition, method="loukas", r_requested=0.5)
# coars.Q (N, n) sparse CSR ; coars.P = pinv(Q) (n, N) ; coars.coarsened = PyG Data of G_c
```

`info` is a dict with `n_clusters`, `r_actual`, `laplacian_kind`, `method`, `K`, etc.

## Laplacian modes for Loukas (`laplacian_kind` kwarg)

Loukas's algorithm preserves the spectrum of **a specific Laplacian**. Two are typically used in coarsening for GNN settings — pass either via `laplacian_kind` :

| Mode | Definition | When to use |
|---|---|---|
| `"combinatorial"` | `L = D − A`                                | Classical graph spectral analysis. Smallest eigvalues = community structure. Use if you care about cuts / clustering. |
| `"normalized_self_loop"` (default) | `L̂ = (D + I)^{-1/2}(D − A)(D + I)^{-1/2}` | Matches the propagation matrix used by the **Kipf-Welling GCN** : `S = D̂^{-1/2}(A + I)D̂^{-1/2} = I − L̂`. Use if downstream task is a GCN. |

A third mode is also supported : `"normalized"` (= `D^{-1/2}(D − A)D^{-1/2}`, classical normalized Laplacian).

**Both are interesting** : combinatorial preserves community-cut structure, normalized_self_loop matches GCN propagation. The right choice depends on your downstream task.

## Coarsening methods for Loukas (`method` kwarg)

Loukas's paper proposes two ways of choosing which edges to contract at each multilevel step :

| Method | Description |
|---|---|
| `"edges"` (default, more standard) | Each candidate is a single edge ; cost = spectral approximation drop if this edge is contracted. Standard, well-tested. |
| `"neighborhood"` | Each candidate is a whole neighborhood (a node + all its neighbors). Tends to produce slightly different cluster shapes ; often a bit faster on graphs with low average degree. |

## References (for further reading / your Claude agent to look up)

- **Loukas, A. (2019).** *Graph reduction with spectral and cut guarantees.*
  Journal of Machine Learning Research, 20(116):1−42. → introduces the multilevel local-variation coarsening implemented in `algorithms/loukas/`. Defines the (R, ε)-restricted spectral approximation, and the variation-edges / variation-neighborhood matching strategies.

- **Dhillon, I., Guan, Y., Kulis, B. (2007).** *Weighted graph cuts without eigenvectors: A multilevel approach.*
  IEEE TPAMI, 29(11):1944–1957. → the original Graclus paper (heavy-edge matching, ratio-cut / normalized-cut multilevel scheme).

- **Kipf, T. N., Welling, M. (2017).** *Semi-supervised classification with graph convolutional networks.*
  ICLR. → GCN paper ; their propagation matrix `D̂^{-1/2}(A + I)D̂^{-1/2}` is exactly `I − L̂` where `L̂` is our `normalized_self_loop` Laplacian. Motivates the choice of `laplacian_kind="normalized_self_loop"` when targeting GCN training.

- **Hammond, D. K., Vandergheynst, P., Gribonval, R. (2011).** *Wavelets on graphs via spectral graph theory.* (background for spectral coarsening / signal smoothness on graphs).

### Related papers from the Joly group (theoretical context for this package)

- **Joly, A., Keriven, N., et al. (2024).** *Graph Coarsening with Message-Passing Guarantees.*
  NeurIPS 2024. → Bounds the GNN propagation error after coarsening by an expression involving Loukas's RSA constant (Theorem 4.1). Motivates why `laplacian_kind="normalized_self_loop"` is the right choice for GCN downstream tasks : the bound is sharp when the preserved Laplacian matches the GCN propagation operator.

- **Joly, A., et al. (2025).** *A Taxonomy of Reduction Matrices for Graph Coarsening.*
  NeurIPS 2025 (accepted). → Defines three nested admissible sets E1 ⊃ E2 ⊃ E3 for the projection matrix P given a fixed lifting matrix Q. The `pinv_well_partitioned` P returned by `build_coarsening` here lies in E3 (it is the Moore-Penrose pseudo-inverse of Q restricted to cluster support). The paper's optimized P (in E2 or E3 via gradient descent) is NOT included in this minimal package — see the parent benchmark for that.

- **Joly, A., et al. (2025+).** *CoRe-GNN: Coarsen and Restore — Multilevel Message Passing on Coarsened Graphs.*
  NeurIPS 2025 submission. → Dual-path GNN that runs in parallel an intra-cluster propagation on the original graph (using the partition from Loukas / Graclus) and an inter-cluster propagation on the coarsened graph (using the `Q`, `P` from `build_coarsening`). The minimal package here gives exactly the (Q, P, A_c, X_c) primitives needed to implement CoRe-GNN downstream.

## File layout

```
coarsen_mini/
├── README.md                    ← this file
├── demo.py                      ← Cora demo (loads → coarsens → prints)
├── requirements.txt
├── algorithms/
│   ├── common.py                ← resolve_target (r / n_clusters dispatch)
│   ├── graclus.py               ← Graclus wrapper (needs torch-cluster)
│   └── loukas/
│       ├── main.py              ← coarsen_loukas (multilevel main loop)
│       ├── cost.py              ← spectral approximation cost per candidate
│       └── matching.py          ← greedy non-overlapping matching
└── core/
    ├── builders.py              ← build_coarsening : assemble Coarsening from Partition
    ├── coarsening.py            ← Coarsening dataclass
    ├── conversions.py           ← Data ↔ torch sparse CSR helpers
    ├── eig.py                   ← smallest_eigenpairs (scipy ARPACK under the hood)
    ├── laplacian.py             ← build_laplacian (combinatorial / normalized / nsl)
    ├── partition.py             ← Partition class + pinv_well_partitioned_torch
    └── q.py                     ← Q_normalized_from_partition, coarsened_adjacency, ...
```

## Notes for your Claude agent

- All public matrices are stored as **torch sparse CSR** (no `torch.sparse_coo` anywhere).
- Default dtype is `float32` to match `torch_geometric` / `GCNConv` conventions ; spectral kernels (`eig.py`) promote to `float64` internally for numerical stability of the eigendecomposition.
- The Loukas implementation is a clean port of the original (Loukas, JMLR 2019) ; it was validated against the original scipy implementation to numerical parity at K=100 (the typical paper setting).
- `build_coarsening` builds `Q` with the normalization consistent with the chosen `laplacian_kind` (binary `Q` for combinatorial, degree-weighted `Q` for normalized variants). `P` is the closed-form Moore-Penrose pseudo-inverse exploiting the well-partitioned structure of `Q`.
