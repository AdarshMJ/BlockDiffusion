"""benchmark/core/coarsening.py — Coarsening dataclass.

The canonical artifact returned by `benchmark.core.builders.build_coarsening`,
consumed by metrics, protocols, IO. Built ONCE per (dataset, algo,
coarsening_hyperparams); training hyperparams vary on top of the same Coarsening.

Naming convention (matches Antonin's papers)
--------------------------------------------
Q       : torch.sparse_csr_tensor of shape (N, n), values = sqrt(D_N[i]/D_n[c]).
          This is THE Q used in every math operation (signal lifting Q @ x_c,
          coarsened propagation S_c = P S Q, etc.). Mathematically the
          degree-aware normalised lifting.

P       : torch.sparse_csr_tensor of shape (n, N), = pinv(Q). Closed-form via
          pinv_well_partitioned_torch (works for any Q with one nonzero per
          row, including Q_normalized which is not binary).

Q_binary : NOT stored as a field. It is the cluster indicator (one 1 per row),
          used internally only to build A_c (the coarsened adjacency). Recover
          on demand via `coarsening.partition.to_Q_combi()`; never needed for
          signal lifting in training / inference.

P_loukas : Optional. The multilevel chain of pinv(intermediate_Q_norm) produced
          by Loukas. Cannot be recovered from the partition alone. None for other
          algorithms.

Multi-split masks (chameleon, squirrel, Roman-empire, Amazon-ratings, ...) :
  data.train_mask may be 1D (N,) for single split (Planetoid) or 2D (N, n_splits)
  for HeterophilousGraphDataset. Coarsening is split-invariant: same Coarsening
  reused across all splits. The trainer iterates over `n_splits` at training time.

Helpers
-------
N, n, n_splits : shortcut properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from coarsen_mini.core.laplacian import LaplacianKind
from coarsen_mini.core.partition import Partition


@dataclass
class Coarsening:
    original: "torch_geometric.data.Data"
    coarsened: "torch_geometric.data.Data"
    partition: Partition

    Q: torch.Tensor                                   # = Q_normalized (papers convention)
    P: torch.Tensor                                   # = pinv(Q) closed form
    P_loukas: Optional[torch.Tensor] = None           # multilevel chain, Loukas only

    method: str = "unknown"
    r_requested: float = 0.0
    r_actual: float = 0.0
    laplacian_kind: LaplacianKind = "normalized_self_loop"

    extras: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.partition.N

    @property
    def n(self) -> int:
        return self.partition.n_clusters

    @property
    def n_splits(self) -> int:
        m = self.original.train_mask
        if m.ndim == 1:
            return 1
        elif m.ndim == 2:
            return int(m.shape[1])
        else:
            raise ValueError(f"train_mask has unexpected ndim={m.ndim}; expected 1 or 2")

    def __repr__(self) -> str:
        return (
            f"Coarsening(method={self.method!r}, N={self.N}, n={self.n}, "
            f"r_requested={self.r_requested:.3f}, r_actual={self.r_actual:.3f}, "
            f"n_splits={self.n_splits}, laplacian_kind={self.laplacian_kind!r}, "
            f"has_P_loukas={self.P_loukas is not None})"
        )
