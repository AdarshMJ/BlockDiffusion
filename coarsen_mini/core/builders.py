"""benchmark/core/builders.py — Coarsening orchestrator (pure torch).

`build_coarsening(data, partition, ...)` takes any algo's output (a Partition)
and assembles the full Coarsening dataclass. Everything in pure torch sparse
CSR, device-aware.

Example
-------
    from coarsen_mini.algorithms.loukas.main import coarsen_loukas
    from coarsen_mini.core.builders import build_coarsening

    partition, info = coarsen_loukas(data, r=0.5, method="edges")
    coarsening = build_coarsening(
        data, partition,
        laplacian_kind="normalized_self_loop",
        method="loukas_edges",
        r_requested=0.5,
        P_loukas=info["P_loukas"],
        extras={"R": info["R"], "info": info},
    )
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from coarsen_mini.core.coarsening import Coarsening
from coarsen_mini.core.conversions import (
    data_to_torch_csr_adjacency,
    extract_triplets,
)
from coarsen_mini.core.laplacian import LaplacianKind
from coarsen_mini.core.partition import Partition, pinv_well_partitioned_torch
from coarsen_mini.core.q import (
    Q_normalized_from_partition,
    coarsened_adjacency,
    degrees_torch,
)


def build_coarsening(
    data,
    partition: Partition,
    *,
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    method: str = "unknown",
    r_requested: Optional[float] = None,
    P_loukas: Optional[torch.Tensor] = None,
    extras: Optional[dict] = None,
    delete_diagonal: bool = True,
    dtype: torch.dtype = torch.float32,        # PyG / training convention
) -> Coarsening:
    """Build a complete Coarsening from a Partition and the original PyG Data.

    Pipeline (all pure torch sparse CSR):
      1. Q_binary = partition.to_Q_combi()              (local, not stored)
      2. A_orig   = data_to_torch_csr_adjacency(data)
      3. A_c      = coarsened_adjacency(A_orig, Q_binary)
      4. deg_orig = degrees_torch(A_orig); deg_c = degrees_torch(A_c)
      5. Q        = Q_normalized_from_partition(Q_binary, deg_orig, deg_c, laplacian_kind)
      6. P        = pinv_well_partitioned_torch(Q)      (the Q here is Q_normalized)
      7. x_c      = P @ x_orig                          (cluster-mean under Q_normalized)
      8. Wrap (A_c, x_c) in a fresh PyG Data; pack the rest in Coarsening.
    """
    if data.num_nodes != partition.N:
        raise ValueError(
            f"data.num_nodes ({data.num_nodes}) != partition.N ({partition.N})"
        )

    # Build everything natively in `dtype` from the start — no later cast needed.
    Q_binary = partition.to_Q_combi(dtype=dtype)             # (N, n) sparse CSR
    A_orig = data_to_torch_csr_adjacency(data, dtype=dtype)  # (N, N) sparse CSR

    A_c = coarsened_adjacency(
        A_orig, Q_binary, delete_diagonal=delete_diagonal, symmetrize=True
    )
    deg_orig = degrees_torch(A_orig)
    deg_c = degrees_torch(A_c)
    Q = Q_normalized_from_partition(Q_binary, deg_orig, deg_c, laplacian_kind)
    P = pinv_well_partitioned_torch(Q)

    # Coarsened features = P @ x. x already matches dtype (or trivially castable).
    x_c = (P @ data.x.to(dtype)) if data.x is not None else None

    # Coarsened PyG Data: no y, no masks. Per-split logic stays at trainer level.
    Ac_row, Ac_col, Ac_vals = extract_triplets(A_c)
    coarsened = Data(
        x=x_c,
        edge_index=torch.stack([Ac_row, Ac_col]),
        edge_weight=Ac_vals,
        num_nodes=partition.n_clusters,
    )

    r_actual = 1.0 - partition.n_clusters / partition.N
    r_req = r_actual if r_requested is None else r_requested

    return Coarsening(
        original=data,
        coarsened=coarsened,
        partition=partition,
        Q=Q,
        P=P,
        P_loukas=P_loukas.to(dtype) if P_loukas is not None else None,  # cast only if external
        method=method,
        r_requested=r_req,
        r_actual=r_actual,
        laplacian_kind=laplacian_kind,
        extras=extras if extras is not None else {},
    )


def convert_laplacian_kind_coarsening(
    coarsening: Coarsening,
    new_laplacian_kind: LaplacianKind,
) -> Coarsening:
    """Rebuild Q and P of a Coarsening under a different laplacian_kind.

    For each laplacian_kind, Q is built with the matching normalization (see
    `Q_normalized_from_partition`) and P is its Moore-Penrose pseudo-inverse:

        combinatorial         : Q = Q_binary (no degree weighting)
        normalized            : Q[i,c] = sqrt(d_orig[i] / d_c[c]) · Q_binary[i,c]
        normalized_self_loop  : Q[i,c] = sqrt((d_orig[i]+1) / (d_c[c]+1)) · Q_binary[i,c]

    All other Coarsening fields are preserved (original, coarsened, partition,
    P_loukas, method, r_*, extras). P_loukas tracks the partition refinement
    chain — independent of the Q normalization, so it stays valid under
    conversion.

    Use case : same partition, evaluate metrics under multiple Laplacian
    conventions without re-running the (expensive) coarsening algorithm.
    """
    if new_laplacian_kind == coarsening.laplacian_kind:
        return coarsening

    dtype = coarsening.Q.values().dtype if coarsening.Q.layout == torch.sparse_csr else coarsening.Q.dtype
    Q_binary = coarsening.partition.to_Q_combi(dtype=dtype)
    # Recompute A_orig + A_c (low overhead, also a sanity check that
    # coarsening.coarsened's adjacency matches what we'd derive from partition).
    A_orig = data_to_torch_csr_adjacency(coarsening.original, dtype=dtype)
    A_c = coarsened_adjacency(A_orig, Q_binary, delete_diagonal=True, symmetrize=True)
    deg_orig = degrees_torch(A_orig)
    deg_c = degrees_torch(A_c)
    new_Q = Q_normalized_from_partition(Q_binary, deg_orig, deg_c, new_laplacian_kind)
    new_P = pinv_well_partitioned_torch(new_Q)

    return Coarsening(
        original=coarsening.original,
        coarsened=coarsening.coarsened,
        partition=coarsening.partition,
        Q=new_Q,
        P=new_P,
        P_loukas=coarsening.P_loukas,
        method=coarsening.method,
        r_requested=coarsening.r_requested,
        r_actual=coarsening.r_actual,
        laplacian_kind=new_laplacian_kind,
        extras=coarsening.extras,
    )
