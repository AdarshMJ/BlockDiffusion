"""benchmark/core/q.py — Q matrix helpers (pure torch sparse CSR, device-aware).

Public functions:

  coarsened_adjacency(A, Q_binary, ...)       : A_c = Q_b^T A Q_b
  Q_normalized_from_partition(Q_b, deg_orig, deg_c, laplacian_kind)
                                                : Q_norm[i,c] = sqrt(D_N[i]/D_n[c]) * Q_b[i,c]
  degrees_torch(A)                              : weighted row-sums

100% pure torch sparse_csr API. No torch.sparse_coo anywhere. Matmuls use
torch >= 2.0 native `csr @ csr`. Element-wise ops (drop diagonal, etc.) use
the csr_from_triplets / extract_triplets pattern from core/conversions.py.

Naming convention (papers):
  Q       = Q_normalized   (degree-aware, the one used for signal lifting)
  Q_binary = cluster indicator (binary one-per-row), used only to build A_c
  P       = pinv(Q_normalized), closed-form via pinv_well_partitioned_torch
"""

from __future__ import annotations

import numpy as np
import torch

from coarsen_mini.core.conversions import csr_from_triplets, extract_triplets
from coarsen_mini.core.laplacian import LaplacianKind, shifted_degrees


def degrees_torch(A: torch.Tensor) -> torch.Tensor:
    """Weighted row-sums of a torch sparse CSR adjacency. 1D, device-aware."""
    if A.layout != torch.sparse_csr:
        raise TypeError(f"A must be torch.sparse_csr, got {A.layout}")
    row, _, vals = extract_triplets(A)
    deg = torch.zeros(A.shape[0], dtype=vals.dtype, device=A.device)
    deg.scatter_add_(0, row, vals)
    return deg


def coarsened_adjacency(
    A: torch.Tensor,
    Q_binary: torch.Tensor,
    *,
    delete_diagonal: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Coarsened adjacency A_c = Q_binary^T A Q_binary, pure torch sparse CSR.

    Implementation notes:
      - `Q_binary.t()` returns a sparse_csc, but torch matmul natively accepts
        `csc @ csr @ csr` (verified Phase 1.A). So we use it direct without
        `.to_sparse_csr()` -- saves ~5x time on the conversion.
      - For symmetrize, addition `csr + csc` is NOT supported (RuntimeError),
        so we DO call `.to_sparse_csr()` once after the transpose of Ac.
    """
    if A.layout != torch.sparse_csr:
        raise TypeError(f"A must be torch.sparse_csr, got {A.layout}")
    if Q_binary.layout != torch.sparse_csr:
        raise TypeError(f"Q_binary must be torch.sparse_csr, got {Q_binary.layout}")
    if A.device != Q_binary.device:
        raise ValueError(
            f"device mismatch: A on {A.device}, Q_binary on {Q_binary.device}"
        )

    Ac = Q_binary.t() @ A @ Q_binary    # CSC @ CSR @ CSR -> CSR, native

    if delete_diagonal:
        Ac = _drop_diagonal(Ac)
    if symmetrize:
        # addition needs both operands in CSR (CSR + CSC is unsupported)
        Ac = (Ac + Ac.t().to_sparse_csr()) * 0.5
        if Ac.layout != torch.sparse_csr:
            Ac = Ac.to_sparse_csr()
    return Ac


def Q_normalized_from_partition(
    Q_binary: torch.Tensor,
    deg_orig: torch.Tensor,
    deg_coarsen: torch.Tensor,
    laplacian_kind: LaplacianKind,
) -> torch.Tensor:
    """Degree-aware lifting matrix (the Q used in papers).

        Q_norm[i, c] = sqrt(D_N[i] / D_n[c]) * Q_binary[i, c]
        D_N = deg_orig + (1 if laplacian_kind == "normalized_self_loop" else 0)
        D_n similarly for the coarsened graph.

    For laplacian_kind == "combinatorial", shifted_degrees returns ones, so Q_norm = Q_binary.
    """
    if Q_binary.layout != torch.sparse_csr:
        raise TypeError(f"Q_binary must be torch.sparse_csr, got {Q_binary.layout}")
    N, n = Q_binary.shape
    if deg_orig.shape != (N,):
        raise ValueError(f"deg_orig shape {tuple(deg_orig.shape)} != ({N},)")
    if deg_coarsen.shape != (n,):
        raise ValueError(f"deg_coarsen shape {tuple(deg_coarsen.shape)} != ({n},)")
    device = Q_binary.device
    dtype = Q_binary.values().dtype

    D_N_np = shifted_degrees(deg_orig.detach().cpu().numpy(), laplacian_kind)
    D_n_np = shifted_degrees(deg_coarsen.detach().cpu().numpy(), laplacian_kind)
    sqrt_D_N = torch.from_numpy(np.sqrt(D_N_np)).to(dtype=dtype, device=device)
    with np.errstate(divide="ignore"):
        sqrt_inv_D_n_np = np.where(D_n_np > 0, D_n_np ** -0.5, 0.0)
    sqrt_inv_D_n = torch.from_numpy(sqrt_inv_D_n_np).to(dtype=dtype, device=device)

    crow = Q_binary.crow_indices()
    col_idx = Q_binary.col_indices()
    vals = Q_binary.values()
    row_idx = torch.repeat_interleave(
        torch.arange(N, dtype=torch.int64, device=device), crow[1:] - crow[:-1]
    )
    new_vals = vals * sqrt_D_N[row_idx] * sqrt_inv_D_n[col_idx]
    # crow + col_idx already canonical (we kept the same ordering as Q_binary)
    return torch.sparse_csr_tensor(crow, col_idx, new_vals, size=(N, n))


# ---------------------------------------------------------------------------
# Internal pure-CSR helpers
# ---------------------------------------------------------------------------

def _drop_diagonal(M: torch.Tensor) -> torch.Tensor:
    """Drop entries where row == col, pure CSR API (no COO)."""
    row, col, vals = extract_triplets(M)
    keep = row != col
    return csr_from_triplets(
        row[keep], col[keep], vals[keep], M.shape[0], M.shape[1]
    )
