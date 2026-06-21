"""benchmark/core/laplacian.py — Laplacian builders and degree helpers.

Public:
  build_laplacian(data, laplacian_kind)         : PyG Data -> torch.sparse_csr_tensor
  degrees(A)                                    : row-sums of a scipy CSR adjacency
  shifted_degrees(deg, laplacian_kind)          : D or D+I depending on laplacian_kind
  normalize_laplacian_sp(A, deg, laplacian_kind): scipy version, used by the Loukas
                                                  multilevel loop where A_intermediate
                                                  stays in scipy

The three Laplacian conventions:
  - "combinatorial"        : L  = D - A
  - "normalized"           : L  = D^{-1/2}     (D - A) D^{-1/2}
  - "normalized_self_loop" : L  = (D+I)^{-1/2} (D - A) (D+I)^{-1/2}

Input contract for build_laplacian (caller responsibility, validated upstream
in datasets/loaders): graph undirected, no self-loops, single LCC. Nothing
in this file re-checks them.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch

from coarsen_mini.core.conversions import data_to_scipy_csr_adjacency, scipy_csr_to_torch_csr

LaplacianKind = Literal["combinatorial", "normalized", "normalized_self_loop"]


def degrees(A: sp.csr_matrix) -> np.ndarray:
    """Node degrees = row sums of A. Returns a 1D float64 array of length N."""
    return np.asarray(A.sum(axis=1)).ravel().astype(np.float64, copy=False)


def shifted_degrees(deg: np.ndarray, laplacian_kind: LaplacianKind) -> np.ndarray:
    """The degree vector inside the normalisation: D (combi/norm) or D+I (self-loop).

    For combinatorial, no normalisation is applied so this is unused; we return
    ones for API symmetry (caller will not multiply by it).
    """
    if laplacian_kind == "normalized_self_loop":
        return deg + 1.0
    elif laplacian_kind == "normalized":
        return deg
    elif laplacian_kind == "combinatorial":
        return np.ones_like(deg)
    else:
        raise ValueError(f"unknown laplacian_kind {laplacian_kind!r}")


def normalize_laplacian_sp(
    A: sp.csr_matrix, deg: np.ndarray, laplacian_kind: LaplacianKind
) -> sp.csr_matrix:
    """Build the chosen Laplacian as scipy CSR from a precomputed adjacency + degrees.

    Used by the Loukas multilevel loop where the intermediate adjacency lives in
    scipy and gets re-Laplacianised at every level. For the one-shot PyG->torch
    public path, use build_laplacian instead.
    """
    L_comb = (sp.diags(deg) - A).tocsr()
    if laplacian_kind == "combinatorial":
        return L_comb
    D_N = shifted_degrees(deg, laplacian_kind)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(D_N > 0, D_N ** -0.5, 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ L_comb @ D_inv_sqrt).tocsr()


def build_laplacian(
    data, laplacian_kind: LaplacianKind, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build the chosen Laplacian as a torch.sparse_csr_tensor.

    Default dtype float32 to match PyG convention. Override to float64 for
    metric computations that need precision (e.g. RSA eigenvalue analysis).
    """
    if laplacian_kind not in ("combinatorial", "normalized", "normalized_self_loop"):
        raise ValueError(
            "laplacian_kind must be combinatorial / normalized / normalized_self_loop, "
            f"got {laplacian_kind!r}"
        )
    A = data_to_scipy_csr_adjacency(data)
    deg = degrees(A)
    L = normalize_laplacian_sp(A, deg, laplacian_kind)
    return scipy_csr_to_torch_csr(L).to(dtype)
