"""benchmark/core/eig.py — smallest eigenpairs of a Laplacian.

Hybrid CPU kernel, chosen empirically (see DESIGN.md §1 Phase 0.5 bench on
Cora LCC):

  - "combinatorial"        : shift-invert (eigsh sigma=-1e-3, which='LM').
                             omega = 2(max_deg+1) explodes for high-degree
                             graphs and makes the reflection trick unstable.
                             Shift-invert factorises (L - sigma*I) once and
                             back-substitutes each iteration; uniformly
                             faster on combinatorial L.
  - "normalized"           : spectral reflection (eigsh on omega*I - L, "LM").
                             Spectrum bounded by 2, so omega=3 is well-
                             conditioned; LU upfront of shift-invert would not
                             amortise.
  - "normalized_self_loop" : same as "normalized".

torch.lobpcg was benched and dropped (30-400x slower than scipy on CPU; less
numerically stable on ill-conditioned Laplacians).

Input:
  L              : torch.sparse_csr_tensor (output of benchmark.core.laplacian.build_laplacian)
  k              : int, number of smallest eigenpairs to return
  laplacian_kind : LaplacianKind, must match the one used to build L
  delta          : float, optional regularisation (delta * I added to L before eig).
          Used for the "eigenvectors_delta_in" Loukas option that pushes the
          trivial 0 eigenvalue away. Default 0.0 (= official Loukas).

Output:
  (eigvals, eigvecs) torch tensors, dtype float64, sorted ascending by eigvals.

Library-style: no printing, no warnings (eig is a side-compute; errors raise).

TODO (Phase 3, only if profiling on large graphs shows need):
  - Shift-invert may explode in LU fill-in on N > 100k. Fall back to a tuned
    reflect or to torch.lobpcg on CUDA with an ILU/AMG preconditioner. Out of
    scope for v1.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from coarsen_mini.core.conversions import torch_csr_to_scipy_csr
from coarsen_mini.core.laplacian import LaplacianKind


def smallest_eigenpairs(
    L: torch.Tensor,
    k: int,
    laplacian_kind: LaplacianKind,
    delta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the k smallest eigenvalues / eigenvectors of L."""
    if L.layout != torch.sparse_csr:
        raise TypeError(f"L must be torch.sparse_csr, got layout {L.layout}")
    if k <= 0 or k >= L.shape[0]:
        raise ValueError(f"k must be in (0, N); got k={k}, N={L.shape[0]}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    L_sp = torch_csr_to_scipy_csr(L)
    if delta > 0:
        L_sp = L_sp + delta * sp.eye(L_sp.shape[0], format="csr")

    if laplacian_kind == "combinatorial":
        vals, vecs = _shift_invert(L_sp, k)
    elif laplacian_kind in ("normalized", "normalized_self_loop"):
        vals, vecs = _reflect(L_sp, k, omega=3.0)
    else:
        raise ValueError(f"unknown laplacian_kind {laplacian_kind!r}")

    return (
        torch.from_numpy(vals.astype(np.float64)),
        torch.from_numpy(vecs.astype(np.float64)),
    )


def _shift_invert(L: sp.csr_matrix, k: int):
    """eigsh with sigma slightly below 0 (avoids exact 0-eigenvalue singularity)."""
    vals, vecs = spla.eigsh(L, k=k, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _reflect(L: sp.csr_matrix, k: int, omega: float):
    """ARPACK on M = omega*I - L (matvec only); recover lambda_L = omega - mu."""
    M = omega * sp.eye(L.shape[0], format="csr") - L
    mu, vecs = spla.eigsh(M, k=k, which="LM")
    vals = omega - mu
    order = np.argsort(vals)
    return vals[order], vecs[:, order]
