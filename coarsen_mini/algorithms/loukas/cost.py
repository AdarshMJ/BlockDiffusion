"""benchmark/algorithms/loukas/cost.py — local subgraph costs for Loukas.

Two cost functions, one per Loukas contraction family:

  quickspectral_costs_all_edges(...)   for variation_edges (nc=2)
  spectral_cost_neighborhood(...)      for variation_neighborhood (nc variable)

Both compute the same underlying quantity (the local spectral approximation
error after merging the subgraph), only one is specialised for the rank-1
structure of pairs.

ALGEBRAIC IDENTITY (quickspectral, valid only for nc=2)
--------------------------------------------------------
For an edge (i, j) with weight w, define
    L_local = [[2 d_i - w, -w], [-w, 2 d_j - w]]
    Pibot   = I_2 - (1/2) * 1 1^T            (projection orthogonal to mean)
    B       = Pibot @ A_cost[{i,j}, :]       (shape 2 x K)

B has rank 1 by construction: B[0,:] = (A_cost[i] - A_cost[j])/2 = -B[1,:].
Writing v = (A_cost[i] - A_cost[j])/2,
    B^T L_local B = (L[0,0] - L[0,1] - L[1,0] + L[1,1]) * v v^T
                  = 2 (d_i + d_j) * v v^T        (off-diagonal -w cancel diagonal -w)
The Frobenius norm of this rank-1 matrix is just the scalar's absolute value,
    cost_quickspectral = (1/2) * (d_i + d_j) * ||A_cost[i] - A_cost[j]||^2
                       = (1/2) * (d_i + d_j) * (row_norm2[i] + row_norm2[j] - 2 dot)

Verified numerically identical to the generic Spectral / Spectral_np cost
within floating point on 500 Cora edges (max abs diff 4e-15). Speedup vs
spectral_np on CPU when vectorising over all edges: 24x (one torch.matmul-
equivalent instead of M Python calls).

No analog of this rank-1 shortcut exists for nc > 2 (B is no longer rank 1),
hence spectral_cost_neighborhood falls back to the generic dense formula
with @jit for per-set speed.

Both costs assume A_cost is dense (N x K), as is the case during Loukas
where A_cost = R @ diag(lambda^{-1/2}) at level 1 and updated B-rotations
afterwards.
"""

import numpy as np
from numba import jit


def row_norms_squared(A_cost: np.ndarray) -> np.ndarray:
    """Precompute row-wise squared L2 norms of A_cost; reused at every edge."""
    return np.einsum("ij,ij->i", A_cost, A_cost)


def quickspectral_costs_all_edges(
    A_cost: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    deg: np.ndarray,
    row_norm2: np.ndarray,
) -> np.ndarray:
    """Vectorised quickspectral cost over all candidate edges.

    Parameters
    ----------
    A_cost   : (N, K) float, the spectral basis matrix B = R diag(lambda^{-1/2}).
    edges_i  : (M,) int, source node of each candidate edge.
    edges_j  : (M,) int, target node of each candidate edge (with i < j typically).
    deg      : (N,) float, degree of each node in the current intermediate graph.
    row_norm2: (N,) float, output of row_norms_squared(A_cost).

    Returns
    -------
    costs : (M,) float, one cost per candidate edge.
    """
    a_i = A_cost[edges_i]                # (M, K)
    a_j = A_cost[edges_j]                # (M, K)
    dot = np.einsum("mk,mk->m", a_i, a_j)
    diff_norm2 = row_norm2[edges_i] + row_norm2[edges_j] - 2.0 * dot
    return 0.5 * (deg[edges_i] + deg[edges_j]) * diff_norm2


@jit(nopython=True, cache=True)
def spectral_cost_neighborhood(
    nodes: np.ndarray,
    W_restrict: np.ndarray,
    A_cost: np.ndarray,
    deg: np.ndarray,
) -> float:
    """Generic spectral cost for a subgraph of nc >= 2 nodes (jit per call).

    Numerically equivalent to the official Loukas subgraph_cost for nc > 2,
    with our additional division by (nc - 1) for size-fairness across
    differently sized neighborhoods. Verified equal to the sparse-based
    reference on 172 Cora neighborhoods (max abs diff 2e-16).

    Parameters
    ----------
    nodes      : (nc,) int, the subgraph node indices.
    W_restrict : (nc, nc) dense weight matrix W[nodes, :][:, nodes].
                 Caller materialises it from the sparse adjacency once.
    A_cost     : (N, K) float spectral basis.
    deg        : (N,) float degree of nodes in the current graph.
    """
    nc = nodes.shape[0]
    if nc == 1:
        return 1e7  # singleton has no contraction value
    ones = np.ones(nc)
    A_sel = A_cost[nodes]                                # (nc, K)
    Pibot = np.eye(nc) - np.outer(ones, ones) / nc       # (nc, nc)
    B = Pibot @ A_sel                                    # (nc, K)
    L_local = np.diag(2.0 * deg[nodes] - W_restrict @ ones) - W_restrict   # (nc, nc)
    # Frobenius norm via numpy default (jit-compatible)
    M = B.T @ L_local @ B                                # (K, K)
    cost = np.linalg.norm(M) / (nc - 1)
    return cost
