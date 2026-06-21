"""benchmark/algorithms/loukas/main.py — short rewrite of Loukas multilevel coarsening.

Mirrors the production entry point
`coarsen_variation_scipy_normalized_laplacian_with_intermediary`
(coarsening_algorithms/intermidiary_loukas/utils_coarsening.py L1089) with the
simplifications validated in Phase 0.5 (see README.md and DESIGN.md).

Key choices:
  - `method` in {"edges", "neighborhood"}
  - cost : `quickspectral` vectorised numpy (edges), `spectral_cost_neighborhood`
           jit (neighborhood)
  - matching : disjoint by default, absorption fallback on stall (edges only)
  - intermediate B update : `loukas_combi` mode hardcoded
  - laplacian_kind : combinatorial / normalized / normalized_self_loop
  - eig kernel : hybrid (shift-invert combinatoire, reflect omega=3 normalised)

See README.md for the stall problem on large graphs and the absorption mode.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from coarsen_mini.algorithms.common import resolve_target
from coarsen_mini.core.conversions import (
    data_to_scipy_csr_adjacency,
    scipy_csr_to_torch_csr,
)
from coarsen_mini.core.eig import smallest_eigenpairs
from coarsen_mini.core.laplacian import (
    LaplacianKind,
    build_laplacian,
    degrees,
    normalize_laplacian_sp,
    shifted_degrees,
)
from coarsen_mini.core.partition import (
    Partition,
    partition_from_Q,
    pinv_well_partitioned,
    relabel_contiguous,
    sanity_check_partition_Q,
)
from coarsen_mini.algorithms.loukas.cost import (
    quickspectral_costs_all_edges,
    row_norms_squared,
    spectral_cost_neighborhood,
)
from coarsen_mini.algorithms.loukas.matching import (
    greedy_match_edges,
    greedy_match_neighborhood,
)

log = logging.getLogger(__name__)


def coarsen_loukas(
    data,
    r: float | None = None,
    n_clusters: int | None = None,
    *,
    method: Literal["edges", "neighborhood"] = "edges",
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    K: int = 100,
    n_e_percent: float = 0.1,
    n_e_adaptive: bool = True,
    max_levels: int = 200,
    max_level_r: float = 0.99,
    delta: float = 0.0,
    progress: bool = True,
    validate: bool = False,
    # Absorption fallback (edges only)
    absorption_on_stall: bool = True,
    stall_threshold: float = 0.5,  # empirical default; see algorithms/loukas/README.md "Empirical defaults"
    absorption_cost_factor: float = 2.0,
    absorption_max_cluster_size: int = 10,
) -> tuple[Partition, dict]:
    """Multilevel variation coarsening (edges or neighborhood family).

    See README.md for the design rationale, especially the absorption mode
    that handles the disjoint-matching stall on large/high-r graphs.

    Parameters
    ----------
    n_e_percent : float
        Per-level cap on the fraction of nodes contracted.
    n_e_adaptive : bool
        If True (default, matches legacy + Loukas 2019 theory): per-level cap
        is `n_e_percent * n_cur` (rescales as the graph shrinks → each level
        contracts ~n_e_percent of the current size → constant aggressivity).
        If False: per-level cap is `n_e_percent * N_initial` (fixed across
        levels → aggressivity grows as n_cur shrinks). Adaptive preserves
        spectral quality better at high r since later levels stay gentle.

    Returns
    -------
    partition : Partition
    info : dict with diagnostics; always contains r_requested, r_actual,
           n_clusters, n_levels, K, method, laplacian_kind, level_times_s,
           eig_time_s, total_time_s, stalled, R (eigenvectors), levels (per-level
           matching info), P_loukas (torch.sparse_csr_tensor)
    """
    t_total = time.perf_counter()

    if method not in ("edges", "neighborhood"):
        raise ValueError(f"method must be edges or neighborhood, got {method!r}")

    A_orig = data_to_scipy_csr_adjacency(data)
    N = A_orig.shape[0]
    n_target, r_req = resolve_target(N, r, n_clusters)
    r = min(r_req, 0.9999)        # legacy clamping for max-r tolerance
    # If not adaptive, precompute once on N_initial; else recompute per level inside the loop
    n_e_fixed = max(1, int(n_e_percent * N)) if not n_e_adaptive else None

    # Level 1: eig of L (full original graph)
    t_eig = time.perf_counter()
    L_torch = build_laplacian(data, laplacian_kind)
    vals_t, vecs_t = smallest_eigenpairs(L_torch, K, laplacian_kind, delta=delta)
    eig_time_s = time.perf_counter() - t_eig

    vals = vals_t.numpy().astype(np.float64)
    R = vecs_t.numpy().astype(np.float64)
    # threshold 1e-6: blocks the trivial constant eigvec (lambda ≈ 1e-9 from
    # ARPACK noise, measured 1-3e-9 on Cora/CiteSeer/PubMed/chameleon LCC, up
    # to 1.2e-8 on random_geometric N=2000) while leaving all useful eigvals
    # untouched (smallest non-trivial lambda_1 measured 1.26e-3 on CiteSeer
    # LCC, marge x1000). 1e-6 keeps extra x10 safety margin vs measured noise
    # for larger graphs (e.g. ogbn-arxiv where ARPACK noise may grow).
    # Legacy used 1e-5 (overly aggressive — could mask lambda_1 on poorly-
    # connected graphs); our initial 1e-10 leaked the constant eigvec with
    # lambda^{-1/2} ~ 1e4 amplification, distorting edge costs.
    mask = vals < 1e-6
    vals_safe = np.where(mask, 1.0, vals)
    lsinv = np.where(mask, 0.0, vals_safe ** -0.5)
    B = R * lsinv[None, :]               # (N, K)
    A_cost = B

    A_inter = A_orig.tocsr().copy()
    n_cur = N
    coarsening_chain: list[sp.csr_matrix] = []
    P_loukas_chain: list[sp.csr_matrix] = []
    level_times: list[float] = []
    level_infos: list[dict] = []

    iterator = tqdm(range(1, max_levels + 1), disable=not progress,
                    desc=f"Loukas[{method}]")
    for level in iterator:
        if n_cur <= n_target:
            break
        t_level = time.perf_counter()

        deg = degrees(A_inter)
        D_N = shifted_degrees(deg, laplacian_kind)

        r_cur = min(max_level_r, 1.0 - n_target / n_cur)
        n_e_level = max(1, int(n_e_percent * n_cur)) if n_e_adaptive else n_e_fixed
        n_target_removed = max(1, min(n_e_level, int(np.ceil(r_cur * n_cur))))

        if method == "edges":
            edges_i, edges_j = _upper_triangular_edges(A_inter)
            if edges_i.shape[0] == 0:
                log.debug(f"level {level}: no edges left, stopping")
                break
            row_norm2 = row_norms_squared(A_cost)
            costs = quickspectral_costs_all_edges(
                A_cost, edges_i, edges_j, deg, row_norm2
            )
            coarsening_list, match_info = greedy_match_edges(
                costs, edges_i, edges_j, n_target_removed=n_target_removed,
                absorption_on_stall=absorption_on_stall,
                stall_threshold=stall_threshold,
                absorption_cost_factor=absorption_cost_factor,
                absorption_max_cluster_size=absorption_max_cluster_size,
                A_inter=A_inter,
                A_cost=A_cost,
                deg=deg,
            )
        else:  # neighborhood
            neighborhoods = _build_neighborhoods(A_inter)
            if not neighborhoods:
                log.debug(f"level {level}: no neighborhoods, stopping")
                break
            costs = np.empty(len(neighborhoods), dtype=np.float64)
            for k, nbh in enumerate(neighborhoods):
                W_sub = np.asarray(A_inter[nbh, :][:, nbh].toarray())
                costs[k] = spectral_cost_neighborhood(nbh, W_sub, A_cost, deg)
            coarsening_list, match_info = greedy_match_neighborhood(
                costs, neighborhoods, n_target_removed=n_target_removed,
            )

        if not coarsening_list:
            log.debug(f"level {level}: no admissible contraction, stopping")
            break

        intermediate_Q_combi = _coarsening_list_to_Q(coarsening_list, n_cur)
        n_next = intermediate_Q_combi.shape[1]

        # A_c = Q^T A Q, then drop diagonal (delete_diagonal hardcoded True),
        # then symmetrise to avoid complex eigenvalues downstream.
        A_next = (intermediate_Q_combi.T @ A_inter @ intermediate_Q_combi).tolil()
        A_next.setdiag(0)
        A_next = A_next.tocsr()
        A_next.eliminate_zeros()
        A_next = ((A_next + A_next.T) * 0.5).tocsr()

        deg_next = degrees(A_next)
        D_n = shifted_degrees(deg_next, laplacian_kind)

        # Degree-aware intermediate lifting + its pinv, for P_loukas chain
        intermediate_Q_norm = (
            sp.diags(np.sqrt(D_N))
            @ intermediate_Q_combi
            @ sp.diags(np.where(D_n > 0, D_n ** -0.5, 0.0))
        )
        intermediate_P_norm = pinv_well_partitioned(intermediate_Q_norm)
        intermediate_P_combi = pinv_well_partitioned(intermediate_Q_combi)

        # B update : loukas_combi mode (pinv of binary intermediate Q)
        B = np.asarray(intermediate_P_combi @ B)

        # Recompute A_cost via eigh of B^T L_norm B (level > 1 trick)
        if n_next > 1:
            L_norm_next = normalize_laplacian_sp(A_next, deg_next, laplacian_kind)
            product = np.asarray((B.T @ L_norm_next @ B))
            d, V = np.linalg.eigh(product)
            mask2 = d < 1e-6  # same threshold rationale as level-1 mask above
            d_safe = np.where(mask2, 1.0, d)
            dinvsqrt = np.where(mask2, 0.0, d_safe ** -0.5)
            A_cost = (B @ V) * dinvsqrt[None, :] @ V.T
        else:
            A_cost = B

        coarsening_chain.append(intermediate_Q_combi)
        P_loukas_chain.append(intermediate_P_norm)

        A_inter = A_next
        n_cur = n_next
        level_times.append(time.perf_counter() - t_level)
        match_info["n_target_removed"] = n_target_removed
        match_info["n_cur_after"] = n_cur
        level_infos.append(match_info)
        log.debug(
            f"level {level}: n -> {n_cur}, accepted_disjoint={match_info['n_disjoint_accepted']}, "
            f"accepted_absorption={match_info['n_absorption_accepted']}, dt={level_times[-1]:.2f}s"
        )

    # Compose chains
    if coarsening_chain:
        Q_lifting_combi = coarsening_chain[0]
        for IQ in coarsening_chain[1:]:
            Q_lifting_combi = Q_lifting_combi @ IQ
        P_loukas = P_loukas_chain[0]
        for IP in P_loukas_chain[1:]:
            P_loukas = IP @ P_loukas
    else:
        Q_lifting_combi = sp.eye(N, format="csr")
        P_loukas = sp.eye(N, format="csr")

    partition = partition_from_Q(Q_lifting_combi)
    n_clusters = partition.n_clusters
    r_actual = 1.0 - n_clusters / N
    stalled = (n_clusters > int(np.ceil((1.0 - r) * N)) * 1.05)

    info: dict = {
        "r_requested": r_req,
        "r_actual": r_actual,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "n_levels": len(level_times),
        "K": K,
        "method": method,
        "laplacian_kind": laplacian_kind,
        "level_times_s": level_times,
        "eig_time_s": eig_time_s,
        "total_time_s": time.perf_counter() - t_total,
        "stalled": stalled,
        "levels": level_infos,
        "R": R,
        "P_loukas": scipy_csr_to_torch_csr(P_loukas.tocsr()),
    }

    if stalled:
        warnings.warn(
            f"Loukas stalled: r_requested={r:.3f}, r_actual={r_actual:.3f}, "
            f"n_clusters={n_clusters} (target ~{int(np.ceil((1.0 - r) * N))})",
            stacklevel=2,
        )

    if validate:
        info["sanity"] = sanity_check_partition_Q(partition, Q_lifting_combi)

    return partition, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _upper_triangular_edges(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges (i < j) as two int64 arrays."""
    coo = sp.triu(A, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


def _build_neighborhoods(A: sp.csr_matrix) -> list[np.ndarray]:
    """Per node, the closed neighborhood [i, neighbors(i)] as a sorted int64 array."""
    N = A.shape[0]
    A = A.tocsr()
    out: list[np.ndarray] = []
    for i in range(N):
        s, e = A.indptr[i], A.indptr[i + 1]
        nbrs = A.indices[s:e]
        if nbrs.shape[0] == 0:
            continue
        nodes = np.unique(np.concatenate([[i], nbrs])).astype(np.int64)
        if nodes.shape[0] >= 2:
            out.append(nodes)
    return out


def _coarsening_list_to_Q(
    coarsening_list: list[np.ndarray], n_cur: int
) -> sp.csr_matrix:
    """Build intermediate binary lifting (n_cur, n_next): one cluster per accepted
    group + one singleton cluster per survivor (node not in any group)."""
    cluster_id = -np.ones(n_cur, dtype=np.int64)
    next_id = 0
    for group in coarsening_list:
        cluster_id[group] = next_id
        next_id += 1
    survivors = np.where(cluster_id < 0)[0]
    for u in survivors:
        cluster_id[u] = next_id
        next_id += 1
    rows = np.arange(n_cur)
    cols = cluster_id
    data = np.ones(n_cur, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cur, next_id))




