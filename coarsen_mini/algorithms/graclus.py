"""benchmark/algorithms/graclus.py — multilevel Graclus heavy-edge matching.

Wraps `torch_geometric.nn.pool.graclus`. Mostly torch end-to-end: the
underlying graclus call takes torch `edge_index` + `edge_weight`, and we
maintain the multilevel coarsened adjacency as a torch sparse CSR (reusing
benchmark.core.q.coarsened_adjacency).

Per-level loop:
  1. extract edge_index + edge_weight from current_A (torch CSR)
  2. graclus matching -> per-node cluster id (length n_cur)
  3. relabel contiguous, compose into cumulative assignment
  4. coarsen current_A via Q_binary^T A Q_binary (the BINARY intermediate Q,
     not Q_normalized -- A_c = Q_b^T A Q_b is the standard coarsened
     adjacency, with Q_normalized it would be a different operator)

Stop when n_cur <= n_target or no further reduction (n_next == n_cur).
Bounded by max_levels (default 50, plenty for any reasonable graph).

Dependency: `torch-cluster` (provides the kernel used by PyG's graclus).
Usually installed on OAR / compute-cluster Python envs. On a fresh local
env, install with `pip install torch-cluster` (may require matching CUDA +
PyTorch versions; can be tricky -- see torch_cluster repo).

Note: graclus from PyG is deterministic given a fixed edge_index ordering.
The `seed` parameter is accepted for API uniformity but ignored.

API:
  coarsen_graclus(data, r, *, seed=14, max_levels=50) -> (Partition, info)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from torch_geometric.nn.pool import graclus

from coarsen_mini.algorithms.common import resolve_target
from coarsen_mini.core.conversions import (
    data_to_torch_csr_adjacency,
    extract_triplets,
)
from coarsen_mini.core.partition import Partition, relabel_contiguous
from coarsen_mini.core.q import coarsened_adjacency, degrees_torch


def coarsen_graclus(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
    max_levels: int = 50,
) -> tuple[Partition, dict]:
    """Multilevel Graclus coarsening (heavy-edge matching).

    Specify exactly one of `r` (ratio) or `n_clusters` (absolute count).

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph. Assumed connected (no isolated nodes); loader enforces this.
    r : float in (0, 1), optional
    n_clusters : int in [1, N], optional
    seed : int, default 14
        Accepted for API uniformity; graclus is deterministic so seed is unused.
    max_levels : int, default 50

    Returns
    -------
    partition : Partition
    info : dict (r_requested, r_actual, n_clusters, n_target, n_levels, total_time_s, seed)
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    current_A = data_to_torch_csr_adjacency(data)            # (N, N) sparse CSR
    # cumulative_assignment[i] = current cluster id of original node i
    cumulative_assignment = torch.arange(N, dtype=torch.int64, device=current_A.device)

    n_cur = N
    n_levels = 0

    while n_cur > n_target and n_levels < max_levels:
        # graclus needs edge_index + edge_weight (torch); pull them from current_A
        row, col, weight = extract_triplets(current_A)
        edge_index = torch.stack([row, col])

        # Heavy-edge matching: returns cluster tensor of length n_cur (non-contiguous ids)
        cluster_torch = graclus(edge_index, weight=weight, num_nodes=n_cur)

        # Relabel to contiguous [0, n_next)
        unique_ids, new_ids = torch.unique(cluster_torch, return_inverse=True)
        n_next = int(unique_ids.shape[0])

        if n_next == n_cur:
            # Graclus could not reduce further (fixed point); stop here.
            break

        # Compose cumulative assignment
        cumulative_assignment = new_ids[cumulative_assignment]

        # Coarsen current_A via Q^T A Q with the intermediate binary Q
        Q_intermediate = Partition(new_ids.cpu().numpy(), n_next).to_Q_combi()
        current_A = coarsened_adjacency(
            current_A, Q_intermediate, delete_diagonal=True, symmetrize=True
        )

        n_cur = n_next
        n_levels += 1

    assignment = cumulative_assignment.cpu().numpy()
    assignment, n_clusters = relabel_contiguous(assignment)
    partition = Partition(assignment, n_clusters)

    info = {
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "n_levels": n_levels,
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
    }
    return partition, info
