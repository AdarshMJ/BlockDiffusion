"""benchmark/core/partition.py — Partition dataclass and Q helpers shared by all algos.

A Partition is the canonical output of every coarsening algorithm in the
benchmark. It carries the cluster assignment (one cluster id per original
node) and knows how to materialise the binary lifting matrix Q_combi as a
torch sparse CSR tensor.

Plus two shared helpers used by Loukas now and by every algo wrapper later
(METIS, Leiden, FGC, UGC, ConvMatch, etc.):

  pinv_well_partitioned(Q)       : closed-form scipy pinv. Used by Loukas internals.
  pinv_well_partitioned_torch(Q) : same formula, native torch sparse CSR.
                                    Used by builders to compute Coarsening.P.
  partition_from_Q(Q)            : extract a Partition from a binary lifting Q.
                                    Accepts scipy CSR or torch sparse CSR.
  sanity_check_partition_Q(p, Q) : verify (QP)^2=QP, QPQ=Q, cluster stats. Opt-in.

Conventions:
  - assignment : np.ndarray of shape (N,), dtype int64. Values in [0, n_clusters).
    Cluster ids are CONTIGUOUS: every value in [0, n_clusters) is used at
    least once. This is the algo's responsibility; Partition.__post_init__
    enforces it.
  - n_clusters : int, the number of supernodes in the coarsened graph.

Q_combi convention (the matrix actually used in math, decided Phase 0):
  - shape (N, n_clusters), torch.sparse_csr_tensor, dtype float64
  - Q[i, c] = 1 if node i is assigned to cluster c, else 0
  - Exactly one 1 per row (well-partitioned). Hence Q^T Q is the diagonal
    matrix of cluster sizes, and P = pinv(Q) = (Q^T Q)^{-1} Q^T has rows
    P[c, :] with value 1/|cluster c| at columns of cluster members.

Library-style usage:
  - Raises ValueError on invalid construction (non-contiguous ids, wrong shape).
  - No printing, no warnings.
  - cluster_sizes is computed on demand (cheap, O(N)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class Partition:
    """Cluster assignment + count, the canonical output of every coarsening algo."""

    assignment: np.ndarray
    n_clusters: int
    _cluster_sizes_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.assignment, np.ndarray):
            self.assignment = np.asarray(self.assignment)
        if self.assignment.ndim != 1:
            raise ValueError(f"assignment must be 1D, got shape {self.assignment.shape}")
        if self.assignment.dtype.kind not in ("i", "u"):
            self.assignment = self.assignment.astype(np.int64, copy=False)
        if self.assignment.dtype != np.int64:
            self.assignment = self.assignment.astype(np.int64, copy=False)
        if int(self.assignment.min()) < 0:
            raise ValueError(f"assignment has negative ids: min={int(self.assignment.min())}")
        if int(self.assignment.max()) >= self.n_clusters:
            raise ValueError(
                f"assignment max={int(self.assignment.max())} >= n_clusters={self.n_clusters}"
            )
        # contiguous: every cluster id in [0, n_clusters) must be used at least once
        used = np.unique(self.assignment)
        if used.shape[0] != self.n_clusters:
            raise ValueError(
                f"assignment has {used.shape[0]} distinct ids but n_clusters={self.n_clusters}; "
                "ids must be contiguous (relabel before constructing Partition)"
            )

    @property
    def N(self) -> int:
        """Number of original nodes."""
        return int(self.assignment.shape[0])

    @property
    def cluster_sizes(self) -> np.ndarray:
        """Number of original nodes per cluster, shape (n_clusters,). Cached."""
        if self._cluster_sizes_cache is None:
            self._cluster_sizes_cache = np.bincount(self.assignment, minlength=self.n_clusters)
        return self._cluster_sizes_cache

    def to_Q_combi(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Binary lifting matrix as torch sparse CSR, shape (N, n_clusters), one 1 per row."""
        N, n = self.N, self.n_clusters
        crow = torch.arange(N + 1, dtype=torch.int64)  # one nonzero per row
        col = torch.from_numpy(self.assignment.astype(np.int64, copy=False))
        vals = torch.ones(N, dtype=dtype)
        return torch.sparse_csr_tensor(crow, col, vals, size=(N, n))


def relabel_contiguous(raw_assignment: np.ndarray) -> tuple[np.ndarray, int]:
    """Remap arbitrary integer ids in `raw_assignment` to a contiguous 0..n-1 range.

    Useful when an algorithm produces sparse cluster ids (e.g. after Loukas merges).
    Returns (new_assignment, n_clusters). Use to build a Partition cleanly:

        new_a, n = relabel_contiguous(raw_a)
        partition = Partition(new_a, n)
    """
    raw_assignment = np.asarray(raw_assignment)
    used, inverse = np.unique(raw_assignment, return_inverse=True)
    return inverse.astype(np.int64), int(used.shape[0])


def pinv_well_partitioned(Q: sp.csr_matrix) -> sp.csr_matrix:
    """Closed-form pseudo-inverse of a lifting Q with one nonzero per row.

    Hypothesis: Q has exactly one nonzero per row (well-partitioned form, as
    produced by every coarsening algorithm in the benchmark and by their
    intermediate levels). Then Q^T Q is diagonal with entries
    `s_c = sum_{i in cluster c} Q[i, c]^2`, and

        pinv(Q) = (Q^T Q)^{-1} Q^T = diag(1/s_c) @ Q^T.

    For binary Q (Q_combi), s_c = |cluster c|, so pinv(Q)[c, i] = 1/|cluster c|
    on the cluster members (the standard cluster-mean projector).

    Returns a scipy CSR matrix. Used internally by Loukas at every level for
    P_loukas / P_combi chains, and by core/builders.py (Phase 1.A) to build
    the final Coarsening.P from Partition.to_Q_combi().
    """
    Q = Q.tocsr()
    QtQ_diag = np.asarray(Q.multiply(Q).sum(axis=0)).ravel()
    QtQ_diag = np.where(QtQ_diag > 0, QtQ_diag, 1.0)
    return (sp.diags(1.0 / QtQ_diag) @ Q.T).tocsr()


def pinv_well_partitioned_torch(Q: torch.Tensor) -> torch.Tensor:
    """Native torch sparse CSR pseudo-inverse for a Q with one nonzero per row.

    Same closed-form as the scipy version `pinv_well_partitioned`: Q^T Q is
    diagonal (one nonzero per row in Q means each column of Q^T has at most one
    nonzero per cluster, but summed per cluster gives the diag), so
        diag(Q^T Q)[c] = sum_{i: assignment[i]==c} Q[i, c]**2
        pinv(Q)        = diag(1 / diag(Q^T Q)) @ Q^T

    Implementation: pure CSR API via csr_from_triplets (no torch.sparse_coo).
    """
    if Q.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr, got {Q.layout}")
    N, n_cols = Q.shape
    vals = Q.values()
    col_idx = Q.col_indices()
    crow = Q.crow_indices()
    device, dtype = vals.device, vals.dtype

    # diag(Q^T Q): bucket sum of squared values, one bucket per column of Q
    diag = torch.zeros(n_cols, dtype=dtype, device=device)
    diag.scatter_add_(0, col_idx, vals * vals)
    diag_safe = torch.where(diag > 0, diag, torch.ones_like(diag))

    # P = diag(1/diag) @ Q^T : same nonzeros as Q^T (= swap rows/cols of Q)
    # with rescaled values
    new_vals = vals / diag_safe[col_idx]
    row_idx_Q = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64), crow[1:] - crow[:-1]
    )
    # P (n_cols x N): rows = col_idx of Q (= cluster ids), cols = original row ids
    from coarsen_mini.core.conversions import csr_from_triplets
    return csr_from_triplets(col_idx.to(torch.int64), row_idx_Q, new_vals, n_cols, N)


def partition_from_Q(Q) -> "Partition":
    """Extract a Partition from a binary lifting Q (one nonzero per row).

    Accepts scipy CSR or torch sparse CSR. Auto-relabels cluster ids to be
    contiguous in case Q has empty columns (rare but possible in some algos).
    """
    if isinstance(Q, torch.Tensor):
        if Q.layout != torch.sparse_csr:
            raise TypeError(f"expected torch.sparse_csr or scipy CSR, got {Q.layout}")
        if Q.crow_indices().shape[0] - 1 != Q.shape[0]:
            raise RuntimeError("Q does not look like a row-stochastic lifting")
        col_idx = Q.col_indices().detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        Q_sp = Q.tocsr()
        if Q_sp.indptr.shape[0] - 1 != Q_sp.shape[0]:
            raise RuntimeError("Q does not look like a row-stochastic lifting")
        col_idx = Q_sp.indices.copy().astype(np.int64)
    assignment, n_clusters = relabel_contiguous(col_idx)
    return Partition(assignment, n_clusters)


def sanity_check_partition_Q(partition: "Partition", Q: sp.csr_matrix) -> dict:
    """Optional consistency checks for any (Partition, Q) pair. Toggled by callers
    via their `validate=True` flag.

    Checks
    ------
    (QP)^2 = QP  : the cluster-mean projector is idempotent
    Q P Q = Q    : P is a generalised inverse of Q (E2/E3 condition)
    cluster size statistics (min/max/mean, n_singletons)

    Cost
    ----
    O(nnz(Q) * cluster_size) per matmul. Safe on small/medium graphs,
    measurable on large (use sparingly).
    """
    sizes = np.asarray(Q.multiply(Q).sum(axis=0)).ravel()
    P_mp = (sp.diags(1.0 / np.where(sizes > 0, sizes, 1.0)) @ Q.T).tocsr()
    QP = (Q @ P_mp).tocsr()
    QPQP = QP @ QP
    diff_idemp = (QPQP - QP).tocsr()
    err_idemp = float(np.abs(diff_idemp.data).max()) if diff_idemp.nnz > 0 else 0.0
    QPQ = Q @ P_mp @ Q
    diff_pinv = (QPQ - Q).tocsr()
    err_pinv = float(np.abs(diff_pinv.data).max()) if diff_pinv.nnz > 0 else 0.0
    return {
        "qp_idempotent_err": err_idemp,
        "pinv_err": err_pinv,
        "cluster_sizes_min": int(partition.cluster_sizes.min()),
        "cluster_sizes_max": int(partition.cluster_sizes.max()),
        "cluster_sizes_mean": float(partition.cluster_sizes.mean()),
        "n_singletons": int((partition.cluster_sizes == 1).sum()),
    }
