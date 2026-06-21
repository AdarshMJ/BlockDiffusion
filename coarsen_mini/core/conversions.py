"""benchmark/core/conversions.py — sparse format conversions and CSR triplet helpers.

Two purposes:

1) Convert between formats at module boundaries:
     PyG Data    <->  scipy.sparse.csr_matrix  <->  torch.sparse_csr_tensor
   (scipy is used only inside Loukas internals and inside core/eig.py where
   scipy.sparse.linalg.eigsh has no torch equivalent.)

2) Manipulate torch.sparse_csr_tensor in pure CSR API (no torch.sparse_coo
   anywhere): the two helpers `csr_from_triplets` and `extract_triplets` let
   us implement masking, scaling, and identity injection on torch CSR
   tensors without ever materialising a sparse_coo intermediate.

All helpers are device-aware (output on the device of the inputs).
"""

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Cross-format conversions (scipy <-> torch <-> PyG)
# ---------------------------------------------------------------------------

def data_to_scipy_csr_adjacency(data) -> sp.csr_matrix:
    """Build scipy CSR adjacency from a PyG Data. Used by Loukas internals and eig.

    Assumptions (enforced by datasets/loaders, not re-checked here):
      - undirected, no self-loops, single LCC.
    """
    edge_index = data.edge_index.detach().cpu().numpy()
    num_nodes = int(data.num_nodes)
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is None:
        w = np.ones(edge_index.shape[1], dtype=np.float64)
    else:
        w = edge_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    return sp.csr_matrix(
        (w, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )


def scipy_csr_to_torch_csr(
    M: sp.csr_matrix, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Convert scipy CSR to torch.sparse_csr_tensor. Copies the three arrays."""
    M = M.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)),
        torch.from_numpy(M.indices.astype(np.int64)),
        torch.from_numpy(M.data.astype(np.float64)).to(dtype),
        size=M.shape,
    )


def torch_csr_to_scipy_csr(T: torch.Tensor) -> sp.csr_matrix:
    """Convert torch.sparse_csr_tensor to scipy CSR. Used by eig only."""
    if T.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr layout, got {T.layout}")
    return sp.csr_matrix(
        (
            T.values().detach().cpu().numpy(),
            T.col_indices().detach().cpu().numpy(),
            T.crow_indices().detach().cpu().numpy(),
        ),
        shape=tuple(T.shape),
    )


# ---------------------------------------------------------------------------
# Pure-CSR triplet helpers (NO torch.sparse_coo anywhere)
# ---------------------------------------------------------------------------

def csr_from_triplets(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    N_rows: int,
    N_cols: int,
) -> torch.Tensor:
    """Build a torch.sparse_csr_tensor from unsorted (rows, cols, vals) triplets.

    Steps (all torch, GPU-friendly):
      1. Sort by (row, col) via argsort on `row * N_cols + col` (int64).
      2. Count entries per row via bincount(sorted_rows, minlength=N_rows).
      3. crow = cumsum prepended with 0.
      4. Construct torch.sparse_csr_tensor.

    Assumes NO duplicate (row, col) entries. Caller responsibility: if you
    expect duplicates (e.g. multi-edge after coarsening), coalesce upstream
    (sum same-(row, col) entries before calling).

    Used everywhere we need to apply a mask, drop diagonal, scale entries, or
    inject new diagonal entries on a torch sparse CSR. Avoids the
    torch.sparse_coo intermediate.
    """
    device = rows.device
    if rows.shape != cols.shape or rows.shape != vals.shape:
        raise ValueError(
            f"shape mismatch: rows {rows.shape}, cols {cols.shape}, vals {vals.shape}"
        )
    rows_long = rows.to(torch.int64)
    cols_long = cols.to(torch.int64)
    order = torch.argsort(rows_long * N_cols + cols_long)
    s_rows = rows_long[order]
    s_cols = cols_long[order]
    s_vals = vals[order]
    counts = torch.bincount(s_rows, minlength=N_rows)
    crow = torch.zeros(N_rows + 1, dtype=torch.int64, device=device)
    crow[1:] = torch.cumsum(counts, dim=0)
    return torch.sparse_csr_tensor(crow, s_cols, s_vals, size=(N_rows, N_cols))


def extract_triplets(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get (row, col, val) tensors from a torch.sparse_csr_tensor.

    Reconstructs row indices from crow_indices via repeat_interleave. Pure CSR
    API, no COO materialisation.
    """
    if M.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr layout, got {M.layout}")
    crow = M.crow_indices()
    col = M.col_indices()
    vals = M.values()
    counts = crow[1:] - crow[:-1]
    row = torch.repeat_interleave(
        torch.arange(M.shape[0], dtype=torch.int64, device=M.device), counts
    )
    return row, col, vals


# ---------------------------------------------------------------------------
# PyG -> torch CSR direct (pure torch, no scipy)
# ---------------------------------------------------------------------------

def data_to_torch_csr_adjacency(data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a torch.sparse_csr_tensor adjacency directly from PyG Data.

    Default dtype float32 to match PyG / GCNConv convention. Override to float64
    for metric computations that need precision.
    """
    ei = data.edge_index
    device = ei.device
    N = int(data.num_nodes)
    ew = getattr(data, "edge_weight", None)
    if ew is None:
        vals = torch.ones(ei.shape[1], dtype=dtype, device=device)
    else:
        vals = ew.to(dtype).to(device)
    return csr_from_triplets(ei[0], ei[1], vals, N, N)
