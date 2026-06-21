"""Shared coarsening logic for the Stage-A -> Stage-B bridge.

Used by both:
  - the unified `main.py` path (via CoarseGraphDataModule, which builds the coarse
    cache lazily on first run), and
  - the optional standalone `build_coarse_dataset.py` (prebuild on a CPU node).

Coarsens planar nx graphs into G_c records and encodes integer super-edge weights
into `de=11` Option-A buckets:  class 0 = no-edge (filled by DiGress's
encode_no_edge), classes 1..9 = exact weight, class 10 = weight >= 10.

A record (per graph) is a dict:
    {
      "coarse":        PyG Data (x=ones, edge_index, edge_attr[de=11], y, n_nodes),
      "assignment":    np.int64 (n_orig,)   original-node -> cluster id  (Stage C/D/E)
      "cluster_sizes": np.int64 (n_c,)       v_i                         (Stage C)
      "n_orig": int, "n_c": int, "r_actual": float,
    }
We coarsen ONCE per (r, K, laplacian_kind, method) and cache to disk so Stages
C/D/E never re-run Loukas.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from coarsen_mini.algorithms.loukas.main import coarsen_loukas
from coarsen_mini.core.builders import build_coarsening

DE = 11  # edge classes: 0=no-edge, 1..9 exact weight, 10 = ">=10" clamp


def nx_to_data(G: nx.Graph) -> Data:
    """Minimal unweighted PyG Data (x=ones) for coarsen_mini."""
    n = G.number_of_nodes()
    e = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([e, e.flip(0)], dim=1)            # undirected, both dirs
    return Data(x=torch.ones(n, 1), edge_index=edge_index, num_nodes=n)


def bucket_edge_attr(edge_weight: torch.Tensor) -> torch.Tensor:
    """Integer super-edge weights -> one-hot over DE classes.

    weight w (>=1 for any present edge):  class = clamp(round(w), 1, 10).
    Class 0 stays empty (reserved for no-edge; DiGress fills it via encode_no_edge).
    """
    wi = edge_weight.round().to(torch.long).clamp_(min=1, max=DE - 1)   # -> {1..10}
    return F.one_hot(wi, num_classes=DE).float()


def coarsen_one(G, r, laplacian_kind, method, K) -> dict | None:
    """Coarsen a single nx graph into a record. None if degenerate (no inter edges)."""
    data = nx_to_data(G)
    partition, info = coarsen_loukas(
        data, r=r, laplacian_kind=laplacian_kind, method=method, K=K,
        progress=False,
    )
    coars = build_coarsening(data, partition, method="loukas",
                             laplacian_kind=laplacian_kind, r_requested=r)
    ei = coars.coarsened.edge_index
    ew = coars.coarsened.edge_weight
    n_c = int(coars.n)
    if ei.numel() == 0:
        return None
    coarse = Data(
        x=torch.ones(n_c, 1, dtype=torch.float),
        edge_index=ei.to(torch.long).contiguous(),
        edge_attr=bucket_edge_attr(ew),
        y=torch.zeros([1, 0], dtype=torch.float),
        n_nodes=torch.tensor([n_c], dtype=torch.long),
    )
    return {
        "coarse": coarse,
        "assignment": partition.assignment.astype(np.int64),
        "cluster_sizes": partition.cluster_sizes.astype(np.int64),
        "n_orig": int(partition.N),
        "n_c": n_c,
        "r_actual": float(info["r_actual"]),
    }


def build_split(graphs, r, laplacian_kind, method, K, label="split"):
    """Coarsen a list of nx graphs into a list of records (with progress)."""
    records, n_skipped = [], 0
    t0 = time.perf_counter()
    for i, G in enumerate(graphs):
        rec = coarsen_one(G, r, laplacian_kind, method, K)
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        if (i + 1) % 250 == 0 or (i + 1) == len(graphs):
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            print(f"    [{label}] {i + 1}/{len(graphs)}  "
                  f"({rate:.1f} graphs/s, ETA {(len(graphs)-(i+1))/rate:5.0f}s)",
                  flush=True)
    if n_skipped:
        print(f"    [{label}] skipped {n_skipped} degenerate graphs (no inter edges)")
    return records


def edge_marginal(records) -> np.ndarray:
    """Empirical edge-class marginal over a split, incl. the no-edge class 0."""
    d = np.zeros(DE, dtype=np.float64)
    for rec in records:
        data, n = rec["coarse"], rec["n_c"]
        d[0] += n * (n - 1) - data.edge_index.shape[1]          # directed pairs
        d[1:] += data.edge_attr.sum(dim=0).numpy()[1:]
    return d / d.sum()


def cache_tag(r, K, laplacian_kind, method) -> str:
    """Cache subdir name encoding the coarsening params (so configs don't collide)."""
    return f"r{r}_{laplacian_kind}_{method}_K{K}"


def build_coarse_cache(planar_dir, cache_root, r, K, laplacian_kind, method,
                       force_rebuild=False, splits=("train", "val", "test")):
    """Ensure the coarse cache exists; build any missing split. Returns the subdir.

    Cache layout:  <cache_root>/<cache_tag>/{train,val,test}.pt
    Each split is (re)built from <planar_dir>/<split>.pkl only if its .pt is
    missing (or force_rebuild). Idempotent — safe to call every run.
    """
    subdir = os.path.join(cache_root, cache_tag(r, K, laplacian_kind, method))
    os.makedirs(subdir, exist_ok=True)
    for split in splits:
        out_path = os.path.join(subdir, f"{split}.pt")
        if os.path.exists(out_path) and not force_rebuild:
            print(f"  [coarsen] {split}: cache hit -> {out_path}")
            continue
        pkl = os.path.join(planar_dir, f"{split}.pkl")
        if not os.path.exists(pkl):
            print(f"  [coarsen] {split}: source {pkl} not found — skipping")
            continue
        with open(pkl, "rb") as fh:
            graphs = pickle.load(fh)
        print(f"  [coarsen] {split}: building from {len(graphs)} planar graphs "
              f"(r={r}, K={K}, {laplacian_kind}, {method}) ...", flush=True)
        records = build_split(graphs, r, laplacian_kind, method, K, split)
        torch.save(records, out_path)
        marg = edge_marginal(records)
        nc = np.array([rec["n_c"] for rec in records])
        print(f"  [coarsen] {split}: saved {len(records)} -> {out_path}  "
              f"(n_c mean={nc.mean():.1f} min={nc.min()} max={nc.max()})")
        print("    edge-class marginal (0=no-edge .. 10=>=10): "
              + " ".join(f"{c}:{p:.4f}" for c, p in enumerate(marg)))
    return subdir
