"""Build decoder records from generated coarse topology, sizes, and weights."""

from __future__ import annotations

import numpy as np


def generated_blueprint_record(coarse_adjacency, cluster_sizes, edge_weights):
    adjacency = np.asarray(coarse_adjacency, dtype=bool)
    sizes = np.asarray(cluster_sizes, dtype=np.int64).reshape(-1)
    if adjacency.shape != (len(sizes), len(sizes)):
        raise ValueError("coarse adjacency and cluster sizes disagree")
    if not np.array_equal(adjacency, adjacency.T) or np.diag(adjacency).any():
        raise ValueError("coarse adjacency must be simple undirected")
    if (sizes < 1).any():
        raise ValueError("cluster sizes must be positive")
    left, right = np.nonzero(np.triu(adjacency, 1))
    weights = np.asarray(edge_weights, dtype=np.int64).reshape(-1)
    if len(weights) != len(left):
        raise ValueError("one weight is required per undirected coarse edge")
    starts = np.concatenate(([0], np.cumsum(sizes)))
    intra = []
    for cluster_id, size in enumerate(sizes):
        nodes = np.arange(starts[cluster_id], starts[cluster_id + 1], dtype=np.int64)
        intra.append({"cluster_id": cluster_id, "global_nodes": nodes,
                      "edges": np.empty((0, 2), dtype=np.int64),
                      "connected": size == 1, "e_i": 0})
    inter = []
    for u, v, weight in zip(left, right, weights):
        capacity = int(sizes[u] * sizes[v])
        if not 1 <= int(weight) <= capacity:
            raise ValueError(f"weight {weight} outside capacity {capacity}")
        inter.append({"left_cluster": int(u), "right_cluster": int(v),
                      "edges": np.empty((0, 2), dtype=np.int64), "w_ij": 0,
                      "sampling_w_ij": int(weight)})
    n_orig = int(sizes.sum())
    return {"n_orig": n_orig, "n_c": len(sizes), "source_index": None,
            "assignment": np.repeat(np.arange(len(sizes)), sizes),
            "cluster_sizes": sizes, "original_edges": np.empty((0, 2), dtype=np.int64),
            "node_degrees": np.zeros(n_orig, dtype=np.int64),
            "intra_blocks": intra, "inter_blocks": inter}
