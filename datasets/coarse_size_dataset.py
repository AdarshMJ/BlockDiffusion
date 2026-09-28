"""Sparse graph packing for Stage-C cluster-size prediction."""

from __future__ import annotations

import numpy as np
import torch


def collate_coarse_size_graphs(records, record_indices, *, max_size, device):
    targets, node_graph, edges = [], [], []
    graph_target_n, graph_n_coarse, graph_max_degree = [], [], []
    offset = 0
    for graph_id, record_index in enumerate(record_indices):
        record = records[int(record_index)]
        sizes = np.asarray(record["cluster_sizes"], dtype=np.int64)
        if int(sizes.max()) > max_size:
            raise ValueError(f"cluster size {int(sizes.max())} exceeds max_size={max_size}")
        local_edges = np.asarray([
            (int(block["left_cluster"]), int(block["right_cluster"]))
            for block in record["inter_blocks"]
        ], dtype=np.int64).reshape(-1, 2)
        degree = np.bincount(local_edges.reshape(-1), minlength=len(sizes))
        targets.extend((sizes - 1).tolist())
        node_graph.extend([graph_id] * len(sizes))
        edges.extend((local_edges + offset).tolist())
        graph_target_n.append(int(record["n_orig"]))
        graph_n_coarse.append(len(sizes))
        graph_max_degree.append(max(1, int(degree.max())))
        offset += len(sizes)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    directed = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return {
        "coarse_degree": torch.bincount(directed[1], minlength=offset),
        "node_graph_index": torch.tensor(node_graph, dtype=torch.long, device=device),
        "graph_target_n": torch.tensor(graph_target_n, dtype=torch.long, device=device),
        "graph_n_coarse": torch.tensor(graph_n_coarse, dtype=torch.long, device=device),
        "graph_max_degree": torch.tensor(graph_max_degree, dtype=torch.long, device=device),
        "edge_index": edge_index,
        "directed_edge_index": directed,
        "target": torch.tensor(targets, dtype=torch.long, device=device),
    }


def pack_generated_coarse_graphs(adjacencies, target_sizes, *, device):
    """Pack class-id/bool adjacency matrices without requiring oracle sizes."""
    records = []
    for adjacency, target_n in zip(adjacencies, target_sizes):
        array = np.asarray(adjacency, dtype=bool)
        left, right = np.nonzero(np.triu(array, 1))
        records.append({
            "n_orig": int(target_n),
            "cluster_sizes": np.ones(array.shape[0], dtype=np.int64),
            "inter_blocks": [
                {"left_cluster": int(u), "right_cluster": int(v)}
                for u, v in zip(left, right)
            ],
        })
    return collate_coarse_size_graphs(
        records, range(len(records)), max_size=1, device=device
    )
