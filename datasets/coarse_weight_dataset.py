"""On-demand graph batches for coarse-edge multiplicity prediction."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


def collate_coarse_weight_graphs(records: Sequence[Mapping], record_indices,
                                 *, max_weight: int, device) -> dict[str, torch.Tensor]:
    """Pack complete sparse coarse graphs; targets are exact unbucketed w_ij."""
    sizes, node_graph, edges, targets, capacities = [], [], [], [], []
    graph_n_orig, graph_max_degree = [], []
    offset = 0
    for packed_index, record_index in enumerate(record_indices):
        record = records[int(record_index)]
        local_sizes = np.asarray(record["cluster_sizes"], dtype=np.int64)
        local_edges = np.asarray([
            (int(block["left_cluster"]), int(block["right_cluster"]))
            for block in record["inter_blocks"]
        ], dtype=np.int64).reshape(-1, 2)
        local_targets = np.asarray(
            [int(block["w_ij"]) for block in record["inter_blocks"]], dtype=np.int64
        )
        if local_targets.size and int(local_targets.max()) > max_weight:
            raise ValueError(
                f"w_ij={int(local_targets.max())} exceeds max_weight={max_weight}; "
                "increase model.max_weight rather than silently clipping"
            )
        local_degree = np.bincount(local_edges.reshape(-1), minlength=len(local_sizes))
        sizes.extend(local_sizes.tolist())
        node_graph.extend([packed_index] * len(local_sizes))
        edges.extend((local_edges + offset).tolist())
        targets.extend((local_targets - 1).tolist())
        capacities.extend((local_sizes[local_edges[:, 0]] * local_sizes[local_edges[:, 1]]).tolist())
        graph_n_orig.append(int(record["n_orig"]))
        graph_max_degree.append(max(1, int(local_degree.max())))
        offset += len(local_sizes)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    directed = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    coarse_degree = torch.bincount(directed[1], minlength=offset)
    return {
        "cluster_sizes": torch.tensor(sizes, dtype=torch.long, device=device),
        "coarse_degree": coarse_degree,
        "node_graph_index": torch.tensor(node_graph, dtype=torch.long, device=device),
        "graph_n_orig": torch.tensor(graph_n_orig, dtype=torch.long, device=device),
        "graph_max_degree": torch.tensor(graph_max_degree, dtype=torch.long, device=device),
        "edge_index": edge_index,
        "directed_edge_index": directed,
        "target": torch.tensor(targets, dtype=torch.long, device=device),
        "capacity": torch.tensor(capacities, dtype=torch.long, device=device),
        "record_indices": torch.tensor(record_indices, dtype=torch.long, device=device),
    }
