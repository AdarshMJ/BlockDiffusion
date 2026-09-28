"""PyTorch dataset and collator for oracle intra/inter decoder regions.

This is the thin ML-runtime adapter around :mod:`datasets.decoder_regions`.
Region construction stays numpy-only and fully unit-tested there.  Samples are
created on demand; the ~1.9M local regions are *not* duplicated into a second
flattened cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.decoder_regions import (
    OracleRegion,
    make_inter_region,
    make_intra_region,
    node_degrees_from_record,
)


INTRA = 0
INTER = 1


def compute_edge_marginals_by_kind(
    records: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Compute exact intra/inter marginals without constructing a region index."""
    counts = np.zeros((2, 2), dtype=np.int64)
    for record in records:
        for block in record["intra_blocks"]:
            n = len(block["global_nodes"])
            if n <= 1:
                continue
            variables = n * (n - 1) // 2
            positives = int(block["e_i"])
            counts[INTRA] += (variables - positives, positives)
        intra = record["intra_blocks"]
        for block in record["inter_blocks"]:
            left = intra[int(block["left_cluster"])]
            right = intra[int(block["right_cluster"])]
            variables = len(left["global_nodes"]) * len(right["global_nodes"])
            positives = int(block["w_ij"])
            counts[INTER] += (variables - positives, positives)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("cannot compute a marginal for an empty region kind")
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return torch.from_numpy(probabilities).float()


def oracle_region_to_tensors(region: OracleRegion) -> dict[str, torch.Tensor]:
    """Convert one semantic region to the tensors consumed by the model."""
    kind = INTRA if region.kind == "intra" else INTER
    cluster_ids = torch.full((2,), -1, dtype=torch.long)
    cluster_ids[:len(region.cluster_ids)] = torch.as_tensor(region.cluster_ids)
    source_index = -1 if region.source_index is None else int(region.source_index)
    return {
        "kind": torch.tensor(kind, dtype=torch.long),
        "graph_index": torch.tensor(region.graph_index, dtype=torch.long),
        "source_index": torch.tensor(source_index, dtype=torch.long),
        "cluster_ids": cluster_ids,
        "global_nodes": torch.from_numpy(region.global_nodes.copy()).long(),
        "cluster_side": torch.from_numpy(region.cluster_side.copy()).long(),
        "clean_adjacency": torch.from_numpy(region.clean_adjacency.copy()).long(),
        "update_mask": torch.from_numpy(region.update_mask.copy()).bool(),
        "total_degree": torch.from_numpy(region.total_degree.copy()).long(),
        "edge_budget": torch.tensor(region.edge_budget, dtype=torch.long),
    }


def load_decoder_records(path: str | Path) -> list[dict]:
    """Load a trusted decoder cache across torch versions."""
    try:
        records = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        records = torch.load(path, map_location="cpu")
    if not isinstance(records, list):
        raise TypeError(f"decoder cache must contain a list, got {type(records).__name__}")
    return records


class OracleRegionDataset(Dataset):
    """Index intra regions, inter regions, or both without duplicating payloads.

    Singleton intra blocks contain no edge variable and are excluded by default.
    They remain part of size-predictor data, but cannot contribute to an edge
    denoising loss.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        region_kind: str = "both",
        include_singleton_intra: bool = False,
    ):
        if region_kind not in {"intra", "inter", "both"}:
            raise ValueError("region_kind must be 'intra', 'inter', or 'both'")
        self.records = list(records)
        self.region_kind = region_kind
        self.include_singleton_intra = include_singleton_intra

        # Existing v1 caches predate node_degrees. Compute them once per graph,
        # not once per local region, and keep them only in memory.
        for record in self.records:
            if "node_degrees" not in record:
                record["node_degrees"] = node_degrees_from_record(record)

        selections: list[tuple[list[int], list[int]]] = []
        n_examples = 0
        for record in self.records:
            intra_indices = []
            inter_indices = []
            if region_kind in {"intra", "both"}:
                intra_indices = [
                    i for i, block in enumerate(record["intra_blocks"])
                    if include_singleton_intra or len(block["global_nodes"]) > 1
                ]
            if region_kind in {"inter", "both"}:
                inter_indices = list(range(len(record["inter_blocks"])))
            selections.append((intra_indices, inter_indices))
            n_examples += len(intra_indices) + len(inter_indices)

        # Compact arrays avoid millions of Python tuples in the index.
        self.graph_indices = np.empty(n_examples, dtype=np.int32)
        self.region_kinds = np.empty(n_examples, dtype=np.uint8)
        self.block_indices = np.empty(n_examples, dtype=np.int32)
        cursor = 0
        for graph_index, (intra_indices, inter_indices) in enumerate(selections):
            for kind, indices in ((INTRA, intra_indices), (INTER, inter_indices)):
                end = cursor + len(indices)
                self.graph_indices[cursor:end] = graph_index
                self.region_kinds[cursor:end] = kind
                self.block_indices[cursor:end] = indices
                cursor = end
        assert cursor == n_examples

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> "OracleRegionDataset":
        return cls(load_decoder_records(path), **kwargs)

    def __len__(self) -> int:
        return int(self.graph_indices.size)

    def edge_marginals_by_kind(self) -> torch.Tensor:
        """Exact no-edge/edge marginals over selected trainable variables."""
        counts = np.zeros((2, 2), dtype=np.int64)
        for record in self.records:
            if self.region_kind in {"intra", "both"}:
                for block in record["intra_blocks"]:
                    n = len(block["global_nodes"])
                    if not self.include_singleton_intra and n <= 1:
                        continue
                    variables = n * (n - 1) // 2
                    positives = int(block["e_i"])
                    counts[INTRA] += (variables - positives, positives)
            if self.region_kind in {"inter", "both"}:
                intra = record["intra_blocks"]
                for block in record["inter_blocks"]:
                    left = intra[int(block["left_cluster"])]
                    right = intra[int(block["right_cluster"])]
                    variables = len(left["global_nodes"]) * len(right["global_nodes"])
                    positives = int(block["w_ij"])
                    counts[INTER] += (variables - positives, positives)
        if (counts.sum(axis=1) == 0).any():
            raise ValueError("cannot compute a marginal for an empty region kind")
        probabilities = counts / counts.sum(axis=1, keepdims=True)
        return torch.from_numpy(probabilities).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        graph_index = int(self.graph_indices[index])
        kind = int(self.region_kinds[index])
        block_index = int(self.block_indices[index])
        record = self.records[graph_index]
        if kind == INTRA:
            region = make_intra_region(record, block_index, graph_index=graph_index)
        else:
            region = make_inter_region(record, block_index, graph_index=graph_index)

        return oracle_region_to_tensors(region)


def collate_oracle_regions(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable local regions into a dense mini-batch.

    Dense local tensors are intentional: observed regions have at most 17 nodes
    for intra blocks and at most roughly twice that for inter blocks. Global
    sparsity comes from instantiating only coarse-edge regions.
    """
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch_size = len(samples)
    sizes = torch.tensor([sample["global_nodes"].numel() for sample in samples], dtype=torch.long)
    max_nodes = int(sizes.max())

    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long)
    update_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    global_nodes = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    cluster_side = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    total_degree = torch.zeros((batch_size, max_nodes), dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        n = int(sizes[batch_index])
        adjacency[batch_index, :n, :n] = sample["clean_adjacency"]
        update_mask[batch_index, :n, :n] = sample["update_mask"]
        node_mask[batch_index, :n] = True
        global_nodes[batch_index, :n] = sample["global_nodes"]
        cluster_side[batch_index, :n] = sample["cluster_side"]
        total_degree[batch_index, :n] = sample["total_degree"]

    # These are model-contract assertions, not merely test conveniences.
    if not torch.equal(adjacency, adjacency.transpose(1, 2)):
        raise ValueError("clean adjacency must be symmetric")
    if not torch.equal(update_mask, update_mask.transpose(1, 2)):
        raise ValueError("update mask must be symmetric")
    if torch.diagonal(update_mask, dim1=1, dim2=2).any():
        raise ValueError("self-edges cannot be denoising variables")

    stack_keys = ("kind", "graph_index", "source_index", "cluster_ids", "edge_budget")
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in stack_keys}
    batch.update({
        "n_nodes": sizes,
        "node_mask": node_mask,
        "global_nodes": global_nodes,
        "cluster_side": cluster_side,
        "clean_adjacency": adjacency,
        "update_mask": update_mask,
        "total_degree": total_degree,
    })
    return batch
