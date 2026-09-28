"""Build graph-contextual training batches for alternating block diffusion.

For each sampled fine graph and shared timestep t, this module constructs:

* phase A: intra_t + inter_t, used to train intra updates with global degrees;
* phase B: intra_{t-1} + inter_t, used to train inter updates with global degrees.

The intra_{t-1} context is teacher-forced from the forward process. At inference
it is supplied by the just-completed learned intra reverse step. All allowed
variables are represented blockwise; no dense N x N graph is materialized.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import torch

from datasets.decoder_dataset import collate_oracle_regions, oracle_region_to_tensors
from datasets.decoder_regions import make_inter_region, make_intra_region


def _to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _collate(regions, device):
    items = [oracle_region_to_tensors(region) for region in regions]
    return _to_device(collate_oracle_regions(items), device)


def _trim(adjacency, n):
    return adjacency[:n, :n]


def _global_degree(
    record: Mapping,
    intra_regions,
    intra_states,
    inter_regions,
    inter_states,
    device,
):
    degree = torch.zeros(int(record["n_orig"]), dtype=torch.long, device=device)
    for region, state in zip(intra_regions, intra_states):
        nodes = torch.as_tensor(region.global_nodes, dtype=torch.long, device=device)
        local = _trim(state, region.n_nodes)
        degree.index_add_(0, nodes, local.sum(dim=1).long())
    for region, state in zip(inter_regions, inter_states):
        left_size = int((region.cluster_side == 0).sum())
        local = _trim(state, region.n_nodes)
        cross = local[:left_size, left_size:]
        left_nodes = torch.as_tensor(
            region.global_nodes[:left_size], dtype=torch.long, device=device
        )
        right_nodes = torch.as_tensor(
            region.global_nodes[left_size:], dtype=torch.long, device=device
        )
        degree.index_add_(0, left_nodes, cross.sum(dim=1).long())
        degree.index_add_(0, right_nodes, cross.sum(dim=0).long())
    return degree


def _sample_indices(rng, population, count):
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(population, size=count, replace=count > population).astype(np.int64)


def _sparse_context(
    record,
    intra_regions,
    intra_states,
    inter_regions,
    inter_states,
    degree,
    timestep,
    total_timesteps,
    phase,
    device,
):
    """Pack one full noisy graph using only allowed block candidate pairs."""
    n_orig = int(record["n_orig"])
    assignment = torch.as_tensor(record["assignment"], dtype=torch.long, device=device)
    cluster_size = torch.zeros(n_orig, dtype=torch.float, device=device)
    coarse_degree_by_cluster = torch.zeros(int(record["n_c"]), dtype=torch.float, device=device)
    for block in record["intra_blocks"]:
        nodes = torch.as_tensor(block["global_nodes"], dtype=torch.long, device=device)
        cluster_size[nodes] = float(len(block["global_nodes"]))
    for block in record["inter_blocks"]:
        coarse_degree_by_cluster[int(block["left_cluster"])] += 1
        coarse_degree_by_cluster[int(block["right_cluster"])] += 1
    phase_a = torch.full((n_orig,), float(phase == "intra"), device=device)
    phase_b = 1.0 - phase_a
    node_features = torch.stack((
        degree.float() / 16.0,
        cluster_size / 20.0,
        coarse_degree_by_cluster[assignment] / 16.0,
        torch.full((n_orig,), float(timestep) / total_timesteps, device=device),
        phase_a,
        phase_b,
    ), dim=-1)

    sources = []
    destinations = []
    features = []
    for region, state in zip(intra_regions, intra_states):
        nodes = torch.as_tensor(region.global_nodes, dtype=torch.long, device=device)
        pairs = torch.triu_indices(region.n_nodes, region.n_nodes, 1, device=device)
        u, v = nodes[pairs[0]], nodes[pairs[1]]
        values = _trim(state, region.n_nodes)[pairs[0], pairs[1]].float()
        sources.extend((u, v))
        destinations.extend((v, u))
        edge = torch.stack((values, torch.ones_like(values), torch.zeros_like(values)), dim=-1)
        features.extend((edge, edge))
    for region, state in zip(inter_regions, inter_states):
        left_size = int((region.cluster_side == 0).sum())
        nodes = torch.as_tensor(region.global_nodes, dtype=torch.long, device=device)
        left = nodes[:left_size]
        right = nodes[left_size:]
        u = left[:, None].expand(-1, right.numel()).reshape(-1)
        v = right[None, :].expand(left.numel(), -1).reshape(-1)
        values = _trim(state, region.n_nodes)[:left_size, left_size:].reshape(-1).float()
        sources.extend((u, v))
        destinations.extend((v, u))
        edge = torch.stack((values, torch.zeros_like(values), torch.ones_like(values)), dim=-1)
        features.extend((edge, edge))
    return {
        "node_features": node_features,
        "edge_index": torch.stack((torch.cat(sources), torch.cat(destinations))),
        "edge_features": torch.cat(features),
    }


@torch.no_grad()
def build_alternating_training_batch(
    records: Sequence[Mapping],
    graph_indices: Sequence[int],
    *,
    diffusion,
    regions_per_graph: int,
    intra_fraction: float,
    device: torch.device,
    rng: np.random.Generator,
    include_sparse_context: bool = False,
):
    """Return a model batch and coherent noisy payload for alternating training."""
    if regions_per_graph < 2:
        raise ValueError("regions_per_graph must be at least 2")
    n_intra_select = max(
        1, min(regions_per_graph - 1, round(regions_per_graph * intra_fraction))
    )
    n_inter_select = regions_per_graph - n_intra_select

    selected_regions = []
    selected_states = []
    selected_degrees = []
    selected_timesteps = []
    selected_context_indices = []
    context_nodes = []
    context_edges = []
    context_edge_features = []
    context_offset = 0

    for graph_index in graph_indices:
        record = records[int(graph_index)]
        intra_regions = [
            make_intra_region(record, block_index, graph_index=int(graph_index))
            for block_index, block in enumerate(record["intra_blocks"])
            if len(block["global_nodes"]) > 1
        ]
        inter_regions = [
            make_inter_region(record, block_index, graph_index=int(graph_index))
            for block_index in range(len(record["inter_blocks"]))
        ]
        if not intra_regions or not inter_regions:
            raise ValueError(f"graph {graph_index} lacks trainable intra/inter regions")

        intra_batch = _collate(intra_regions, device)
        inter_batch = _collate(inter_regions, device)
        timestep = int(rng.integers(1, diffusion.T + 1))
        intra_t = diffusion.q_sample(
            intra_batch,
            torch.full((len(intra_regions),), timestep, device=device, dtype=torch.long),
        )["adjacency_t"]
        inter_t = diffusion.q_sample(
            inter_batch,
            torch.full((len(inter_regions),), timestep, device=device, dtype=torch.long),
        )["adjacency_t"]

        # Teacher-forced post-intra context. At t=1, s=0 is exactly clean.
        if timestep == 1:
            intra_s = intra_batch["clean_adjacency"].clone()
        else:
            intra_s = diffusion.q_sample(
                intra_batch,
                torch.full(
                    (len(intra_regions),), timestep - 1, device=device, dtype=torch.long
                ),
            )["adjacency_t"]

        degree_phase_a = _global_degree(
            record, intra_regions, intra_t, inter_regions, inter_t, device
        )
        degree_phase_b = _global_degree(
            record, intra_regions, intra_s, inter_regions, inter_t, device
        )
        if include_sparse_context:
            phase_a_context = _sparse_context(
                record, intra_regions, intra_t, inter_regions, inter_t,
                degree_phase_a, timestep, diffusion.T, "intra", device,
            )
            phase_b_context = _sparse_context(
                record, intra_regions, intra_s, inter_regions, inter_t,
                degree_phase_b, timestep, diffusion.T, "inter", device,
            )
            phase_a_offset = context_offset
            for context in (phase_a_context, phase_b_context):
                context_nodes.append(context["node_features"])
                context_edges.append(context["edge_index"] + context_offset)
                context_edge_features.append(context["edge_features"])
                context_offset += context["node_features"].shape[0]
            phase_b_offset = phase_a_offset + int(record["n_orig"])
        intra_state_by_cluster = {
            region.cluster_ids[0]: _trim(state, region.n_nodes)
            for region, state in zip(intra_regions, intra_s)
        }
        # Singleton clusters have a fixed empty 1x1 intra state.
        for block in record["intra_blocks"]:
            cluster_id = int(block["cluster_id"])
            if cluster_id not in intra_state_by_cluster:
                intra_state_by_cluster[cluster_id] = torch.zeros(
                    (1, 1), dtype=torch.long, device=device
                )

        chosen_intra = _sample_indices(rng, len(intra_regions), n_intra_select)
        for index in chosen_intra:
            region = intra_regions[int(index)]
            selected_regions.append(region)
            selected_states.append(_trim(intra_t[int(index)], region.n_nodes))
            nodes = torch.as_tensor(region.global_nodes, dtype=torch.long, device=device)
            selected_degrees.append(degree_phase_a[nodes])
            selected_timesteps.append(timestep)
            if include_sparse_context:
                selected_context_indices.append(nodes + phase_a_offset)

        chosen_inter = _sample_indices(rng, len(inter_regions), n_inter_select)
        for index in chosen_inter:
            index = int(index)
            region = inter_regions[index]
            left_id, right_id = region.cluster_ids
            left_state = intra_state_by_cluster[left_id]
            right_state = intra_state_by_cluster[right_id]
            left_size = left_state.shape[0]
            context = region.clean_adjacency.copy()
            context[:left_size, :left_size] = left_state.cpu().numpy()
            context[left_size:, left_size:] = right_state.cpu().numpy()
            contextual_region = replace(region, clean_adjacency=context)

            current = torch.as_tensor(context, dtype=torch.long, device=device)
            inter_local = _trim(inter_t[index], region.n_nodes)
            current[:left_size, left_size:] = inter_local[:left_size, left_size:]
            current[left_size:, :left_size] = inter_local[left_size:, :left_size]
            selected_regions.append(contextual_region)
            selected_states.append(current)
            nodes = torch.as_tensor(region.global_nodes, dtype=torch.long, device=device)
            selected_degrees.append(degree_phase_b[nodes])
            selected_timesteps.append(timestep)
            if include_sparse_context:
                selected_context_indices.append(nodes + phase_b_offset)

    batch = _collate(selected_regions, device)
    adjacency_t = torch.zeros_like(batch["clean_adjacency"])
    current_degree = torch.zeros_like(batch["total_degree"])
    for item, (state, degree, region) in enumerate(
        zip(selected_states, selected_degrees, selected_regions)
    ):
        n = region.n_nodes
        adjacency_t[item, :n, :n] = state
        current_degree[item, :n] = degree
    batch["current_total_degree"] = current_degree
    if include_sparse_context:
        context_node_index = torch.full_like(batch["global_nodes"], -1)
        for item, indices in enumerate(selected_context_indices):
            context_node_index[item, :indices.numel()] = indices
        batch.update({
            "global_node_features": torch.cat(context_nodes),
            "global_edge_index": torch.cat(context_edges, dim=1),
            "global_edge_features": torch.cat(context_edge_features),
            "context_node_index": context_node_index,
        })
    t_int = torch.as_tensor(selected_timesteps, dtype=torch.long, device=device)
    noisy = {
        "adjacency_t": adjacency_t,
        "E_t": diffusion.adjacency_one_hot(adjacency_t, batch["node_mask"]),
        "t_int": t_int,
        "t": t_int.float() / diffusion.T,
    }
    if not torch.equal(adjacency_t, adjacency_t.transpose(1, 2)):
        raise AssertionError("contextual training state is asymmetric")
    if (current_degree < 0).any():
        raise AssertionError("contextual training produced a negative degree")
    return batch, noisy
