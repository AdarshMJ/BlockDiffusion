"""Lossless multiscale coordinate targets aligned with decoder blocks."""

from __future__ import annotations

from typing import Mapping

import numpy as np


class CoordinateBlockError(ValueError):
    """Raised when positions and a decoder record are misaligned."""


def build_coordinate_record(record: Mapping, positions: np.ndarray) -> dict:
    """Decompose points into cluster centroids and centered child offsets.

    The representation is permutation-aligned with global node ids. Adding the
    centroid selected by ``assignment`` to each offset reconstructs the source
    positions exactly (up to floating-point roundoff).
    """
    n_orig = int(record["n_orig"])
    n_c = int(record["n_c"])
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (n_orig, 2):
        raise CoordinateBlockError(
            f"positions must have shape ({n_orig}, 2), got {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise CoordinateBlockError("positions contain NaN or infinity")

    assignment = np.asarray(record["assignment"], dtype=np.int64).reshape(-1)
    if assignment.shape != (n_orig,):
        raise CoordinateBlockError("assignment length does not equal n_orig")
    if (assignment < 0).any() or (assignment >= n_c).any():
        raise CoordinateBlockError("assignment contains an invalid cluster id")

    blocks_by_id = {
        int(block["cluster_id"]): np.asarray(block["global_nodes"], dtype=np.int64)
        for block in record["intra_blocks"]
    }
    if set(blocks_by_id) != set(range(n_c)):
        raise CoordinateBlockError("intra blocks do not cover cluster ids 0..n_c-1")

    centroids = np.empty((n_c, 2), dtype=np.float64)
    offsets = np.empty_like(positions)
    cluster_sizes = np.empty(n_c, dtype=np.int64)
    rms_radius = np.empty(n_c, dtype=np.float64)
    max_radius = np.empty(n_c, dtype=np.float64)
    for cluster_id in range(n_c):
        nodes = blocks_by_id[cluster_id]
        if nodes.size == 0 or np.unique(nodes).size != nodes.size:
            raise CoordinateBlockError(f"cluster {cluster_id} is empty or duplicated")
        if not np.all(assignment[nodes] == cluster_id):
            raise CoordinateBlockError(f"cluster {cluster_id} disagrees with assignment")
        centroid = positions[nodes].mean(axis=0)
        local_offsets = positions[nodes] - centroid
        distances = np.linalg.norm(local_offsets, axis=1)
        centroids[cluster_id] = centroid
        offsets[nodes] = local_offsets
        cluster_sizes[cluster_id] = nodes.size
        rms_radius[cluster_id] = np.sqrt(np.mean(distances ** 2))
        max_radius[cluster_id] = distances.max()

    if int(cluster_sizes.sum()) != n_orig:
        raise CoordinateBlockError("cluster blocks do not partition all nodes")
    reconstructed = centroids[assignment] + offsets
    if not np.allclose(reconstructed, positions, rtol=0.0, atol=1e-12):
        raise AssertionError("centroid/offset decomposition is not lossless")
    centered_sum = np.zeros((n_c, 2), dtype=np.float64)
    np.add.at(centered_sum, assignment, offsets)
    if not np.allclose(centered_sum, 0.0, rtol=0.0, atol=1e-12):
        raise AssertionError("cluster offsets are not centered")

    coarse_edges = np.asarray([
        (int(block["left_cluster"]), int(block["right_cluster"]))
        for block in record["inter_blocks"]
    ], dtype=np.int64)
    if coarse_edges.size == 0:
        coarse_edges = np.empty((0, 2), dtype=np.int64)
    source_value = record.get("source_index")
    return {
        "n_orig": n_orig,
        "n_c": n_c,
        "source_index": None if source_value is None else int(source_value),
        "assignment": assignment.copy(),
        "cluster_centroids": centroids,
        "local_offsets": offsets,
        "cluster_sizes": cluster_sizes,
        "cluster_rms_radius": rms_radius,
        "cluster_max_radius": max_radius,
        "coarse_edges": coarse_edges,
    }


def summarize_coordinate_records(records) -> dict:
    """Return compact geometry statistics for a coordinate cache."""
    radii = np.concatenate([record["cluster_rms_radius"] for record in records])
    max_radii = np.concatenate([record["cluster_max_radius"] for record in records])
    sizes = np.concatenate([record["cluster_sizes"] for record in records])
    return {
        "num_graphs": len(records),
        "num_clusters": int(sizes.size),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_max": int(sizes.max()),
        "cluster_rms_radius_mean": float(radii.mean()),
        "cluster_rms_radius_max": float(radii.max()),
        "cluster_max_radius_mean": float(max_radii.mean()),
        "cluster_max_radius_max": float(max_radii.max()),
    }
