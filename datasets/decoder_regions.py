"""Create oracle-conditioned local denoising regions from decoder block caches.

The representation makes the intended factorization explicit:

* An intra region contains one cluster. Every off-diagonal pair is denoisable.
* An inter region contains ``B_i`` followed by ``B_j``. Intra edges are visible
  context, while only the cross-cluster rectangle is denoisable.

Arrays are numpy-only so region semantics can be tested without an ML runtime.
The future torch Dataset/collator should be a mechanical conversion of this
representation rather than reimplementing its graph logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class RegionConstructionError(ValueError):
    """Raised when a serialized decoder record is internally inconsistent."""


@dataclass(frozen=True)
class OracleRegion:
    """One local denoising problem with oracle coarse/block conditioning."""

    kind: str
    graph_index: int
    source_index: int | None
    cluster_ids: tuple[int, ...]
    global_nodes: np.ndarray
    cluster_side: np.ndarray
    clean_adjacency: np.ndarray
    update_mask: np.ndarray
    total_degree: np.ndarray
    edge_budget: int

    @property
    def n_nodes(self) -> int:
        return int(self.global_nodes.size)

    @property
    def n_edge_variables(self) -> int:
        """Number of undirected variables updated by this region."""
        return int(np.triu(self.update_mask, k=1).sum())


def _edges(value: Any, label: str) -> np.ndarray:
    edges = np.asarray(value, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise RegionConstructionError(f"{label} edges must have shape (m, 2)")
    return edges


def _add_undirected_edges(adjacency: np.ndarray, edges: np.ndarray, offset: int = 0) -> None:
    n = adjacency.shape[0]
    for u_raw, v_raw in edges:
        u, v = int(u_raw) + offset, int(v_raw) + offset
        if u == v or not (0 <= u < n and 0 <= v < n):
            raise RegionConstructionError(f"invalid local edge ({u}, {v}) for region size {n}")
        adjacency[u, v] = 1
        adjacency[v, u] = 1


def node_degrees_from_record(record: Mapping[str, Any]) -> np.ndarray:
    """Return total fine-graph degrees, supporting both v1 and newer caches."""
    n_orig = int(record["n_orig"])
    if "node_degrees" in record:
        degree = np.asarray(record["node_degrees"], dtype=np.int64).reshape(-1)
        if degree.size != n_orig:
            raise RegionConstructionError("node_degrees length does not equal n_orig")
        return degree

    if "original_edges" not in record:
        raise RegionConstructionError("record needs node_degrees or original_edges")
    degree = np.zeros(n_orig, dtype=np.int64)
    for u_raw, v_raw in _edges(record["original_edges"], "original"):
        u, v = int(u_raw), int(v_raw)
        if not (0 <= u < n_orig and 0 <= v < n_orig):
            raise RegionConstructionError("original edge endpoint outside n_orig")
        degree[u] += 1
        degree[v] += 1
    return degree


def _intra_by_id(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result = {int(block["cluster_id"]): block for block in record["intra_blocks"]}
    if len(result) != int(record["n_c"]):
        raise RegionConstructionError("intra blocks do not uniquely cover all clusters")
    return result


def make_intra_region(
    record: Mapping[str, Any],
    block_index: int,
    *,
    graph_index: int = -1,
) -> OracleRegion:
    """Create an ``A_ii`` denoising example."""
    block = record["intra_blocks"][block_index]
    cluster_id = int(block["cluster_id"])
    global_nodes = np.asarray(block["global_nodes"], dtype=np.int64).reshape(-1)
    n = int(global_nodes.size)
    if n < 1:
        raise RegionConstructionError("intra block cannot be empty")

    adjacency = np.zeros((n, n), dtype=np.int8)
    _add_undirected_edges(adjacency, _edges(block["edges"], "intra"))
    update_mask = ~np.eye(n, dtype=bool)
    edge_budget = int(block.get("e_i", np.triu(adjacency, k=1).sum()))
    if edge_budget != int(np.triu(adjacency, k=1).sum()):
        raise RegionConstructionError(f"intra edge budget mismatch for cluster {cluster_id}")

    degree = node_degrees_from_record(record)
    return OracleRegion(
        kind="intra",
        graph_index=graph_index,
        source_index=record.get("source_index"),
        cluster_ids=(cluster_id,),
        global_nodes=global_nodes.copy(),
        cluster_side=np.zeros(n, dtype=np.int8),
        clean_adjacency=adjacency,
        update_mask=update_mask,
        total_degree=degree[global_nodes].copy(),
        edge_budget=edge_budget,
    )


def make_inter_region(
    record: Mapping[str, Any],
    block_index: int,
    *,
    graph_index: int = -1,
    edge_budget_override: int | None = None,
) -> OracleRegion:
    """Create an ``A_ij`` region with frozen intra-edge context.

    The cached ``w_ij`` is always validated against the oracle edge list.
    Generation may separately override the sampling cardinality; this keeps a
    predicted blueprint from masquerading as a valid training-data record.
    """
    block = record["inter_blocks"][block_index]
    left_id = int(block["left_cluster"])
    right_id = int(block["right_cluster"])
    if left_id >= right_id:
        raise RegionConstructionError("inter cluster ids must be canonical (left < right)")

    intra_blocks = record["intra_blocks"]
    # Current caches store blocks in cluster-id order. Keep a validated fallback
    # for hand-built/legacy records that do not.
    if (
        left_id < len(intra_blocks)
        and right_id < len(intra_blocks)
        and int(intra_blocks[left_id]["cluster_id"]) == left_id
        and int(intra_blocks[right_id]["cluster_id"]) == right_id
    ):
        left, right = intra_blocks[left_id], intra_blocks[right_id]
    else:
        intra = _intra_by_id(record)
        if left_id not in intra or right_id not in intra:
            raise RegionConstructionError("inter block references a missing intra block")
        left, right = intra[left_id], intra[right_id]
    left_nodes = np.asarray(left["global_nodes"], dtype=np.int64).reshape(-1)
    right_nodes = np.asarray(right["global_nodes"], dtype=np.int64).reshape(-1)
    n_left, n_right = int(left_nodes.size), int(right_nodes.size)
    n = n_left + n_right

    # Older v1 caches duplicated these mappings in each inter block. Validate
    # them when present, while allowing the compact format to omit them.
    if "left_global_nodes" in block and not np.array_equal(
        left_nodes, np.asarray(block["left_global_nodes"], dtype=np.int64)
    ):
        raise RegionConstructionError("left inter/intra node mapping mismatch")
    if "right_global_nodes" in block and not np.array_equal(
        right_nodes, np.asarray(block["right_global_nodes"], dtype=np.int64)
    ):
        raise RegionConstructionError("right inter/intra node mapping mismatch")

    adjacency = np.zeros((n, n), dtype=np.int8)
    _add_undirected_edges(adjacency, _edges(left["edges"], "left intra"))
    _add_undirected_edges(adjacency, _edges(right["edges"], "right intra"), offset=n_left)

    cross_edges = _edges(block["edges"], "inter")
    for u_raw, v_raw in cross_edges:
        u, v = int(u_raw), int(v_raw)
        if not (0 <= u < n_left and 0 <= v < n_right):
            raise RegionConstructionError(
                f"cross edge ({u}, {v}) outside bipartite shape ({n_left}, {n_right})"
            )
        adjacency[u, n_left + v] = 1
        adjacency[n_left + v, u] = 1

    update_mask = np.zeros((n, n), dtype=bool)
    update_mask[:n_left, n_left:] = True
    update_mask[n_left:, :n_left] = True
    oracle_edge_budget = int(block.get("w_ij", cross_edges.shape[0]))
    if oracle_edge_budget != int(cross_edges.shape[0]):
        raise RegionConstructionError(f"inter edge budget mismatch for ({left_id}, {right_id})")
    edge_budget = oracle_edge_budget if edge_budget_override is None else int(edge_budget_override)
    if not 1 <= edge_budget <= n_left * n_right:
        raise RegionConstructionError(
            f"inter sampling budget {edge_budget} outside [1, {n_left * n_right}] "
            f"for ({left_id}, {right_id})"
        )

    global_nodes = np.concatenate((left_nodes, right_nodes))
    degree = node_degrees_from_record(record)
    return OracleRegion(
        kind="inter",
        graph_index=graph_index,
        source_index=record.get("source_index"),
        cluster_ids=(left_id, right_id),
        global_nodes=global_nodes,
        cluster_side=np.concatenate((
            np.zeros(n_left, dtype=np.int8),
            np.ones(n_right, dtype=np.int8),
        )),
        clean_adjacency=adjacency,
        update_mask=update_mask,
        total_degree=degree[global_nodes].copy(),
        edge_budget=edge_budget,
    )
