"""Extract exact intra/inter decoder blocks from a Loukas coarsening.

This module is deliberately independent of torch, PyG, and NetworkX at import
time.  It can therefore be unit-tested on a CPU-only machine, while accepting
the objects used by the training pipeline through a small duck-typed boundary.

Edges are stored once, as undirected local endpoint pairs:

* ``IntraBlock.edges[k] = (u_local, v_local)`` with ``u_local < v_local``.
* ``InterBlock.edges[k] = (u_in_left, v_in_right)``.

The exact inter-block edge count is the decoder budget ``w_ij``.  It is computed
from the original graph rather than recovered from the bucketed coarse edge
class, which loses information when ``w_ij >= 10``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class BlockExtractionError(ValueError):
    """Raised when a fine graph and coarse record are inconsistent."""


def _empty_edges() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def _edge_array(edges: Iterable[tuple[int, int]]) -> np.ndarray:
    values = list(edges)
    if not values:
        return _empty_edges()
    return np.asarray(values, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class IntraBlock:
    """The fine subgraph induced by one coarse node."""

    cluster_id: int
    global_nodes: np.ndarray
    edges: np.ndarray
    connected: bool

    @property
    def n_nodes(self) -> int:
        return int(self.global_nodes.size)

    @property
    def edge_count(self) -> int:
        """Exact intra-cluster budget ``e_i``."""
        return int(self.edges.shape[0])


@dataclass(frozen=True)
class InterBlock:
    """The bipartite fine edges represented by one coarse edge."""

    left_cluster: int
    right_cluster: int
    left_global_nodes: np.ndarray
    right_global_nodes: np.ndarray
    edges: np.ndarray

    @property
    def edge_count(self) -> int:
        """Exact, unbucketed super-edge budget ``w_ij``."""
        return int(self.edges.shape[0])


@dataclass(frozen=True)
class DecoderBlocks:
    """A lossless partition of a fine graph's edge variables."""

    n_orig: int
    n_c: int
    assignment: np.ndarray
    cluster_sizes: np.ndarray
    intra_blocks: tuple[IntraBlock, ...]
    inter_blocks: tuple[InterBlock, ...]
    original_edges: np.ndarray
    source_index: int | None = None

    @property
    def n_edges(self) -> int:
        return int(self.original_edges.shape[0])

    @property
    def intra_edge_count(self) -> int:
        return sum(block.edge_count for block in self.intra_blocks)

    @property
    def inter_edge_count(self) -> int:
        return sum(block.edge_count for block in self.inter_blocks)

    def reconstructed_edges(self) -> np.ndarray:
        """Assemble the canonical global edge list from all decoder blocks."""
        assembled: list[tuple[int, int]] = []

        for block in self.intra_blocks:
            for u_local, v_local in block.edges:
                u = int(block.global_nodes[u_local])
                v = int(block.global_nodes[v_local])
                assembled.append((min(u, v), max(u, v)))

        for block in self.inter_blocks:
            for u_local, v_local in block.edges:
                u = int(block.left_global_nodes[u_local])
                v = int(block.right_global_nodes[v_local])
                assembled.append((min(u, v), max(u, v)))

        return _edge_array(sorted(assembled))

    def to_record(self) -> dict[str, Any]:
        """Return a plain, stable cache record (no custom classes required)."""
        node_degrees = np.zeros(self.n_orig, dtype=np.int64)
        for u, v in self.original_edges:
            node_degrees[int(u)] += 1
            node_degrees[int(v)] += 1
        return {
            "n_orig": self.n_orig,
            "n_c": self.n_c,
            "source_index": self.source_index,
            "assignment": self.assignment.copy(),
            "cluster_sizes": self.cluster_sizes.copy(),
            "original_edges": self.original_edges.copy(),
            "node_degrees": node_degrees,
            "intra_blocks": [
                {
                    "cluster_id": block.cluster_id,
                    "global_nodes": block.global_nodes.copy(),
                    "edges": block.edges.copy(),
                    "connected": block.connected,
                    "e_i": block.edge_count,
                }
                for block in self.intra_blocks
            ],
            "inter_blocks": [
                {
                    "left_cluster": block.left_cluster,
                    "right_cluster": block.right_cluster,
                    "left_global_nodes": block.left_global_nodes.copy(),
                    "right_global_nodes": block.right_global_nodes.copy(),
                    "edges": block.edges.copy(),
                    "w_ij": block.edge_count,
                }
                for block in self.inter_blocks
            ],
        }


def _as_numpy(value: Any) -> np.ndarray:
    """Convert numpy/torch-like values without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _canonical_edges(original_graph: Any, n_orig: int) -> np.ndarray:
    """Read a simple undirected edge set from NetworkX/PyG/an iterable."""
    if hasattr(original_graph, "edge_index"):
        raw = _as_numpy(original_graph.edge_index)
        if raw.ndim != 2 or raw.shape[0] != 2:
            raise BlockExtractionError("fine edge_index must have shape (2, m)")
        edge_iter = raw.T
    elif hasattr(original_graph, "edges"):
        edges_attr = original_graph.edges
        edge_iter = edges_attr() if callable(edges_attr) else edges_attr
    else:
        edge_iter = original_graph

    canonical: set[tuple[int, int]] = set()
    for edge in edge_iter:
        if len(edge) < 2:
            raise BlockExtractionError(f"invalid fine edge: {edge!r}")
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            raise BlockExtractionError("self-loops are not supported by the decoder extractor")
        if not (0 <= u < n_orig and 0 <= v < n_orig):
            raise BlockExtractionError(
                f"fine edge ({u}, {v}) is outside contiguous node range [0, {n_orig})"
            )
        canonical.add((min(u, v), max(u, v)))
    return _edge_array(sorted(canonical))


def _coarse_edge_pairs(record: Mapping[str, Any], n_c: int) -> set[tuple[int, int]]:
    try:
        edge_index = _as_numpy(record["coarse"].edge_index)
    except (KeyError, AttributeError) as exc:
        raise BlockExtractionError("coarse record must contain coarse.edge_index") from exc

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise BlockExtractionError("coarse edge_index must have shape (2, m)")

    pairs: set[tuple[int, int]] = set()
    for u_raw, v_raw in edge_index.T:
        u, v = int(u_raw), int(v_raw)
        if u == v:
            continue
        if not (0 <= u < n_c and 0 <= v < n_c):
            raise BlockExtractionError(
                f"coarse edge ({u}, {v}) is outside cluster range [0, {n_c})"
            )
        pairs.add((min(u, v), max(u, v)))
    return pairs


def _is_connected(n_nodes: int, edges: Sequence[tuple[int, int]]) -> bool:
    if n_nodes <= 1:
        return True
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == n_nodes


def extract_decoder_blocks(
    original_graph: Any,
    coarse_record: Mapping[str, Any],
    *,
    require_connected_clusters: bool = True,
) -> DecoderBlocks:
    """Partition every fine edge into exactly one intra or inter block.

    Parameters
    ----------
    original_graph:
        A NetworkX-like graph, a PyG-like object with ``edge_index``, or an
        iterable of endpoint pairs.  Node ids must be contiguous ``0..N-1``, as
        required by the existing coarsening pipeline.
    coarse_record:
        One dictionary produced by :mod:`datasets.coarsen_pipeline`.
    require_connected_clusters:
        Fail if a Loukas cluster does not induce a connected fine subgraph.
    """
    assignment = np.asarray(coarse_record["assignment"], dtype=np.int64).reshape(-1)
    cluster_sizes = np.asarray(coarse_record["cluster_sizes"], dtype=np.int64).reshape(-1)
    n_orig = int(coarse_record["n_orig"])
    n_c = int(coarse_record["n_c"])

    if assignment.size != n_orig:
        raise BlockExtractionError(
            f"assignment has {assignment.size} entries but n_orig={n_orig}"
        )
    if cluster_sizes.size != n_c:
        raise BlockExtractionError(
            f"cluster_sizes has {cluster_sizes.size} entries but n_c={n_c}"
        )
    if n_c <= 0 or np.any(assignment < 0) or np.any(assignment >= n_c):
        raise BlockExtractionError("assignment contains an invalid cluster id")

    actual_sizes = np.bincount(assignment, minlength=n_c).astype(np.int64)
    if not np.array_equal(actual_sizes, cluster_sizes):
        raise BlockExtractionError(
            f"cluster_sizes mismatch: stored={cluster_sizes.tolist()}, "
            f"actual={actual_sizes.tolist()}"
        )
    if np.any(cluster_sizes < 1) or int(cluster_sizes.sum()) != n_orig:
        raise BlockExtractionError("clusters must be nonempty and sum to n_orig")

    original_edges = _canonical_edges(original_graph, n_orig)
    coarse_pairs = _coarse_edge_pairs(coarse_record, n_c)
    nodes_by_cluster = tuple(np.flatnonzero(assignment == i) for i in range(n_c))
    local_index = np.empty(n_orig, dtype=np.int64)
    for nodes in nodes_by_cluster:
        local_index[nodes] = np.arange(nodes.size, dtype=np.int64)

    intra_edges: list[list[tuple[int, int]]] = [[] for _ in range(n_c)]
    inter_edges: dict[tuple[int, int], list[tuple[int, int]]] = {
        pair: [] for pair in coarse_pairs
    }

    for u_raw, v_raw in original_edges:
        u, v = int(u_raw), int(v_raw)
        cu, cv = int(assignment[u]), int(assignment[v])
        if cu == cv:
            lu, lv = int(local_index[u]), int(local_index[v])
            intra_edges[cu].append((min(lu, lv), max(lu, lv)))
            continue

        left, right = min(cu, cv), max(cu, cv)
        pair = (left, right)
        if pair not in inter_edges:
            raise BlockExtractionError(
                f"fine edge ({u}, {v}) connects clusters {pair}, absent from G_c"
            )
        if cu == left:
            inter_edges[pair].append((int(local_index[u]), int(local_index[v])))
        else:
            inter_edges[pair].append((int(local_index[v]), int(local_index[u])))

    empty_coarse_pairs = [pair for pair, edges in inter_edges.items() if not edges]
    if empty_coarse_pairs:
        raise BlockExtractionError(
            f"G_c contains edges with no corresponding fine edge: {empty_coarse_pairs}"
        )

    intra_blocks: list[IntraBlock] = []
    for cluster_id in range(n_c):
        edges = sorted(intra_edges[cluster_id])
        connected = _is_connected(int(cluster_sizes[cluster_id]), edges)
        if require_connected_clusters and not connected:
            raise BlockExtractionError(
                f"cluster {cluster_id} is disconnected in the induced fine subgraph"
            )
        intra_blocks.append(
            IntraBlock(
                cluster_id=cluster_id,
                global_nodes=nodes_by_cluster[cluster_id].copy(),
                edges=_edge_array(edges),
                connected=connected,
            )
        )

    inter_blocks = tuple(
        InterBlock(
            left_cluster=left,
            right_cluster=right,
            left_global_nodes=nodes_by_cluster[left].copy(),
            right_global_nodes=nodes_by_cluster[right].copy(),
            edges=_edge_array(sorted(inter_edges[(left, right)])),
        )
        for left, right in sorted(coarse_pairs)
    )

    result = DecoderBlocks(
        n_orig=n_orig,
        n_c=n_c,
        assignment=assignment.copy(),
        cluster_sizes=cluster_sizes.copy(),
        intra_blocks=tuple(intra_blocks),
        inter_blocks=inter_blocks,
        original_edges=original_edges,
        source_index=(
            int(coarse_record["source_index"])
            if coarse_record.get("source_index") is not None
            else None
        ),
    )

    if result.intra_edge_count + result.inter_edge_count != result.n_edges:
        raise BlockExtractionError("decoder block edge counts do not sum to |E|")
    if not np.array_equal(result.reconstructed_edges(), result.original_edges):
        raise BlockExtractionError("decoder blocks do not reconstruct the original graph")
    return result


def extract_decoder_dataset(
    original_graphs: Sequence[Any],
    coarse_records: Sequence[Mapping[str, Any]],
    *,
    require_connected_clusters: bool = True,
) -> list[DecoderBlocks]:
    """Extract a split, safely matching records to their source graphs.

    New caches carry ``source_index`` and tolerate skipped coarse records.
    Legacy caches are accepted only when graph and record counts are identical,
    in which case their historical positional alignment is unambiguous.
    """
    has_source_indices = all(record.get("source_index") is not None for record in coarse_records)
    if not has_source_indices and len(original_graphs) != len(coarse_records):
        raise BlockExtractionError(
            "legacy coarse cache has no source_index and its length differs from the "
            "fine-graph split; rebuild the cache to avoid silent graph misalignment"
        )

    extracted: list[DecoderBlocks] = []
    seen_indices: set[int] = set()
    for position, record in enumerate(coarse_records):
        source_index = int(record["source_index"]) if has_source_indices else position
        if source_index in seen_indices:
            raise BlockExtractionError(f"duplicate source_index {source_index}")
        if not 0 <= source_index < len(original_graphs):
            raise BlockExtractionError(f"source_index {source_index} is outside the fine split")
        seen_indices.add(source_index)
        extracted.append(
            extract_decoder_blocks(
                original_graphs[source_index],
                record,
                require_connected_clusters=require_connected_clusters,
            )
        )
    return extracted
