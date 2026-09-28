"""Generate a balanced two-block SBM dataset above the KS threshold.

The output format matches ``datasets.generate_planar``: each split is a pickle
containing a list of ``networkx.Graph`` objects.  Unlike the old fixed-size
planar set, each split mixes the requested graph sizes.  Every node has a
``block`` attribute (0 or 1), and every graph records ``n_nodes``, ``p_in``,
``p_out`` and ``ks_margin`` as graph attributes.

For two equal blocks, write ``c_in`` and ``c_out`` for the expected number of
same-block and other-block neighbours.  Then

    mean_degree = c_in + c_out
    KS margin   = (c_in - c_out)^2 / (2 * mean_degree)

and community recovery is possible asymptotically when the margin is greater
than one.  The default margin of 7 is deliberately well above the boundary.

Example (5,000/500/500 total graphs, half at each size)::

    cd BlockDiffusion
    python datasets/generate_sbm_ks.py --out data/sbm_ks_n64_n128

Calling the file directly is intentional: ``datasets/__init__.py`` imports the
PyTorch/PyG training stack, while dataset generation itself only needs NumPy
and NetworkX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time

import networkx as nx
import numpy as np


def solve_two_block_parameters(mean_degree: float, ks_margin: float) -> tuple[float, float]:
    """Return ``(c_in, c_out)`` for a balanced two-block SBM."""
    if mean_degree <= 0:
        raise ValueError("mean_degree must be positive")
    if ks_margin <= 1:
        raise ValueError("ks_margin must be greater than the recoverability threshold 1")
    if mean_degree < 2 * ks_margin:
        raise ValueError(
            "mean_degree must be at least 2 * ks_margin so c_out is non-negative"
        )
    delta = math.sqrt(2.0 * mean_degree * ks_margin)
    c_out = (mean_degree - delta) / 2.0
    c_in = c_out + delta
    return c_in, c_out


def probabilities_at_size(n_nodes: int, c_in: float, c_out: float) -> tuple[float, float]:
    """Convert expected block degrees to exact finite-size edge probabilities."""
    if n_nodes < 4 or n_nodes % 2:
        raise ValueError("n_nodes must be an even integer of at least 4")
    block_size = n_nodes // 2
    p_in = c_in / (block_size - 1)
    p_out = c_out / block_size
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError(
            f"invalid probabilities at n={n_nodes}: p_in={p_in:.6g}, p_out={p_out:.6g}"
        )
    return p_in, p_out


def ks_margin_from_probabilities(n_nodes: int, p_in: float, p_out: float) -> float:
    """Recompute the finite-size parameterized KS margin used by this generator."""
    block_size = n_nodes // 2
    c_in = p_in * (block_size - 1)
    c_out = p_out * block_size
    mean_degree = c_in + c_out
    return (c_in - c_out) ** 2 / (2.0 * mean_degree)


def generate_graph(
    n_nodes: int,
    p_in: float,
    p_out: float,
    ks_margin: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Sample one balanced SBM with a fresh random assignment of block labels."""
    labels = np.repeat(np.arange(2, dtype=np.int8), n_nodes // 2)
    rng.shuffle(labels)

    rows, cols = np.triu_indices(n_nodes, k=1)
    same_block = labels[rows] == labels[cols]
    probabilities = np.where(same_block, p_in, p_out)
    selected = rng.random(rows.size) < probabilities

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(zip(rows[selected].tolist(), cols[selected].tolist()))
    nx.set_node_attributes(graph, {i: int(labels[i]) for i in range(n_nodes)}, "block")
    graph.graph.update(
        n_nodes=n_nodes,
        p_in=float(p_in),
        p_out=float(p_out),
        ks_margin=float(ks_margin),
        generator="balanced_two_block_sbm",
    )
    return graph


def graph_fingerprint(graph: nx.Graph) -> str:
    """A stable labeled-graph fingerprint used to detect split leakage."""
    digest = hashlib.sha256()
    digest.update(f"n={graph.number_of_nodes()}|".encode())
    for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
        digest.update(f"{u},{v};".encode())
    return digest.hexdigest()


def summarize(graphs: list[nx.Graph]) -> dict:
    nodes = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    edges = np.asarray([g.number_of_edges() for g in graphs], dtype=float)
    degrees = 2.0 * edges / nodes
    connected = np.asarray([nx.is_connected(g) for g in graphs], dtype=float)
    by_size = {}
    for n_nodes in sorted({g.number_of_nodes() for g in graphs}):
        subset = [g for g in graphs if g.number_of_nodes() == n_nodes]
        subset_edges = np.asarray([g.number_of_edges() for g in subset], dtype=float)
        by_size[str(n_nodes)] = {
            "num_graphs": len(subset),
            "edges_mean": float(subset_edges.mean()),
            "edges_std": float(subset_edges.std()),
            "degree_mean": float((2.0 * subset_edges / n_nodes).mean()),
            "connected_fraction": float(np.mean([nx.is_connected(g) for g in subset])),
        }
    return {
        "num_graphs": len(graphs),
        "nodes_mean": float(nodes.mean()),
        "edges_mean": float(edges.mean()),
        "edges_std": float(edges.std()),
        "degree_mean": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "connected_fraction": float(connected.mean()),
        "by_size": by_size,
    }


def generate_split(
    count_per_size: int,
    sizes: list[int],
    probabilities: dict[int, tuple[float, float]],
    ks_margin: float,
    rng: np.random.Generator,
    label: str,
) -> list[nx.Graph]:
    graphs = []
    t0 = time.perf_counter()
    for n_nodes in sizes:
        p_in, p_out = probabilities[n_nodes]
        for _ in range(count_per_size):
            graphs.append(generate_graph(n_nodes, p_in, p_out, ks_margin, rng))
    rng.shuffle(graphs)
    elapsed = time.perf_counter() - t0
    print(
        f"  [{label}] generated {len(graphs)} graphs "
        f"({len(graphs) / max(elapsed, 1e-9):.1f} graphs/s)",
        flush=True,
    )
    return graphs


def validate_dataset(splits: dict[str, list[nx.Graph]], sizes: list[int]) -> None:
    """Check graph invariants, balanced labels, KS margin, and split leakage."""
    fingerprints: dict[str, str] = {}
    expected_sizes = set(sizes)
    for split, graphs in splits.items():
        for index, graph in enumerate(graphs):
            n_nodes = graph.number_of_nodes()
            if n_nodes not in expected_sizes or set(graph.nodes()) != set(range(n_nodes)):
                raise AssertionError(f"{split}[{index}] has invalid node set")
            if nx.number_of_selfloops(graph):
                raise AssertionError(f"{split}[{index}] contains self loops")
            labels = nx.get_node_attributes(graph, "block")
            counts = np.bincount([labels[i] for i in range(n_nodes)], minlength=2)
            if counts.tolist() != [n_nodes // 2, n_nodes // 2]:
                raise AssertionError(f"{split}[{index}] has unbalanced/missing block labels")
            margin = ks_margin_from_probabilities(
                n_nodes, graph.graph["p_in"], graph.graph["p_out"]
            )
            if margin <= 1.0:
                raise AssertionError(f"{split}[{index}] is below the KS threshold")
            fingerprint = graph_fingerprint(graph)
            if fingerprint in fingerprints:
                raise AssertionError(
                    f"duplicate labeled graph across {fingerprints[fingerprint]} and {split}[{index}]"
                )
            fingerprints[fingerprint] = f"{split}[{index}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--n-train-per-size", type=int, default=2500)
    parser.add_argument("--n-val-per-size", type=int, default=250)
    parser.add_argument("--n-test-per-size", type=int, default=250)
    parser.add_argument("--mean-degree", type=float, default=16.0)
    parser.add_argument("--ks-margin", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sbm_ks_n64_n128")
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    c_in, c_out = solve_two_block_parameters(args.mean_degree, args.ks_margin)
    probabilities = {
        n_nodes: probabilities_at_size(n_nodes, c_in, c_out) for n_nodes in sizes
    }
    for n_nodes, (p_in, p_out) in probabilities.items():
        recovered = ks_margin_from_probabilities(n_nodes, p_in, p_out)
        if not math.isclose(recovered, args.ks_margin, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("KS parameterization failed its round-trip check")

    print(
        f"Balanced 2-block SBM: sizes={sizes}, mean_degree={args.mean_degree:g}, "
        f"KS margin={args.ks_margin:g} (>1), seed={args.seed}"
    )
    for n_nodes, (p_in, p_out) in probabilities.items():
        print(f"  n={n_nodes}: p_in={p_in:.9f}, p_out={p_out:.9f}")

    counts = {
        "train": args.n_train_per_size,
        "val": args.n_val_per_size,
        "test": args.n_test_per_size,
    }
    seed_sequence = np.random.SeedSequence(args.seed)
    rngs = {
        split: np.random.default_rng(child)
        for split, child in zip(counts, seed_sequence.spawn(len(counts)))
    }
    splits = {
        split: generate_split(count, sizes, probabilities, args.ks_margin, rngs[split], split)
        for split, count in counts.items()
        if count > 0
    }
    validate_dataset(splits, sizes)

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "generator": "balanced_two_block_sbm",
        "seed": args.seed,
        "sizes": sizes,
        "count_per_size": counts,
        "mean_degree_target": args.mean_degree,
        "ks_margin_target": args.ks_margin,
        "ks_threshold": 1.0,
        "c_in": c_in,
        "c_out": c_out,
        "parameters_by_size": {
            str(n): {"p_in": probabilities[n][0], "p_out": probabilities[n][1]}
            for n in sizes
        },
        "notes": [
            "Block assignments are independently shuffled for every graph.",
            "Graphs are not rejected for connectivity; samples follow the stated SBM.",
            "Node attribute 'block' is ground truth and may be omitted from model inputs.",
        ],
        "splits": {},
    }
    for split, graphs in splits.items():
        path = os.path.join(args.out, f"{split}.pkl")
        with open(path, "wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        meta["splits"][split] = {"path": path, **summarize(graphs)}
        print(f"  [{split}] saved -> {path}")

    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Validated all graphs and split disjointness. Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
