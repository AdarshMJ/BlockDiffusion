"""Generate a synthetic planar-graph dataset for BlockDiffusion.

Planar graphs are produced by **Delaunay triangulation** of random points in the
unit square — the same construction used by the SPECTRE planar benchmark. A
Delaunay triangulation of points in general position is always a connected planar
graph, so every generated graph is guaranteed planar and connected (we assert
both as a sanity check, and regenerate the rare degenerate / collinear case).

Default: 5000 training graphs of 500 nodes each (the BlockDiffusion training set),
plus held-out val/test splits. Graphs are saved as a pickle list of
``networkx.Graph`` objects per split — the same format consumed by the EDGE-style
loader (`_EdgeNetworkDataModule`), and trivially convertible to PyG `Data` via
`datasets.spectre_dataset._nx_to_pyg` or fed straight into `coarsen_mini`.

Usage
-----
    cd BlockDiffusion
    python -m datasets.generate_planar \
        --n-train 5000 --n-val 500 --n-test 500 \
        --n-nodes 500 --out data/planar500 --seed 42

Output (under --out):
    train.pkl  val.pkl  test.pkl   (each: list[networkx.Graph])
    meta.json                       (generation parameters + summary stats)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay


def delaunay_graph(points: np.ndarray) -> nx.Graph:
    """Build the undirected Delaunay graph for a 2D point array."""
    points = np.asarray(points)
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edges.add((min(a, b), max(a, b)))
        edges.add((min(a, c), max(a, c)))
        edges.add((min(b, c), max(b, c)))
    graph = nx.Graph()
    graph.add_nodes_from(range(points.shape[0]))
    graph.add_edges_from(edges)
    return graph


def generate_planar_graph_and_positions(
    n_nodes: int,
    rng: np.random.Generator,
    max_tries: int = 10,
) -> tuple[nx.Graph, np.ndarray]:
    """Return one connected planar graph and its generating 2D points.

    Sample `n_nodes` uniform points in the unit square and take the edges of
    their Delaunay triangulation. Retries on the (measure-zero) degenerate case
    where the points are collinear / the result is not planar+connected.
    """
    for _ in range(max_tries):
        pts = rng.random((n_nodes, 2))
        try:
            G = delaunay_graph(pts)
        except Exception:
            continue  # collinear / coplanar input — resample

        if G.number_of_nodes() == n_nodes and nx.is_connected(G) \
                and nx.check_planarity(G)[0]:
            return G, pts

    raise RuntimeError(
        f"Failed to generate a connected planar graph on {n_nodes} nodes "
        f"after {max_tries} tries (degenerate point samples)."
    )


def generate_planar_graph(n_nodes: int, rng: np.random.Generator,
                          max_tries: int = 10) -> nx.Graph:
    """Return one connected planar graph on exactly ``n_nodes`` nodes."""
    graph, _ = generate_planar_graph_and_positions(n_nodes, rng, max_tries)
    return graph


def generate_split(n_graphs: int, n_nodes: int, rng: np.random.Generator,
                   label: str) -> list[nx.Graph]:
    """Generate a list of `n_graphs` planar graphs, with progress printing."""
    graphs: list[nx.Graph] = []
    t0 = time.perf_counter()
    for i in range(n_graphs):
        graphs.append(generate_planar_graph(n_nodes, rng))
        if (i + 1) % 250 == 0 or (i + 1) == n_graphs:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            eta = (n_graphs - (i + 1)) / rate
            print(f"  [{label}] {i + 1}/{n_graphs}  "
                  f"({rate:.1f} graphs/s, ETA {eta:5.1f}s)", flush=True)
    return graphs


def summarize(graphs: list[nx.Graph]) -> dict:
    """Mean/std of basic structural stats over a split (for the meta file)."""
    n = np.array([G.number_of_nodes() for G in graphs], dtype=float)
    e = np.array([G.number_of_edges() for G in graphs], dtype=float)
    dens = np.array([nx.density(G) for G in graphs], dtype=float)
    return {
        "num_graphs": len(graphs),
        "nodes_mean": float(n.mean()), "nodes_std": float(n.std()),
        "edges_mean": float(e.mean()), "edges_std": float(e.std()),
        "density_mean": float(dens.mean()), "density_std": float(dens.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic planar graphs.")
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Number of training graphs (default 5000).")
    parser.add_argument("--n-val", type=int, default=500,
                        help="Number of validation graphs (default 500).")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Number of test graphs (default 500).")
    parser.add_argument("--n-nodes", type=int, default=500,
                        help="Nodes per graph (default 500).")
    parser.add_argument("--out", type=str, default="data/planar500",
                        help="Output directory (relative to cwd).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Independent RNG streams per split (disjoint, reproducible from one seed).
    ss = np.random.SeedSequence(args.seed)
    rng_train, rng_val, rng_test = (np.random.default_rng(s)
                                    for s in ss.spawn(3))

    print(f"Generating planar dataset (n_nodes={args.n_nodes}, seed={args.seed}) "
          f"-> {args.out}")
    print(f"  train={args.n_train}  val={args.n_val}  test={args.n_test}")

    splits = {
        "train": (args.n_train, rng_train),
        "val":   (args.n_val,   rng_val),
        "test":  (args.n_test,  rng_test),
    }

    meta = {"n_nodes": args.n_nodes, "seed": args.seed,
            "generator": "delaunay_unit_square", "splits": {}}

    for name, (count, rng) in splits.items():
        if count <= 0:
            continue
        graphs = generate_split(count, args.n_nodes, rng, name)
        out_path = os.path.join(args.out, f"{name}.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump(graphs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        stats = summarize(graphs)
        meta["splits"][name] = {"path": out_path, **stats}
        print(f"  [{name}] saved {count} graphs -> {out_path}  "
              f"(edges {stats['edges_mean']:.0f}±{stats['edges_std']:.0f}, "
              f"density {stats['density_mean']:.4f})")

    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Done. Metadata -> {os.path.join(args.out, 'meta.json')}")


if __name__ == "__main__":
    main()
