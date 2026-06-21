"""Coarsening diagnostic for the BlockDiffusion planar dataset.

Coarsens a sample of planar graphs at several reduction ratios `r` and reports
the distributions that drive two pending design decisions:

  1. Which `r` to use  -> cluster-size `v_i`, coarse-graph size `n_c`, `|E_c|`,
                          stall behaviour, coarsening time.
  2. `K` for edge-weight buckets (Option A) -> super-edge weight distribution
                          (`A_c[i,j]` = #contracted cross-edges), with percentiles
                          and a suggested `K`.

Run from inside BlockDiffusion/ (so `coarsen_mini` is importable):

    python coarsen_diagnostic.py --n-graphs 50 --n-nodes 500 \
        --r-list 0.5,0.8,0.9,0.95 --K 100

By default it generates its own small planar sample (Delaunay, same construction
as datasets/generate_planar.py). Pass --from-pickle data/planar500/train.pkl to
use already-generated graphs instead.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay

import torch
from torch_geometric.data import Data

from coarsen_mini.algorithms.loukas.main import coarsen_loukas
from coarsen_mini.core.builders import build_coarsening


# --------------------------------------------------------------------------- #
# Graph source
# --------------------------------------------------------------------------- #
def planar_graph(n_nodes: int, rng: np.random.Generator) -> nx.Graph:
    """One connected planar graph via Delaunay (mirrors generate_planar.py)."""
    for _ in range(10):
        pts = rng.random((n_nodes, 2))
        try:
            tri = Delaunay(pts)
        except Exception:
            continue
        edges = set()
        for s in tri.simplices:
            a, b, c = int(s[0]), int(s[1]), int(s[2])
            edges.update({(min(a, b), max(a, b)),
                          (min(a, c), max(a, c)),
                          (min(b, c), max(b, c))})
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        if nx.is_connected(G) and nx.check_planarity(G)[0]:
            return G
    raise RuntimeError("could not generate a connected planar graph")


def nx_to_data(G: nx.Graph) -> Data:
    """Minimal PyG Data (unweighted, x=ones) for coarsen_mini."""
    n = G.number_of_nodes()
    if G.number_of_edges() == 0:
        raise ValueError("empty graph")
    e = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([e, e.flip(0)], dim=1)          # undirected, both dirs
    return Data(x=torch.ones(n, 1), edge_index=edge_index, num_nodes=n)


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def pct(a: np.ndarray, qs=(50, 90, 95, 99, 100)) -> dict:
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def describe(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), **pct(a)}


def super_edge_weights(coars) -> np.ndarray:
    """Undirected super-edge weights from the coarsened Data (one per edge)."""
    ei = coars.coarsened.edge_index
    ew = coars.coarsened.edge_weight
    if ew is None or ei.numel() == 0:
        return np.zeros(0)
    m = ei[0] < ei[1]                                       # upper triangle only
    return ew[m].detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_for_r(graphs_data, r, laplacian_kind, method, K):
    """Coarsen every graph at ratio r; pool the per-graph diagnostics."""
    v_all, w_all = [], []
    n_c_list, ec_list, ract_list, time_list = [], [], [], []
    n_stalled = 0
    for data in graphs_data:
        t0 = time.perf_counter()
        partition, info = coarsen_loukas(
            data, r=r, laplacian_kind=laplacian_kind, method=method, K=K,
            progress=False,
        )
        coars = build_coarsening(data, partition, method="loukas",
                                 laplacian_kind=laplacian_kind, r_requested=r)
        time_list.append(time.perf_counter() - t0)

        v_all.append(partition.cluster_sizes.astype(np.float64))
        w_all.append(super_edge_weights(coars))
        n_c_list.append(coars.n)
        ei = coars.coarsened.edge_index
        ec_list.append(int((ei[0] < ei[1]).sum().item()))
        ract_list.append(float(info["r_actual"]))
        n_stalled += int(bool(info.get("stalled", False)))

    v_all = np.concatenate(v_all) if v_all else np.zeros(0)
    w_all = np.concatenate(w_all) if w_all else np.zeros(0)
    n_c = np.array(n_c_list, dtype=float)
    ec = np.array(ec_list, dtype=float)

    # Suggested K: covers the 99th-pct weight; everything above goes to ">=K".
    suggested_K = int(np.ceil(np.percentile(w_all, 99))) if w_all.size else 0

    return {
        "r_requested": r,
        "r_actual_mean": float(np.mean(ract_list)) if ract_list else float("nan"),
        "n_stalled": n_stalled,
        "n_c": {"mean": float(n_c.mean()), "min": float(n_c.min()),
                "max": float(n_c.max())},
        "E_c": {"mean": float(ec.mean()), "min": float(ec.min()),
                "max": float(ec.max())},
        "cluster_size_v": describe(v_all),
        "super_edge_weight": describe(w_all),
        "weight_frac_eq1": float((w_all == 1).mean()) if w_all.size else float("nan"),
        "weight_frac_le2": float((w_all <= 2).mean()) if w_all.size else float("nan"),
        "coarsen_time_s_mean": float(np.mean(time_list)) if time_list else float("nan"),
        "suggested_K": suggested_K,
        # training-example volume (per graph, on average)
        "intra_problems_per_graph": float(n_c.mean()),
        "inter_problems_per_graph": float(ec.mean()),
    }


def print_report(results, args):
    line = "=" * 78
    print("\n" + line)
    print(f"COARSENING DIAGNOSTIC  ({args.n_graphs} planar graphs, "
          f"n_nodes={args.n_nodes}, laplacian={args.laplacian_kind}, "
          f"method={args.method}, K={args.K})")
    print(line)
    hdr = (f"{'r_req':>6} {'r_act':>6} {'n_c':>7} {'|E_c|':>7} "
           f"{'v̄':>6} {'v_max':>6} {'w̄':>6} {'w_p99':>6} {'w_max':>6} "
           f"{'sugK':>5} {'stall':>6} {'t(s)':>7}")
    print(hdr)
    print("-" * 78)
    for res in results:
        v = res["cluster_size_v"]
        w = res["super_edge_weight"]
        print(f"{res['r_requested']:>6.2f} {res['r_actual_mean']:>6.3f} "
              f"{res['n_c']['mean']:>7.1f} {res['E_c']['mean']:>7.1f} "
              f"{v.get('mean', float('nan')):>6.2f} {v.get('max', float('nan')):>6.0f} "
              f"{w.get('mean', float('nan')):>6.2f} {w.get('p99', float('nan')):>6.1f} "
              f"{w.get('max', float('nan')):>6.0f} {res['suggested_K']:>5d} "
              f"{res['n_stalled']:>6d} {res['coarsen_time_s_mean']:>7.3f}")
    print(line)
    print("Legend: v̄/v_max = cluster-size mean/max (intra padding budget);")
    print("        w̄/w_p99/w_max = super-edge weight mean/99th/max (Option-A buckets);")
    print("        sugK = ceil(99th-pct weight); stall = #graphs Loukas couldn't reduce.")
    print(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Coarsening diagnostic.")
    ap.add_argument("--n-graphs", type=int, default=50)
    ap.add_argument("--n-nodes", type=int, default=500)
    ap.add_argument("--r-list", type=str, default="0.5,0.8,0.9,0.95")
    ap.add_argument("--K", type=int, default=100, help="#preserved eigenvectors")
    ap.add_argument("--laplacian-kind", type=str, default="normalized_self_loop")
    ap.add_argument("--method", type=str, default="edges",
                    choices=["edges", "neighborhood"])
    ap.add_argument("--from-pickle", type=str, default=None,
                    help="Load nx graphs from this pickle instead of generating.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="coarsen_diagnostic.json")
    args = ap.parse_args()

    r_list = [float(x) for x in args.r_list.split(",")]

    # --- assemble the sample of PyG Data --------------------------------------
    if args.from_pickle:
        print(f"Loading graphs from {args.from_pickle}")
        with open(args.from_pickle, "rb") as fh:
            nx_graphs = pickle.load(fh)[: args.n_graphs]
        args.n_graphs = len(nx_graphs)
    else:
        print(f"Generating {args.n_graphs} planar graphs (n={args.n_nodes}) ...")
        rng = np.random.default_rng(args.seed)
        nx_graphs = [planar_graph(args.n_nodes, rng) for _ in range(args.n_graphs)]
    graphs_data = [nx_to_data(G) for G in nx_graphs]
    print(f"  ready: {len(graphs_data)} graphs "
          f"(avg edges {np.mean([G.number_of_edges() for G in nx_graphs]):.0f})")

    results = []
    for r in r_list:
        print(f"\nCoarsening at r={r} ...", flush=True)
        res = run_for_r(graphs_data, r, args.laplacian_kind, args.method, args.K)
        results.append(res)
        v, w = res["cluster_size_v"], res["super_edge_weight"]
        print(f"  n_c≈{res['n_c']['mean']:.0f}  v̄={v['mean']:.2f} "
              f"(max {v['max']:.0f})  w̄={w['mean']:.2f} "
              f"(p99 {w['p99']:.1f}, max {w['max']:.0f})  "
              f"suggested K={res['suggested_K']}  stalled={res['n_stalled']}")

    print_report(results, args)

    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    print(f"Full diagnostic written to {args.out}")


if __name__ == "__main__":
    main()
