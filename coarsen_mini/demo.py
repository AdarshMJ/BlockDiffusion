"""coarsen_mini demo — load Cora, coarsen via Loukas (3 variants) + Graclus.

Run from the parent directory of `coarsen_mini/` :
    cd /path/to/parent && python -m coarsen_mini.demo
or
    PYTHONPATH=/path/to/parent python coarsen_mini/demo.py
"""

import torch
from torch_geometric.datasets import Planetoid

from coarsen_mini.algorithms.loukas.main import coarsen_loukas
from coarsen_mini.core.builders import build_coarsening


def main():
    torch.manual_seed(14)
    # Adjust root to wherever you keep the Cora download
    data = Planetoid(root="/tmp/cora_demo", name="Cora")[0]
    print(f"Cora : N={data.num_nodes}  E={data.num_edges//2}  features={data.x.shape[1]}\n")

    # --- Loukas, combinatorial Laplacian preserved ---
    partition, info = coarsen_loukas(
        data, r=0.5, laplacian_kind="combinatorial", progress=False,
    )
    print(f"Loukas — combinatorial preserved : n_clusters={info['n_clusters']:5d}  r_actual={info['r_actual']:.3f}")

    # --- Loukas, normalized_self_loop Laplacian preserved (default, matches GCN propagation) ---
    partition, info = coarsen_loukas(
        data, r=0.5, laplacian_kind="normalized_self_loop", progress=False,
    )
    print(f"Loukas — nsl preserved          : n_clusters={info['n_clusters']:5d}  r_actual={info['r_actual']:.3f}")

    # --- Loukas, neighborhood matching method (vs default 'edges') ---
    partition_nh, info_nh = coarsen_loukas(
        data, r=0.5, method="neighborhood", progress=False,
    )
    print(f"Loukas — nsl + neighborhood     : n_clusters={info_nh['n_clusters']:5d}  r_actual={info_nh['r_actual']:.3f}")

    # --- Graclus (requires torch-cluster) ---
    try:
        from coarsen_mini.algorithms.graclus import coarsen_graclus
        partition_g, info_g = coarsen_graclus(data, r=0.5)
        print(f"Graclus                          : n_clusters={info_g['n_clusters']:5d}  r_actual={info_g['r_actual']:.3f}")
    except ImportError as e:
        print(f"Graclus skipped — {e}")

    # --- Assemble the full Coarsening (Q, P, A_c, X_c) for downstream use ---
    coars = build_coarsening(data, partition, method="loukas", r_requested=0.5)
    print(
        f"\nFull Coarsening (built from Loukas partition):"
        f"\n  Q.shape={tuple(coars.Q.shape)}  P.shape={tuple(coars.P.shape)}"
        f"\n  A_c built (n×n), X_c built (cluster-mean of features)"
        f"\n  laplacian_kind stored = {coars.laplacian_kind}"
    )


if __name__ == "__main__":
    main()
