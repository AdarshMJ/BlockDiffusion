"""benchmark/algorithms/loukas/matching.py — greedy matching for Loukas contraction.

Two-pass strategy for edges (see README.md for the design rationale):

  Pass 1 — disjoint greedy : paper-faithful Loukas. Pop edges cheapest first,
  accept iff both endpoints are unassigned, mark them as a new cluster of size 2.

  Pass 2 — absorption (optional, on stall) : when Pass 1 under-delivers
  (`n_removed < stall_threshold * n_target_removed`), iterate over remaining
  edges. For each edge with exactly one endpoint already in a cluster,
  recompute the proper spectral_cost_neighborhood on the extended cluster
  and accept if the cost is within `absorption_cost_factor` of the worst
  Pass 1 cost. Size-capped at `absorption_max_cluster_size`.

For neighborhood matching, a single pass with a soft overshoot rule on the
last accepted candidate (cf. README "neighborhood overshoot final").

Return convention (both functions):
  coarsening_list : list[np.ndarray]   one node-id array per supernode group
  info : dict                          per-level telemetry, see fields below
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from coarsen_mini.algorithms.loukas.cost import spectral_cost_neighborhood


def greedy_match_edges(
    costs: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    n_target_removed: int,
    *,
    max_take: Optional[int] = None,
    # Absorption fallback (only kicks in if Pass 1 stalls)
    absorption_on_stall: bool = False,
    stall_threshold: float = 0.3,
    absorption_cost_factor: float = 2.0,
    absorption_max_cluster_size: int = 10,
    A_inter: Optional[sp.csr_matrix] = None,
    A_cost: Optional[np.ndarray] = None,
    deg: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], dict]:
    """Greedy edge matching, disjoint by default, absorption fallback on stall.

    Pass 1 (disjoint) is the official Loukas heavy-edge matching. Pass 2
    (absorption) is our addition; see README.md for the math and the
    rationale. Pass 2 needs A_inter, A_cost, deg to recompute extended-cluster
    costs honestly.

    Returns
    -------
    coarsening_list : list of np.ndarray, each array = node-ids of one supernode
    info : dict with keys
        n_disjoint_accepted, n_absorption_accepted,
        n_absorption_skipped_too_costly, n_absorption_skipped_cluster_full,
        max_cluster_size, n_visited, total_cost, stalled
    """
    M = costs.shape[0]
    if M == 0:
        return [], _empty_info()

    if absorption_on_stall and (A_inter is None or A_cost is None or deg is None):
        raise ValueError("absorption_on_stall=True requires A_inter, A_cost, deg")

    N = int(max(edges_i.max(), edges_j.max())) + 1
    order = np.argsort(costs)

    # Pass 1 — disjoint
    node_cluster = -np.ones(N, dtype=np.int64)
    cluster_members: list[list[int]] = []
    cluster_cost: list[float] = []
    n_removed_disjoint = 0
    n_visited = 0
    limit = max_take if max_take is not None else M

    for idx in order:
        n_visited += 1
        i, j = int(edges_i[idx]), int(edges_j[idx])
        if node_cluster[i] >= 0 or node_cluster[j] >= 0:
            continue
        cid = len(cluster_members)
        node_cluster[i] = cid
        node_cluster[j] = cid
        cluster_members.append([i, j])
        cluster_cost.append(float(costs[idx]))
        n_removed_disjoint += 1
        if n_removed_disjoint >= n_target_removed or len(cluster_members) >= limit:
            break

    n_absorption_accepted = 0
    n_skip_costly = 0
    n_skip_size = 0

    # Pass 2 — absorption (only if stalled and enabled)
    stalled = (n_removed_disjoint < stall_threshold * n_target_removed) if n_target_removed > 0 else False
    if absorption_on_stall and stalled and cluster_cost:
        max_accepted_cost = max(cluster_cost)
        cost_ceiling = absorption_cost_factor * max_accepted_cost
        remaining_target = n_target_removed - n_removed_disjoint

        for idx in order:
            if n_absorption_accepted >= remaining_target:
                break
            i, j = int(edges_i[idx]), int(edges_j[idx])
            ci, cj = node_cluster[i], node_cluster[j]
            if ci >= 0 and cj >= 0:
                continue  # both assigned (same or different clusters): no-op
            if ci < 0 and cj < 0:
                continue  # both free but Pass 1 stopped early: not the absorption case

            free_node = i if ci < 0 else j
            host_cid = cj if ci < 0 else ci
            host_members = cluster_members[host_cid]

            if len(host_members) >= absorption_max_cluster_size:
                n_skip_size += 1
                continue

            extended = host_members + [free_node]
            nodes_arr = np.array(extended, dtype=np.int64)
            W_sub = np.asarray(A_inter[nodes_arr, :][:, nodes_arr].toarray())
            cost_recomputed = float(
                spectral_cost_neighborhood(nodes_arr, W_sub, A_cost, deg)
            )

            if cost_recomputed > cost_ceiling:
                n_skip_costly += 1
                continue

            cluster_members[host_cid].append(free_node)
            node_cluster[free_node] = host_cid
            n_absorption_accepted += 1

    # Final coarsening_list (singletons of unassigned nodes are NOT created here —
    # main.py handles survivors when building the intermediate Q matrix).
    coarsening_list = [np.array(m, dtype=np.int64) for m in cluster_members]

    cluster_sizes = [len(m) for m in cluster_members]
    info = {
        "n_disjoint_accepted": len(cluster_members) - n_absorption_accepted,
        "n_absorption_accepted": n_absorption_accepted,
        "n_absorption_skipped_too_costly": n_skip_costly,
        "n_absorption_skipped_cluster_full": n_skip_size,
        "n_removed": n_removed_disjoint + n_absorption_accepted,
        "n_visited": n_visited,
        "total_cost": sum(cluster_cost),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "stalled_pass1": stalled,
    }
    return coarsening_list, info


def greedy_match_neighborhood(
    costs: np.ndarray,
    neighborhoods: list[np.ndarray],
    n_target_removed: int,
    *,
    max_take: Optional[int] = None,
) -> tuple[list[np.ndarray], dict]:
    """Greedy neighborhood matching, disjoint, with last-step overshoot allowance.

    Each accepted neighborhood of size nc removes (nc - 1) nodes. If on a given
    iteration the cheapest neighborhood would overshoot `n_target_removed`, we
    skip it and look for a smaller one; however if NO candidate is accepted
    (= we'd return empty), we allow overshooting by 1 node to make progress.
    """
    M = len(neighborhoods)
    if M == 0:
        return [], _empty_info()
    if costs.shape[0] != M:
        raise ValueError(f"costs length {costs.shape[0]} != neighborhoods count {M}")

    order = np.argsort(costs)
    max_id = max(int(nbh.max()) for nbh in neighborhoods)
    marked = np.zeros(max_id + 1, dtype=bool)

    coarsening_list: list[np.ndarray] = []
    cluster_cost = []
    n_removed = 0
    n_visited = 0
    limit = max_take if max_take is not None else M

    # Pass 1 — strict (no overshoot)
    for idx in order:
        n_visited += 1
        nbh = neighborhoods[idx]
        if marked[nbh].any():
            continue
        nc = nbh.shape[0]
        if n_removed + (nc - 1) > n_target_removed:
            continue
        marked[nbh] = True
        coarsening_list.append(nbh)
        cluster_cost.append(float(costs[idx]))
        n_removed += nc - 1
        if n_removed >= n_target_removed or len(coarsening_list) >= limit:
            break

    # Overshoot allowance: if Pass 1 accepted zero candidates, take the
    # cheapest unmarked candidate even if it overshoots by 1.
    if not coarsening_list:
        for idx in order:
            nbh = neighborhoods[idx]
            if marked[nbh].any():
                continue
            marked[nbh] = True
            coarsening_list.append(nbh)
            cluster_cost.append(float(costs[idx]))
            n_removed += nbh.shape[0] - 1
            break

    sizes = [c.shape[0] for c in coarsening_list]
    info = {
        "n_disjoint_accepted": len(coarsening_list),
        "n_absorption_accepted": 0,
        "n_absorption_skipped_too_costly": 0,
        "n_absorption_skipped_cluster_full": 0,
        "n_removed": n_removed,
        "n_visited": n_visited,
        "total_cost": sum(cluster_cost),
        "max_cluster_size": max(sizes) if sizes else 0,
        "stalled_pass1": False,  # neighborhood does not stall the same way
    }
    return coarsening_list, info


def _empty_info() -> dict:
    return {
        "n_disjoint_accepted": 0,
        "n_absorption_accepted": 0,
        "n_absorption_skipped_too_costly": 0,
        "n_absorption_skipped_cluster_full": 0,
        "n_removed": 0,
        "n_visited": 0,
        "total_cost": 0.0,
        "max_cluster_size": 0,
        "stalled_pass1": False,
    }
