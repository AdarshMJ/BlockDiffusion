"""benchmark/algorithms/common.py — helpers shared by all algorithms in benchmark.algorithms.

Right now: just `resolve_target` which standardizes the "how big should the
coarsened graph be?" question across all algos. Every algo in benchmark.algorithms
should accept the same dual `r OR n_clusters` interface, never both, never neither.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def resolve_target(
    N: int,
    r: Optional[float],
    n_clusters: Optional[int],
) -> tuple[int, float]:
    """Standardize the target-size specification across all coarsening algos.

    Exactly one of `r` or `n_clusters` must be given.

    Returns
    -------
    (n_target, r_equivalent) :
        n_target : int, the requested number of clusters
        r_equivalent : float, the equivalent reduction ratio (1 - n_target/N)
    """
    if (r is None) == (n_clusters is None):
        raise ValueError(
            "Specify exactly one of `r` (float in (0,1)) or `n_clusters` (int in [1, N]); "
            f"got r={r}, n_clusters={n_clusters}"
        )
    if r is not None:
        if not (0.0 < r < 1.0):
            raise ValueError(f"r must be in (0, 1), got {r}")
        n_target = max(1, int(np.ceil((1.0 - r) * N)))
        return n_target, r
    # n_clusters route
    if not (1 <= n_clusters <= N):
        raise ValueError(f"n_clusters must be in [1, {N}], got {n_clusters}")
    return int(n_clusters), 1.0 - n_clusters / N
