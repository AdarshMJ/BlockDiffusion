"""OPTIONAL standalone prebuild of the coarse G_c cache.

You do NOT need to run this — `main.py` builds the coarse cache lazily on first
run (see CoarseGraphDataModule, driven by the `coarsening:` config section). This
script is only useful to prebuild the cache on a CPU node so GPU time isn't spent
coarsening. It writes the exact same cache `main.py` would build/read.

    python build_coarse_dataset.py --planar-dir data/planar500 \
        --cache-dir data/coarse_planar --r 0.9 --K 100

Output: <cache-dir>/<tag>/{train,val,test}.pt  where <tag> encodes the params.
"""

from __future__ import annotations

import argparse

from datasets.coarsen_pipeline import build_coarse_cache


def main():
    ap = argparse.ArgumentParser(description="Prebuild the coarse G_c cache (optional).")
    ap.add_argument("--planar-dir", type=str, default="data/planar500")
    ap.add_argument("--cache-dir", type=str, default="data/coarse_planar")
    ap.add_argument("--r", type=float, default=0.9)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--laplacian-kind", type=str, default="normalized_self_loop")
    ap.add_argument("--method", type=str, default="edges",
                    choices=["edges", "neighborhood"])
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    subdir = build_coarse_cache(
        args.planar_dir, args.cache_dir, args.r, args.K,
        args.laplacian_kind, args.method, force_rebuild=args.force_rebuild,
    )
    print(f"\nDone. Coarse cache at {subdir}")


if __name__ == "__main__":
    main()
