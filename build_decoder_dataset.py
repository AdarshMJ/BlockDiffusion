"""Build lossless Stage-D/E block datasets from fine/coarse graph pairs.

Run from ``BlockDiffusion/`` after the planar and coarse caches exist::

    python build_decoder_dataset.py \
        --planar-dir data/planar500 \
        --coarse-dir data/coarse_planar/r0.8_normalized_edges_K100 \
        --out-dir data/decoder_blocks/r0.8_normalized_edges_K100

Each output ``<split>.pt`` is a list of plain dictionaries.  No custom Python
class needs to be importable when a downstream training job calls
``torch.load``.  ``summary.json`` records block-size/budget distributions and
the result of the connectivity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from datasets.decoder_blocks import BlockExtractionError, extract_decoder_dataset


def _torch_load(path: Path):
    """Load trusted project caches across old/new torch defaults."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch versions before weights_only was introduced
        return torch.load(path)


def _distribution(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def summarize(records: list[dict], elapsed_seconds: float) -> dict:
    cluster_sizes: list[int] = []
    intra_budgets: list[int] = []
    inter_budgets: list[int] = []
    disconnected: list[dict] = []
    n_edges: list[int] = []

    for record_index, record in enumerate(records):
        cluster_sizes.extend(int(v) for v in record["cluster_sizes"])
        n_edges.append(int(record["original_edges"].shape[0]))
        for block in record["intra_blocks"]:
            intra_budgets.append(int(block["e_i"]))
            if not block["connected"]:
                disconnected.append({
                    "record_index": record_index,
                    "source_index": record["source_index"],
                    "cluster_id": int(block["cluster_id"]),
                })
        inter_budgets.extend(int(block["w_ij"]) for block in record["inter_blocks"])

    return {
        "num_graphs": len(records),
        "num_intra_blocks": len(intra_budgets),
        "num_inter_blocks": len(inter_budgets),
        "elapsed_seconds": elapsed_seconds,
        "graphs_per_second": len(records) / elapsed_seconds if elapsed_seconds else None,
        "nodes_per_cluster": _distribution(cluster_sizes),
        "intra_edges_per_block": _distribution(intra_budgets),
        "inter_edges_per_block": _distribution(inter_budgets),
        "fine_edges_per_graph": _distribution(n_edges),
        "num_disconnected_clusters": len(disconnected),
        "disconnected_clusters": disconnected,
    }


def build_split(
    planar_path: Path,
    coarse_path: Path,
    output_path: Path,
    *,
    require_connected_clusters: bool,
    force: bool,
) -> dict:
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")

    with planar_path.open("rb") as handle:
        original_graphs = pickle.load(handle)
    coarse_records = _torch_load(coarse_path)

    print(
        f"[{planar_path.stem}] extracting {len(coarse_records)} coarse records "
        f"from {len(original_graphs)} fine graphs",
        flush=True,
    )
    started = time.perf_counter()
    blocks = extract_decoder_dataset(
        original_graphs,
        coarse_records,
        require_connected_clusters=require_connected_clusters,
    )
    records = [item.to_record() for item in blocks]
    elapsed = time.perf_counter() - started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(records, temporary_path)
    os.replace(temporary_path, output_path)

    result = summarize(records, elapsed)
    result.update({
        "fine_source": str(planar_path),
        "coarse_source": str(coarse_path),
        "output": str(output_path),
    })
    print(
        f"[{planar_path.stem}] saved {len(records)} graphs, "
        f"{result['num_intra_blocks']} intra blocks, "
        f"{result['num_inter_blocks']} inter blocks -> {output_path} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar-dir", type=Path, default=Path("data/planar500"))
    parser.add_argument("--coarse-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--allow-disconnected-clusters", action="store_true",
        help="record disconnected clusters instead of failing immediately",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    for split in args.splits:
        planar_path = args.planar_dir / f"{split}.pkl"
        coarse_path = args.coarse_dir / f"{split}.pt"
        if not planar_path.exists():
            raise FileNotFoundError(planar_path)
        if not coarse_path.exists():
            raise FileNotFoundError(coarse_path)
        summaries[split] = build_split(
            planar_path,
            coarse_path,
            args.out_dir / f"{split}.pt",
            require_connected_clusters=not args.allow_disconnected_clusters,
            force=args.force,
        )

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except BlockExtractionError as exc:
        raise SystemExit(f"decoder block extraction failed: {exc}") from exc
