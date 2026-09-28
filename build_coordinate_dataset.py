"""Build centroid/offset coordinate caches aligned with decoder records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from datasets.coordinate_blocks import (
    build_coordinate_record,
    summarize_coordinate_records,
)
from datasets.decoder_dataset import load_decoder_records


def _atomic_torch_save(value, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--position-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in args.splits:
        decoder_records = load_decoder_records(args.decoder_dir / f"{split}.pt")
        position_archive = np.load(args.position_dir / f"{split}.npz")
        positions = position_archive["positions"]
        missing_sources = any(
            record.get("source_index") is None for record in decoder_records
        )
        if missing_sources and len(decoder_records) != len(positions):
            raise ValueError(
                f"{split}: legacy decoder records require positional alignment, "
                f"but lengths differ ({len(decoder_records)} != {len(positions)})"
            )

        coordinate_records = []
        for record_index, record in enumerate(decoder_records):
            source_value = record.get("source_index")
            source_index = record_index if source_value is None else int(source_value)
            if not 0 <= source_index < len(positions):
                raise ValueError(f"{split}: source_index {source_index} is out of range")
            coordinate = build_coordinate_record(record, positions[source_index])
            coordinate["source_index"] = source_index
            coordinate_records.append(coordinate)

        destination = args.output_dir / f"{split}.pt"
        _atomic_torch_save(coordinate_records, destination)
        summaries[split] = {
            **summarize_coordinate_records(coordinate_records),
            "path": str(destination),
            "position_dtype": str(positions.dtype),
        }
        print(json.dumps({"split": split, **summaries[split]}), flush=True)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Saved coordinate dataset summary -> {summary_path}")


if __name__ == "__main__":
    main()
