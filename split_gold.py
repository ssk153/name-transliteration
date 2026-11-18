#!/usr/bin/env python3
"""Create a reproducible 80:10:10 gold split, grouped by exact Thai name."""

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path,
                        default=Path("contest3-2026/romanization-train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite existing output: {args.output_dir}")
        raw = args.input_csv.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        fields = reader.fieldnames
        if not {"name", "romanize1"}.issubset(fields or []):
            raise ValueError("Input CSV must contain name and romanize1 columns.")
        rows = list(reader)
        for line, row in enumerate(rows, 2):
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(f"Malformed CSV row on line {line}.")
            if not row["name"].strip() or not row["romanize1"].strip():
                raise ValueError(f"Blank name or romanize1 on CSV line {line}.")
        names = sorted({row["name"] for row in rows})
        if len(names) < 10:
            raise ValueError("At least 10 unique names are required for an 80:10:10 split.")
        random.Random(args.seed).shuffle(names)
        train_end = len(names) * 80 // 100
        validation_end = train_end + (len(names) - train_end) // 2
        groups = {
            "train": set(names[:train_end]),
            "validation": set(names[train_end:validation_end]),
            "test": set(names[validation_end:]),
        }
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[first] & groups[second]:
                raise RuntimeError("Name overlap between splits.")
        assignments = {name: split for split, group in groups.items() for name in group}
        partitions = {split: [] for split in groups}
        for row in rows:
            partitions[assignments[row["name"]]].append(row)
        if sum(map(len, partitions.values())) != len(rows):
            raise RuntimeError("Split did not preserve every source row.")
        args.output_dir.mkdir(parents=True)
        manifest = {
            "source_csv": str(args.input_csv.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": args.seed, "requested_ratio": "80:10:10",
            "group_by": "exact name; no text normalization",
            "rounding": "floor 80% of unique names for train; divide remainder equally, extra to test",
            "source_rows": len(rows), "source_unique_names": len(names),
            "all_columns_and_rows_preserved": True,
            "row_order": "original source order within each split",
            "pairwise_name_overlap": 0, "splits": {},
        }
        for split, subset in partitions.items():
            path = args.output_dir / f"gold_{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset)
            manifest["splits"][split] = {
                "file": path.name, "rows": len(subset), "unique_names": len(groups[split]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            print(f"{split}: {len(subset)} rows, {len(groups[split])} unique names -> {path}")
        (args.output_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Split failed: {error}\n")


if __name__ == "__main__":
    main()
