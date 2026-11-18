#!/usr/bin/env python3
"""Evaluate saved best checkpoints on the labeled local gold test split."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from train_model import EXPERIMENTS, edit_distance, file_hash, read_gold


def score_prediction(prediction, references):
    """Use the same case-sensitive, multi-reference policy as validation."""
    reference = min(references, key=lambda ref: edit_distance(prediction, ref) / len(ref))
    edits = edit_distance(prediction, reference)
    return {
        "exact_match": prediction in references,
        "cer_reference": reference,
        "edit_distance": edits,
        "reference_characters": len(reference),
        "cer": edits / len(reference),
    }


def evaluate_checkpoint(checkpoint, rows, args, device):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    predictions = []
    unknown_rows = 0
    for row in rows:
        ids = tokenizer.encode(row["name"])
        if len(ids) > args.max_length:
            raise ValueError("Test name exceeds --max-length; increase it to avoid truncating names.")
        unknown_rows += tokenizer.unk_token_id in ids
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            inputs = tokenizer([row["name"] for row in batch], padding=True, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs, do_sample=False, num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            predictions.extend(tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            ))
            print(f"  Predicted {len(predictions)}/{len(rows)} test rows", flush=True)
    if len(predictions) != len(rows):
        raise RuntimeError("Prediction count does not match test row count.")
    details = []
    for index, (row, prediction) in enumerate(zip(rows, predictions), 1):
        references = [row[key] for key in ("romanize1", "romanize2", "romanize3")
                      if (row.get(key) or "").strip()]
        details.append({"row": index, "name": row["name"], "prediction": prediction,
                        "references": json.dumps(references, ensure_ascii=False),
                        **score_prediction(prediction, references)})
    exact = sum(item["exact_match"] for item in details)
    edits = sum(item["edit_distance"] for item in details)
    chars = sum(item["reference_characters"] for item in details)
    metrics = {
        "test_rows": len(rows), "test_unique_names": len({r["name"] for r in rows}),
        "test_exact_match": exact / len(rows), "test_exact_matches": exact,
        "test_cer": edits / chars, "total_edit_distance": edits,
        "total_reference_characters": chars, "source_rows_with_unknown_tokens": unknown_rows,
    }
    return metrics, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--experiments", choices=EXPERIMENTS, nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/gold_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Pipeline check: evaluate first N rows; 0 evaluates the full split.")
    args = parser.parse_args()
    for key in ("batch_size", "max_length", "max_new_tokens", "num_beams", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    if args.limit < 0:
        parser.error("--limit cannot be negative.")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("Experiments must not be repeated.")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Refusing to overwrite existing output: {args.output_dir}")
    rows = read_gold(args.test_csv)
    test_hash, full_count = file_hash(args.test_csv), len(rows)
    runs = {}
    for experiment in args.experiments:
        folder = args.runs_dir / experiment
        checkpoint = folder / "best"
        if not checkpoint.is_dir():
            raise ValueError(f"Missing trained checkpoint: {checkpoint}. Train this experiment first.")
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        original_test = str(Path(config["arguments"]["split_dir"]) / "gold_test.csv")
        expected_hash = config["source_hashes"].get(original_test)
        if expected_hash != test_hash:
            raise ValueError(f"{experiment}: test CSV differs from the held-out split recorded during training.")
        runs[experiment] = (checkpoint, config)
    if args.limit:
        rows = rows[:args.limit]

    import torch

    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix=".evaluate-t5-", dir=args.output_dir.parent) as temp:
        staging = Path(temp)
        for experiment, (checkpoint, config) in runs.items():
            print(f"Evaluating {experiment} on {len(rows)} rows using {device}...", flush=True)
            metrics, details = evaluate_checkpoint(checkpoint, rows, args, device)
            metrics.update({
                "experiment": experiment, "checkpoint": str(checkpoint.resolve()),
                "test_csv": str(args.test_csv.resolve()), "test_csv_sha256": test_hash,
                "full_test_rows": full_count, "partial_evaluation": len(rows) < full_count,
                "training_smoke_check": config.get("smoke_check", False), "device": device,
                "generation": {"num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens, "do_sample": False},
                "reference_policy": "Case-sensitive exact match against any reference; corpus CER uses the lowest normalized edit-distance reference per row. Duplicate rows are scored separately.",
            })
            folder = staging / experiment
            folder.mkdir()
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(details[0]))
                writer.writeheader()
                writer.writerows(details)
            summaries.append(metrics)
            print(f"  Exact match: {metrics['test_exact_match']:.2%}; CER: {metrics['test_cer']:.4f}", flush=True)
        (staging / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        fields = ["experiment", "test_rows", "test_exact_match", "test_cer", "partial_evaluation", "training_smoke_check"]
        with (staging / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        args.output_dir.mkdir()
        for item in staging.iterdir():
            shutil.move(str(item), args.output_dir / item.name)
    print(f"Saved test comparison and per-model predictions to {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, csv.Error) as error:
        raise SystemExit(f"Evaluation failed: {error}") from error
