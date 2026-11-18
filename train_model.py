#!/usr/bin/env python3
"""Train one prepared T5 experiment; select checkpoints using gold validation loss."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

EXPERIMENTS = ("char_gold", "char_gold_silver", "bpe_gold")


def read_gold(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"name", "romanize1"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires name and romanize1 columns.")
        rows = list(reader)
    if not rows or any(not (r.get("name") or "").strip() or
                       not (r.get("romanize1") or "").strip() for r in rows):
        raise ValueError(f"{path} is empty or has blank required fields.")
    return rows


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_silver(args, tokenizer, excluded, gold_pairs):
    """Reservoir sample eligible unique pairs without loading the full silver CSV."""
    rng = random.Random(args.seed)
    selected, seen = [], set(gold_pairs)
    counts = {"scanned": 0, "held_out": 0, "invalid_or_low_trust": 0,
              "duplicate_pair": 0, "unrepresentable_or_too_long": 0, "eligible": 0}
    with args.silver_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"name", "romanize", "trust score"}.issubset(reader.fieldnames or []):
            raise ValueError("Silver CSV requires name, romanize, and trust score columns.")
        for row in reader:
            counts["scanned"] += 1
            if counts["scanned"] % 500000 == 0:
                print(f"Silver: scanned {counts['scanned']:,} rows; {counts['eligible']:,} eligible unique pairs", flush=True)
            name, target = row.get("name") or "", row.get("romanize") or ""
            if name in excluded:
                counts["held_out"] += 1
                continue
            try:
                trust = float(row["trust score"])
            except (TypeError, ValueError):
                trust = math.nan
            if not name.strip() or not target.strip() or not math.isfinite(trust) or trust < args.silver_min_trust:
                counts["invalid_or_low_trust"] += 1
                continue
            pair = (name, target)
            if pair in seen:
                counts["duplicate_pair"] += 1
                continue
            seen.add(pair)
            encodings = [tokenizer.encode(text) for text in pair]
            if any(len(ids) > args.max_length or tokenizer.unk_token_id in ids for ids in encodings):
                counts["unrepresentable_or_too_long"] += 1
                continue
            counts["eligible"] += 1
            if len(selected) < args.silver_limit:
                selected.append(pair)
            else:
                index = rng.randrange(counts["eligible"])
                if index < args.silver_limit:
                    selected[index] = pair
    if not selected:
        raise ValueError("No silver examples passed the configured filters.")
    counts["selected"] = len(selected)
    return selected, counts


def encode_pairs(pairs, tokenizer, max_length):
    encoded = []
    for name, target in pairs:
        source_ids, target_ids = tokenizer.encode(name), tokenizer.encode(target)
        if max(len(source_ids), len(target_ids)) > max_length:
            raise ValueError("Gold example exceeds --max-length; increase it instead of truncating names.")
        encoded.append({"input_ids": source_ids, "labels": target_ids})
    return encoded


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared_models"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--silver-csv", type=Path, default=Path("contest3-2026/romanization-silver.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5, help="Epochs without validation-loss improvement; 0 disables.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--silver-limit", type=int, default=10000)
    parser.add_argument("--silver-min-trust", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads.")
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke check: limit gold training rows; 0 means all.")
    parser.add_argument("--eval-limit", type=int, default=0, help="Smoke check: limit validation rows; 0 means all.")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke check: total optimizer updates; 0 means unlimited.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and select data without training or writing outputs.")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "eval_batch_size", "max_length", "max_new_tokens", "num_beams", "threads", "silver_limit"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    for key in ("patience", "train_limit", "eval_limit", "max_steps"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} cannot be negative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not math.isfinite(args.silver_min_trust):
        parser.error("Weight decay must be nonnegative and finite; silver trust must be finite.")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Refusing to overwrite existing output: {args.output_dir}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, T5ForConditionalGeneration, set_seed

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_path, validation_path, test_path = [args.split_dir / f"gold_{split}.csv" for split in ("train", "validation", "test")]
    train_rows, validation_rows = read_gold(train_path), read_gold(validation_path)
    # Only test names are read, to exclude them from silver training. Test answers are never evaluated here.
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "name" not in (reader.fieldnames or []):
            raise ValueError("Local test CSV requires a name column.")
        test_names = {row["name"] for row in reader}
    train_names, validation_names = [{row["name"] for row in rows} for rows in (train_rows, validation_rows)]
    if train_names & validation_names or train_names & test_names or validation_names & test_names:
        raise ValueError("Gold splits contain overlapping Thai names.")
    tokenizer = AutoTokenizer.from_pretrained(args.prepared_dir / args.experiment, local_files_only=True)
    prep_manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    if prep_manifest["training_csv_sha256"] != file_hash(train_path):
        raise ValueError("Gold training CSV differs from the one used to prepare tokenizers/models.")
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    if args.eval_limit:
        validation_rows = validation_rows[:args.eval_limit]
    pairs = [(row["name"], row["romanize1"]) for row in train_rows]
    silver, silver_counts = [], None
    if args.experiment == "char_gold_silver":
        print("Scanning silver data, filtering held-out names, and sampling eligible pairs...", flush=True)
        silver, silver_counts = select_silver(args, tokenizer, validation_names | test_names, set(pairs))
    training = encode_pairs(pairs + silver, tokenizer, args.max_length)
    if any(tokenizer.unk_token_id in item[key] for item in training for key in ("input_ids", "labels")):
        raise ValueError("Training text contains tokens absent from the prepared vocabulary.")
    validation = encode_pairs([(r["name"], r["romanize1"]) for r in validation_rows], tokenizer, args.max_length)
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "device": device, "gold_training_rows": len(pairs), "silver_selection": silver_counts,
                "validation_rows": len(validation_rows), "checkpoint_selection": "minimum validation loss on romanize1",
                "source_hashes": {str(path): file_hash(path) for path in (train_path, validation_path, test_path)},
                "smoke_check": bool(args.train_limit or args.eval_limit or args.max_steps)}
    if silver:
        manifest["source_hashes"][str(args.silver_csv)] = file_hash(args.silver_csv)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    model = T5ForConditionalGeneration.from_pretrained(args.prepared_dir / args.experiment, local_files_only=True).to(device)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if silver:
        with (args.output_dir / "selected_silver.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "romanize"])
            writer.writerows(silver)

    def collate(items):
        batch = tokenizer.pad([{"input_ids": item["input_ids"]} for item in items], return_tensors="pt")
        labels = tokenizer.pad([{"input_ids": item["labels"]} for item in items], return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return {key: value.to(device) for key, value in batch.items()}

    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.eval_batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, stale, step = math.inf, 0, 0
    best_dir = args.output_dir / "best"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, train_tokens = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            tokens = (batch["labels"] != -100).sum().item()
            train_total += loss.item() * tokens
            train_tokens += tokens
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        val_total, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in validation_loader:
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite validation loss.")
                tokens = (batch["labels"] != -100).sum().item()
                val_total += loss.item() * tokens
                val_tokens += tokens
        val_loss = val_total / val_tokens
        record = {"epoch": epoch, "step": step, "train_loss": train_total / train_tokens, "validation_loss": val_loss}
        print(json.dumps(record), flush=True)
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if val_loss < best:
            best, stale = val_loss, 0
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            (args.output_dir / "best_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        else:
            stale += 1
        if (args.patience and stale >= args.patience) or (args.max_steps and step >= args.max_steps):
            break

    model = T5ForConditionalGeneration.from_pretrained(best_dir, local_files_only=True).to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(validation_rows), args.eval_batch_size):
            batch_rows = validation_rows[start:start + args.eval_batch_size]
            inputs = tokenizer([row["name"] for row in batch_rows], padding=True, return_tensors="pt").to(device)
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            predictions.extend(tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    exact, edits, reference_chars = 0, 0, 0
    with (args.output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "prediction", "references"])
        for row, prediction in zip(validation_rows, predictions):
            references = [row[key] for key in ("romanize1", "romanize2", "romanize3") if (row.get(key) or "").strip()]
            exact += prediction in references
            reference = min(references, key=lambda ref: edit_distance(prediction, ref) / len(ref))
            edits += edit_distance(prediction, reference)
            reference_chars += len(reference)
            writer.writerow([row["name"], prediction, json.dumps(references, ensure_ascii=False)])
    metrics = {"validation_loss": best, "validation_exact_match": exact / len(validation_rows),
               "validation_cer": edits / reference_chars, "validation_rows": len(validation_rows),
               "reference_policy": "case-sensitive exact match against any reference; CER uses lowest normalized edit-distance reference per row",
               "smoke_check": manifest["smoke_check"]}
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved best trained model to {best_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Training failed: {error}") from error
