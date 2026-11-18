#!/usr/bin/env python3
"""Prepare untrained T5 experiments using only an already-split gold training CSV."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"]
EXPERIMENTS = {
    "char_gold": "Character T5; intended for gold-only training.",
    "char_gold_silver": "Identical character T5; intended for gold plus selected silver training.",
    "bpe_gold": "BPE T5; intended for gold-only training.",
}


def read_training_texts(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if not {"name", "romanize1"}.issubset(reader.fieldnames or []):
        raise ValueError("Training CSV must contain name and romanize1 columns.")
    texts = []
    rows = 0
    for line, row in enumerate(reader, start=2):
        for column in ("name", "romanize1"):
            value = row.get(column)
            if value is None or not value.strip():
                raise ValueError(f"Missing or blank {column} on CSV line {line}.")
            texts.append(value)
        rows += 1
    if not rows:
        raise ValueError("Training CSV has no data rows.")
    return texts, rows, hashlib.sha256(raw).hexdigest()


def make_tokenizer(texts, kind):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(set("".join(texts)))
    if any(token in text for token in SPECIAL_TOKENS for text in texts):
        raise ValueError("Training text contains a reserved special-token string.")
    if kind == "char":
        vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS + alphabet)}
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Split(Regex(""), behavior="isolated")
    else:
        if len(alphabet) + len(SPECIAL_TOKENS) > 512:
            raise ValueError("Training alphabet exceeds the 512-token BPE vocabulary budget.")
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.train_from_iterator(
            texts,
            trainer=trainers.BpeTrainer(
                vocab_size=512, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                initial_alphabet=alphabet, show_progress=False,
            ),
        )
    backend.decoder = decoders.Fuse()
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", backend.token_to_id("</s>"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )


def verify_tokenizer(tokenizer, texts, character_level):
    # Check every training string, including Thai combining marks and any spaces.
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.unk_token_id in ids:
            raise RuntimeError("Training text produced an unknown token.")
        if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
            raise RuntimeError("Tokenizer failed exact text round trip.")
        if character_level and tokenizer.convert_ids_to_tokens(ids) != list(text):
            raise RuntimeError("Character tokenizer must emit one token per Unicode code point.")
        if tokenizer.encode(text)[-1] != tokenizer.eos_token_id:
            raise RuntimeError("Tokenizer failed to append end-of-sequence token.")


def save_model(folder, tokenizer):
    from transformers import T5Config, T5ForConditionalGeneration, set_seed

    set_seed(42)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(tokenizer), d_model=128, d_kv=32, d_ff=512,
        num_layers=2, num_decoder_layers=2, num_heads=4, dropout_rate=0.1,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    ))
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    return sum(parameter.numel() for parameter in model.parameters())


def verify_saved_models(root, texts):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    results = {}
    for name in EXPERIMENTS:
        folder = root / name
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        verify_tokenizer(tokenizer, texts, character_level=name.startswith("char_"))
        model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval()
        inputs = tokenizer([texts[0]], return_tensors="pt")
        labels = tokenizer([texts[1]], return_tensors="pt")["input_ids"]
        with torch.no_grad():
            loss = model(**inputs, labels=labels).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite forward-pass loss for {name}.")
        results[name] = {"reload_and_round_trip": "passed", "forward_loss": loss.item()}
    # Both folders are copies of the same serialization, so compare all assets byte-for-byte.
    first, second = root / "char_gold", root / "char_gold_silver"
    for file in first.iterdir():
        if file.read_bytes() != (second / file.name).read_bytes():
            raise RuntimeError(f"Character initialization differs: {file.name}")
    results["character_assets_identical"] = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path,
                        help="Gold training subset only; requires name and romanize1 columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_models"),
                        help="New output directory (default: prepared_models).")
    args = parser.parse_args()
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"Refusing to overwrite existing output: {output}")
    try:
        texts, rows, digest = read_training_texts(args.train_csv)
        char_tokenizer = make_tokenizer(texts, "char")
        bpe_tokenizer = make_tokenizer(texts, "bpe")
        for tokenizer, is_char in ((char_tokenizer, True), (bpe_tokenizer, False)):
            verify_tokenizer(tokenizer, texts, is_char)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Finish verification in a temporary sibling directory before publishing outputs.
        with tempfile.TemporaryDirectory(prefix=".prepare-t5-", dir=output.parent) as temporary:
            staging = Path(temporary)
            char_count = save_model(staging / "char_gold", char_tokenizer)
            shutil.copytree(staging / "char_gold", staging / "char_gold_silver")
            bpe_count = save_model(staging / "bpe_gold", bpe_tokenizer)
            checks = verify_saved_models(staging, texts)
            import importlib.metadata
            manifest = {
                "status": "randomly_initialized_untrained", "seed": 42,
                "training_csv": str(args.train_csv.resolve()), "training_csv_sha256": digest,
                "training_rows": rows, "target_column": "romanize1",
                "tokenizer_fit_data": "gold training subset only; source and target strings",
                "normalization": "none; preserve case, spaces, and Unicode code points",
                "bpe_target_vocab_size": 512, "bpe_min_frequency": 2,
                "experiments": {
                    name: {"description": description,
                           "parameter_count": bpe_count if name == "bpe_gold" else char_count,
                           "vocab_size": len(bpe_tokenizer if name == "bpe_gold" else char_tokenizer)}
                    for name, description in EXPERIMENTS.items()
                },
                "package_versions": {name: importlib.metadata.version(name)
                                     for name in ("torch", "transformers", "tokenizers", "safetensors")},
                "verification": checks,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            # mkdir is exclusive: another process cannot cause existing outputs to be overwritten.
            output.mkdir()
            for item in staging.iterdir():
                shutil.move(str(item), output / item.name)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Prepared and verified three UNTRAINED T5 experiments in {output}")
    print("The two character models have identical initial weights and tokenizers.")
    print("See manifest.json for configuration provenance and verification results.")


if __name__ == "__main__":
    main()
