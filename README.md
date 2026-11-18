# Name romanization: prepare small T5 models

`prepare_models.py` creates three **randomly initialized, untrained** T5 model folders.
It does not download pretrained weights, split data, select silver rows, or train models.
Train these models before evaluating romanization performance.

| Folder               | Tokenization                                      | Intended training data         |
| -------------------- | ------------------------------------------------- | ------------------------------ |
| `char_gold`        | One Unicode code point per token                  | Gold training data             |
| `char_gold_silver` | Identical character tokenizer and initial weights | Gold plus selected silver data |
| `bpe_gold`         | BPE, target vocabulary 512                        | Gold training data             |

Each model has 2 encoder layers, 2 decoder layers, hidden size 128, 4 attention heads,
key/value size 32, feed-forward size 512, and dropout 0.1. Initialization uses seed 42.
These are small starting architectures, not tuned results. Parameter counts differ
between character and BPE models because their embedding vocabularies differ.

## Install and prepare

The repository contains the code and instructions. Course datasets, assignment PDFs,
generated splits, model weights, and run outputs stay local. Obtain the assignment
datasets separately and place these files in `contest3-2026/`:

- `romanization-train.csv`
- `romanization-silver.csv`
- `romanization-test-no-answer.csv`

The split and model preparation commands below recreate the generated assets.

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create the selected **80:10:10 train/validation/local-test split** with seed 42:

```bash
python split_gold.py
```

This groups rows by exact Thai name, so duplicate names and all their romanizations
stay together. Ratios apply to unique names; row ratios may differ slightly. Every
column and row is preserved. Files are saved to `data/splits/`, along with a manifest
of counts and hashes. Existing output directories are rejected. The provided
unlabeled assignment test file is not modified. Use the same split for all experiments.

Keep validation/test names out of tokenizer fitting. The preparation script requires
a CSV with `name` and `romanize1` columns;
it uses exactly those columns and preserves their text, including case and spaces.
Additional columns are ignored. Blank values are rejected. It neither splits the
provided assignment CSV nor determines whether your input has already been split.

```bash
python prepare_models.py --train-csv data/splits/gold_train.csv --output-dir prepared_models
```

The preparation output directory must
not already exist. The script fits tokenizers only on the supplied source names and
first romanizations. BPE uses minimum merge frequency 2; its actual vocabulary can
be smaller than 512. An alphabet too large for that vocabulary budget is rejected.
Characters absent from the training alphabet map to `<unk>` in both tokenizers.
Thai combining marks are separate character tokens; normalization is disabled.

All folders use `<pad>`, `</s>`, and `<unk>`. Encoding appends `</s>`; decoding with
`skip_special_tokens=True` removes it. The decoder starts with `<pad>`.

The script verifies exact tokenization round trips for every training string,
reloads every saved model/tokenizer, runs a finite-loss forward pass, and checks
that both character folders are byte-for-byte identical. It saves `manifest.json`
with the input hash, row count, seed, parameter counts, package versions, and checks.
The forward-pass loss is a technical check on random weights, not a performance score.

## Load a prepared model

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

folder = "prepared_models/char_gold"
tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True)

inputs = tokenizer(["กุลสตรี"], return_tensors="pt", padding=True)
labels = tokenizer(["kunlasattri"], return_tensors="pt", padding=True)["input_ids"]
labels[labels == tokenizer.pad_token_id] = -100
loss = model(**inputs, labels=labels).loss
```

Use `char_gold_silver` for your later gold-plus-silver experiment. Reusing the same
character initialization and tokenizer helps isolate the effect of extra training
data. The training commands below select silver examples while excluding held-out
evaluation names.

## Train the models

After installing dependencies, run one experiment or all three sequentially:

```bash
./run.sh char_gold
./run.sh char_gold_silver
./run.sh bpe_gold

# Alternative: run all three with the same settings.
./run.sh all
```

`run.sh` uses `.venv/bin/python` if available, otherwise `python3`. Set `PYTHON` to
use another Python environment. It works from any current directory. For help:

```bash
./run.sh --help
./run.sh char_gold --help
```

Defaults are 30 epochs, batch size 32, learning rate 0.001, AdamW with weight decay
0.01, gradient norm clipping at 1.0, and early stopping after 5 epochs without a
validation-loss improvement. The learning rate stays constant. CPU, Apple MPS, or
CUDA is selected automatically; training uses full precision. These are configurable
starting settings, not tuned hyperparameters or demonstrated performance results.

Configure environment variables or append Python flags (flags take precedence):

```bash
EPOCHS=50 BATCH_SIZE=16 LR=0.0005 OUTPUT_ROOT=runs/trial2 ./run.sh all
SILVER_LIMIT=5000 SILVER_MIN_TRUST=0.9 ./run.sh char_gold_silver
./run.sh bpe_gold --device cpu --patience 8 --output-dir runs/bpe_trial2

# Validate inputs without training or saving anything.
./run.sh char_gold --dry-run

# Brief pipeline check; its results are NOT model performance measurements.
./run.sh char_gold --epochs 1 --train-limit 8 --eval-limit 4 --max-steps 1 --output-dir runs/smoke
```

Every run needs a new output directory; use `OUTPUT_ROOT` for a new group of runs or
`--output-dir` for an individual experiment. Runs start from the prepared initialization;
there is no automatic resume. `all` stops if one experiment fails. Use `./run.sh split`
or `./run.sh prepare` only when those outputs do not already exist.

### Data use and silver selection

All experiments train on `romanize1` for each gold training row. Validation loss also
uses `romanize1`. The script verifies disjoint gold names and checks that the training
CSV matches the tokenizer-preparation input hash. Gold names are not silently truncated;
increase `MAX_LENGTH` (default 128 tokens including EOS) if needed.

By default, `char_gold_silver` samples up to **10,000 distinct silver pairs with trust
score at least 0.8**, using seed 42. This threshold is a heuristic, not a guarantee
of correct labels. The script scans the entire silver CSV and uses reservoir sampling
to give eligible distinct pairs equal selection probability. It excludes every exact
Thai name from the full gold validation and local test splits, blank/invalid rows,
duplicate pairs (including gold training pairs), and examples that exceed the token
limit or contain characters absent from the frozen tokenizer. Gold and selected silver
pairs are mixed and shuffled together each epoch, without separate weighting.

All three experiments retain their prepared tokenizers. The silver experiment does
not fit a new tokenizer on silver data. Filtering uses exact names without normalization,
matching the split policy. The local test CSV is read only for name exclusion; local
test answers and the unlabeled assignment test file are not used for model selection.

### Saved outputs

Each `runs/<experiment>/` directory contains:

- `best/`: model and tokenizer selected by lowest gold validation loss.
- `run_config.json`: settings, data hashes, actual data counts, and silver filter counts.
- `history.jsonl` and `best_checkpoint.json`: epoch losses and selected epoch.
- `validation_metrics.json` and `validation_predictions.csv`: predictions from the best
  checkpoint, exact match against any available reference, and character error rate.
- `selected_silver.csv`: exact sampled training pairs, for the silver experiment only.

Validation generation uses greedy decoding by default (`NUM_BEAMS=1`), with at most
64 new tokens (`MAX_NEW_TOKENS`). Metrics preserve case. Exact match accepts any
nonblank gold romanization. For CER, each prediction is compared against the reference
with the lowest normalized edit distance; edit counts and reference lengths are then
summed across rows. Duplicate gold rows are evaluated separately. Validation results
guide development; the held-out local test split remains available for final evaluation.

The training script uses T5's automatic decoder-input shifting and ignores padded
labels with `-100`, following the [T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5).

## Evaluate the local test split

After training, evaluate all three best checkpoints:

```bash
./run.sh evaluate
# Equivalent Python command:
.venv/bin/python evaluate_models.py

# Evaluate a single model or checkpoints under another runs directory:
./run.sh evaluate --experiments char_gold --output-dir evaluations/char_test
.venv/bin/python evaluate_models.py --runs-dir runs/trial2 --output-dir evaluations/trial2
```

The default input is `data/splits/gold_test.csv`, the labeled local test split.
The script checks its file hash against each training run's recorded held-out split.
The unlabeled assignment test file cannot be scored by this script. It loads
`runs/<experiment>/best/` with its saved tokenizer, runs inference without modifying
weights, and scores all available romanizations using the same exact-match and CER
policy as validation. Scores are fractions (exact match 0–1; CER can exceed 1).
Unknown source tokens are counted in each model's metrics; source names are never
silently truncated. Default decoding is greedy with a 64-new-token limit. Use the
generation settings chosen during validation for the final test comparison.

Outputs in `evaluations/test/` include `comparison.csv`, `comparison.json`, and a
`metrics.json` plus `predictions.csv` for each model. Predictions preserve the test
row order and include references, exact-match flags, edit counts, and per-row CER.
Existing output directories are rejected. Batch size, device, token limits, and beam
count are configurable through flags or the corresponding `run.sh` environment variables.
Run `./run.sh evaluate --help` for all options. `--limit N` supports a brief pipeline
check, marked as a partial evaluation; checkpoints from training smoke checks are
also labeled in the results. Test scores are for final comparison, not early stopping.
