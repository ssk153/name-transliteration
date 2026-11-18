# Thai Name Romanization

Convert Thai names into Roman spellings using small T5 models. This project compares
character-level and BPE tokenization, and the effect of adding silver training data.

## Models and results

Results on the local test split: 328 rows containing 324 unique names.

| Model | Training data | Tokenizer | Accuracy | CER ↓ |
| --- | --- | --- | --- | --- |
| `char_gold` | 2,614 gold rows | Character-level, 89 tokens | 44.82% | 18.07% |
| `char_gold_silver` | Gold + 10,000 silver pairs | Same character tokenizer | **82.62%** | **4.22%** |
| `bpe_gold` | 2,614 gold rows | BPE, 512 tokens | 42.07% | 62.38% |

All models are trained from scratch with **2 encoder layers, 2 decoder layers,
4 attention heads, hidden size 128, feed-forward size 512, and dropout 0.1**.
Tokenizers are fitted on the gold training split only.

Accuracy measures exact matches against any accepted romanization. CER measures
character edits relative to reference length; lower is better.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the course datasets in `contest3-2026/`:

- `romanization-train.csv`
- `romanization-silver.csv`
- `romanization-test-no-answer.csv`

Datasets and generated model files are kept locally and are not included in this repo.

## Prepare and train

```bash
./run.sh split       # Create the 80:10:10 train/validation/test split
./run.sh prepare     # Prepare models and tokenizers
./run.sh all         # Train all three models
```

Duplicate Thai names stay in the same split. The split uses seed 42.

To train one model:

```bash
./run.sh char_gold
./run.sh char_gold_silver
./run.sh bpe_gold
```

Defaults: **30 epochs, batch size 32, learning rate 0.001, AdamW**, and early stopping
with patience 5. The best checkpoint is selected by validation loss.

The silver experiment samples up to 10,000 distinct pairs with trust score ≥ 0.8,
excluding validation/test names and pairs outside the tokenizer vocabulary or length limit.
Gold training uses the first romanization for each row.

Change settings through environment variables or command-line flags:

```bash
EPOCHS=50 LR=0.0005 OUTPUT_ROOT=runs/trial2 ./run.sh all
SILVER_LIMIT=5000 SILVER_MIN_TRUST=0.9 ./run.sh char_gold_silver
./run.sh bpe_gold --batch-size 16 --output-dir runs/bpe_trial2
```

Use a new output directory for each run. See `./run.sh --help` for available settings.

## Evaluate

```bash
./run.sh evaluate

# Evaluate one model
./run.sh evaluate --experiments char_gold --output-dir evaluations/char_test
```

Evaluation uses the labeled local test split in `data/splits/gold_test.csv`.

- `runs/<model>/best/`: best model and tokenizer.
- `runs/<model>/`: training history and validation results.
- `evaluations/test/`: test comparison, metrics, and per-name predictions.
