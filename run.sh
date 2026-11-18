#!/usr/bin/env bash
# Run from any directory. Environment variables set defaults; CLI flags override them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

usage() {
    cat <<'EOF'
Usage: ./run.sh {char_gold|char_gold_silver|bpe_gold|all|evaluate|split|prepare} [Python CLI flags]

  char_gold         Train the small character T5 on gold only.
  char_gold_silver  Train the same character architecture on gold plus sampled silver.
  bpe_gold          Train the BPE T5 on gold only.
  all               Train all three sequentially (stops if one fails).
  evaluate          Evaluate trained best checkpoints on the local labeled test split.
  split             Create the 80:10:10 name-grouped split (refuses existing output).
  prepare           Initialize all three models (refuses existing output).

Defaults, configurable as environment variables:
  PYTHON=.venv/bin/python (if available, otherwise python3)
  PREPARED_DIR=prepared_models   SPLIT_DIR=data/splits   OUTPUT_ROOT=runs
  EPOCHS=30   BATCH_SIZE=32   EVAL_BATCH_SIZE=32   LR=0.001
  WEIGHT_DECAY=0.01   PATIENCE=5   DEVICE=auto   THREADS=4   SEED=42
  MAX_LENGTH=128   MAX_NEW_TOKENS=64   NUM_BEAMS=1
  SILVER_CSV=contest3-2026/romanization-silver.csv
  SILVER_LIMIT=10000   SILVER_MIN_TRUST=0.8
  EVAL_OUTPUT_DIR=evaluations/test (for evaluate; OUTPUT_ROOT selects trained runs)

Examples:
  ./run.sh char_gold
  EPOCHS=50 LR=0.0005 ./run.sh all
  SILVER_LIMIT=5000 SILVER_MIN_TRUST=0.9 ./run.sh char_gold_silver
  ./run.sh bpe_gold --batch-size 16 --output-dir runs/bpe_trial2
  ./run.sh char_gold --dry-run
  ./run.sh evaluate
  ./run.sh evaluate --experiments char_gold --output-dir evaluations/char_test
  ./run.sh char_gold --epochs 1 --train-limit 8 --eval-limit 4 --max-steps 1 --output-dir runs/smoke

Each training output directory must be new. best/ contains the selected model.
Extra flags after 'all' apply to all runs; use OUTPUT_ROOT to change their parent.
EOF
}

train() {
    local experiment="$1"
    shift
    "$PYTHON" train_model.py \
        --experiment "$experiment" \
        --prepared-dir "${PREPARED_DIR:-prepared_models}" \
        --split-dir "${SPLIT_DIR:-data/splits}" \
        --output-dir "${OUTPUT_ROOT:-runs}/$experiment" \
        --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-32}" \
        --eval-batch-size "${EVAL_BATCH_SIZE:-32}" --learning-rate "${LR:-0.001}" \
        --weight-decay "${WEIGHT_DECAY:-0.01}" --patience "${PATIENCE:-5}" \
        --device "${DEVICE:-auto}" --threads "${THREADS:-4}" --seed "${SEED:-42}" \
        --max-length "${MAX_LENGTH:-128}" --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
        --num-beams "${NUM_BEAMS:-1}" --silver-csv "${SILVER_CSV:-contest3-2026/romanization-silver.csv}" \
        --silver-limit "${SILVER_LIMIT:-10000}" --silver-min-trust "${SILVER_MIN_TRUST:-0.8}" "$@"
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
    char_gold|char_gold_silver|bpe_gold) train "$action" "$@" ;;
    all)
        # One shared --output-dir would collide across experiments.
        for argument in "$@"; do
            if [[ "$argument" == --output-dir || "$argument" == --output-dir=* || "$argument" == --experiment || "$argument" == --experiment=* ]]; then
                echo "For 'all', use OUTPUT_ROOT and keep experiment selection automatic." >&2
                exit 2
            fi
        done
        for experiment in char_gold char_gold_silver bpe_gold; do train "$experiment" "$@"; done
        ;;
    split) "$PYTHON" split_gold.py --output-dir "${SPLIT_DIR:-data/splits}" --seed "${SEED:-42}" "$@" ;;
    evaluate)
        "$PYTHON" evaluate_models.py --runs-dir "${OUTPUT_ROOT:-runs}" \
            --test-csv "${SPLIT_DIR:-data/splits}/gold_test.csv" \
            --output-dir "${EVAL_OUTPUT_DIR:-evaluations/test}" \
            --batch-size "${EVAL_BATCH_SIZE:-32}" --device "${DEVICE:-auto}" \
            --threads "${THREADS:-4}" --max-length "${MAX_LENGTH:-128}" \
            --max-new-tokens "${MAX_NEW_TOKENS:-64}" --num-beams "${NUM_BEAMS:-1}" "$@"
        ;;
    prepare) "$PYTHON" prepare_models.py --train-csv "${SPLIT_DIR:-data/splits}/gold_train.csv" --output-dir "${PREPARED_DIR:-prepared_models}" "$@" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
