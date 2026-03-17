#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/DataRater/.venv/bin/python}"

DATASET="${DATASET:-Bindwell/PPBA}"
DATA_MODE="${DATA_MODE:-all}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
META_BATCH_SIZE="${META_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-10}"
META_STEPS="${META_STEPS:-1500}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-10}"
B_VAL="${B_VAL:-32}"
TEMPERATURE="${TEMPERATURE:-0.5}"
KEEP_RATIO="${KEEP_RATIO:-0.95}"
OUTER_OBJECTIVE="${OUTER_OBJECTIVE:-pearson}"
T_WINDOW="${T_WINDOW:-8}"
T_BACKPROP="${T_BACKPROP:-8}"
N_INNER_MODELS="${N_INNER_MODELS:-8}"
LIFETIME="${LIFETIME:-2000}"
BANK_BASE_SEED="${BANK_BASE_SEED:-1000}"
BANK_EPOCHS="${BANK_EPOCHS:-5,5,6,6,7,7,8,8}"
BANK_JITTER="${BANK_JITTER:-1}"
RANDOM_MODE="${RANDOM_MODE:-matched_source_counts}"
SUITE_TAG="${SUITE_TAG:-}"

sanitize_tag() {
  local raw="$1"
  raw="${raw//,/x}"
  raw="${raw// /}"
  raw="${raw//[^A-Za-z0-9._-]/_}"
  printf "%s" "$raw"
}

BANK_TAG="$(sanitize_tag "$BANK_EPOCHS")"
EXTRA_TAG=""
if [[ -n "$SUITE_TAG" ]]; then
  EXTRA_TAG="_$(sanitize_tag "$SUITE_TAG")"
fi

COMMON_ARGS=(
  --dataset "$DATASET"
  --data_mode "$DATA_MODE"
  --train_ratio "$TRAIN_RATIO"
  --max_length "$MAX_LENGTH"
  --epochs "$EPOCHS"
  --meta_steps "$META_STEPS"
  --retrain_epochs "$RETRAIN_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --meta_batch_size "$META_BATCH_SIZE"
  --B "$B_VAL"
  --temperature "$TEMPERATURE"
  --keep_ratio "$KEEP_RATIO"
  --outer_objective "$OUTER_OBJECTIVE"
  --datarater_arch single
  --outer_sampling random
  --T_window "$T_WINDOW"
  --T_backprop "$T_BACKPROP"
  --n_inner_models "$N_INNER_MODELS"
  --lifetime "$LIFETIME"
  --seed "$SEED"
  --random_baseline
  --random_mode "$RANDOM_MODE"
)

BANK_OUTPUT_ROOT="experiments/p8_inner_bank_mix_ep${EPOCHS}_bank${BANK_TAG}_seed${SEED}${EXTRA_TAG}"

echo "==> Building mixed-epoch inner-init bank"
"$PYTHON_BIN" scripts/build_inner_init_bank.py \
  --dataset "$DATASET" \
  --data_mode "$DATA_MODE" \
  --train_ratio "$TRAIN_RATIO" \
  --seed "$SEED" \
  --bank_base_seed "$BANK_BASE_SEED" \
  --bank_epochs "$BANK_EPOCHS" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$BANK_OUTPUT_ROOT"

BANK_DIR="$(ls -td "$BANK_OUTPUT_ROOT"/inner_init_bank_* | head -n 1)"
echo "==> Using bank dir: $BANK_DIR"

run_one() {
  local name="$1"
  shift
  local out_root="experiments/p8_single_${name}_ep${EPOCHS}_rep${RETRAIN_EPOCHS}_bs${BATCH_SIZE}_metabs${META_BATCH_SIZE}_tau${TEMPERATURE}_B${B_VAL}_outer-${OUTER_OBJECTIVE}_T${T_WINDOW}_Tb${T_BACKPROP}_n${N_INNER_MODELS}_ms${META_STEPS}_seed${SEED}${EXTRA_TAG}"
  echo "==> Running $name"
  "$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" --output_dir "$out_root" "$@"
}

run_one "randominit" --inner_reset_strategy random_init
run_one "carryover" --inner_reset_strategy carryover
run_one "bankmix" \
  --inner_reset_strategy checkpoint_bank \
  --inner_init_bank_dir "$BANK_DIR" \
  --inner_init_bank_jitter "$BANK_JITTER"

echo "==> Suite complete"
