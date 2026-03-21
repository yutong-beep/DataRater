#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON_ARGS=(
  --phase 1,2,3,4,5
  --data_mode combined_train
  --epochs 10
  --meta_steps 2500
  --retrain_epochs 10
  --batch_size 32
  --meta_batch_size 24
  --B 32
  --temperature 0.5
  --keep_ratio 0.95
  --outer_objective source_stratified_mse
  --datarater_arch single
  --outer_sampling random
  --T_window 8
  --T_backprop 2
  --n_inner_models 4
  --N_ref 22304
  --lr 1e-4
  --random_baseline
  --random_mode stratified_ratio
  --strict_deterministic
  --num_workers 2
)

run_one() {
  local seed="$1"
  local out_root="experiments/p9_historical_source_stratified_det_seed${seed}"
  echo "== running seed ${seed} =="
  "${PYTHON_BIN}" main.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --random_seed "${seed}" \
    --output_dir "${out_root}"
}

run_one 42
run_one 43
run_one 44
