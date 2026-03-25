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
  --T_window 8
  --T_backprop 2
  --n_inner_models 4
  --N_ref 22304
  --lr 1e-4
  --use_zscore_inner
  --meta_grad_clip 1.0
  --canary_interval 200
  --random_baseline
  --random_mode stratified_ratio
  --datarater_arch single
  --inner_batch_scope per_inner
  --outer_batch_scope per_inner
  --seed 43
  --random_seed 43
)

run_case() {
  local name="$1"
  shift
  echo "=================================================="
  echo "RUNNING: ${name}"
  echo "TIME: $(date)"
  echo "=================================================="
  "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" "$@"
}

run_case \
  "p11 strongest random outer" \
  --outer_sampling random \
  --output_dir experiments/p11_strongest_seed43

run_case \
  "p11 balanced all-source outer" \
  --outer_sampling balanced \
  --output_dir experiments/p11_outer_balanced_seed43

run_case \
  "p11 dataset-ratio outer" \
  --outer_sampling dataset_ratio \
  --output_dir experiments/p11_outer_dataset_ratio_seed43

run_case \
  "p11 no-PDZ outer" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=0" \
  --output_dir experiments/p11_outer_no_pdz_seed43
