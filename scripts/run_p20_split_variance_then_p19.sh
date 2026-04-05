#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
P19_SEED="${P19_SEED:-43}"

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
  --outer_sampling custom_ratio
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=1"
)

run_one() {
  local name="$1"
  shift
  echo "==== ${name} @ $(date -u) ===="
  "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" "$@"
}

run_one \
  "ratio 0.6 / seed42" \
  --train_ratio 0.6 \
  --seed 42 \
  --random_seed 42 \
  --output_dir experiments/p20_ratio0p6_seed42

run_one \
  "ratio 0.6 / seed43" \
  --train_ratio 0.6 \
  --seed 43 \
  --random_seed 43 \
  --output_dir experiments/p20_ratio0p6_seed43

run_one \
  "ratio 0.5 / seed42" \
  --train_ratio 0.5 \
  --seed 42 \
  --random_seed 42 \
  --output_dir experiments/p20_ratio0p5_seed42

run_one \
  "ratio 0.5 / seed43" \
  --train_ratio 0.5 \
  --seed 43 \
  --random_seed 43 \
  --output_dir experiments/p20_ratio0p5_seed43

run_one \
  "kfold4 / seed42" \
  --train_ratio 0.8 \
  --phase2_kfold 4 \
  --seed 42 \
  --random_seed 42 \
  --output_dir experiments/p20_kfold4_seed42

run_one \
  "kfold4 / seed43" \
  --train_ratio 0.8 \
  --phase2_kfold 4 \
  --seed 43 \
  --random_seed 43 \
  --output_dir experiments/p20_kfold4_seed43

echo "==== source-only suite @ $(date -u) ===="
PYTHON_BIN="${PYTHON_BIN}" SEED="${P19_SEED}" bash scripts/run_p19_source_only_suite.sh
