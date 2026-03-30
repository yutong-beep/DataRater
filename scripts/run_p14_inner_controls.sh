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
  --score_grad_log_interval 100
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

# 1) PDZ inner soft penalty + cap on top of best no-PDZ outer
run_case \
  "p14 no-PDZ outer + PDZ inner soft penalty" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=0" \
  --inner_source_score_bias "PDZ_PBM=-0.75" \
  --inner_source_weight_cap "PDZ_PBM=0.35" \
  --output_dir experiments/p14_no_pdz_inner_pdz_penalty_seed43

# 2) Per-source spread regularization + source-bias penalty
run_case \
  "p14 no-PDZ outer + score regularization" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=0" \
  --score_within_source_std_floor 0.25 \
  --score_within_source_std_penalty_coef 0.05 \
  --score_source_bias_penalty_coef 0.05 \
  --output_dir experiments/p14_no_pdz_score_regularized_seed43

# 3) Higher temperature (more uniform softmax weighting)
run_case \
  "p14 no-PDZ outer + higher temperature" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=0" \
  --temperature 1.0 \
  --output_dir experiments/p14_no_pdz_high_temp_seed43

# 4) Sigmoid-normalized independent weighting
run_case \
  "p14 no-PDZ outer + sigmoid weighting" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=0" \
  --inner_weighting sigmoid_norm \
  --output_dir experiments/p14_no_pdz_sigmoid_weighting_seed43

# 5) Control: PDZ inner soft penalty + uniform outer
run_case \
  "p14 uniform outer + PDZ inner soft penalty" \
  --outer_sampling custom_ratio \
  --outer_source_weights "ATLAS=1,PDBbind v2020=1,SAbDab=1,SKEMPI v2.0=1,PDZ_PBM=1" \
  --inner_source_score_bias "PDZ_PBM=-0.75" \
  --inner_source_weight_cap "PDZ_PBM=0.35" \
  --output_dir experiments/p14_uniform_inner_pdz_penalty_seed43
