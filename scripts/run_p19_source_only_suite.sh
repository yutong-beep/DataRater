#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-43}"

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
  --seed "${SEED}"
  --random_seed "${SEED}"
)

run_one() {
  local name="$1"
  local weights="$2"
  local outdir="$3"
  echo "==== ${name} @ $(date -u) ===="
  "${PYTHON_BIN}" main.py \
    "${COMMON_ARGS[@]}" \
    --outer_source_weights "${weights}" \
    --output_dir "${outdir}"
}

run_one \
  "PDZ-only outer" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=1" \
  "experiments/p19_outer_pdz_only_seed${SEED}"

run_one \
  "SKEMPI-only outer" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=1,PDZ_PBM=0" \
  "experiments/p19_outer_skempi_only_seed${SEED}"

run_one \
  "PDBbind-only outer" \
  "ATLAS=0,PDBbind v2020=1,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p19_outer_pdbbind_only_seed${SEED}"

run_one \
  "SAbDab-only outer" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=1,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p19_outer_sabdab_only_seed${SEED}"

run_one \
  "ATLAS-only outer" \
  "ATLAS=1,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p19_outer_atlas_only_seed${SEED}"
