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
  local keep_source="$2"
  local weights="$3"
  local outdir="$4"
  local exclude_sources=""

  case "${keep_source}" in
    "PDZ_PBM")
      exclude_sources="ATLAS,PDBbind v2020,SAbDab,SKEMPI v2.0"
      ;;
    "SKEMPI v2.0")
      exclude_sources="ATLAS,PDBbind v2020,SAbDab,PDZ_PBM"
      ;;
    "PDBbind v2020")
      exclude_sources="ATLAS,SAbDab,SKEMPI v2.0,PDZ_PBM"
      ;;
    "SAbDab")
      exclude_sources="ATLAS,PDBbind v2020,SKEMPI v2.0,PDZ_PBM"
      ;;
    "ATLAS")
      exclude_sources="PDBbind v2020,SAbDab,SKEMPI v2.0,PDZ_PBM"
      ;;
    *)
      echo "Unknown keep_source: ${keep_source}" >&2
      exit 1
      ;;
  esac

  echo "==== ${name} @ $(date -u) ===="
  "${PYTHON_BIN}" main.py \
    "${COMMON_ARGS[@]}" \
    --exclude_sources "${exclude_sources}" \
    --outer_source_weights "${weights}" \
    --output_dir "${outdir}"
}

run_one \
  "PDZ-only full pipeline" \
  "PDZ_PBM" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=1" \
  "experiments/p21_full_pdz_only_seed${SEED}"

run_one \
  "SKEMPI-only full pipeline" \
  "SKEMPI v2.0" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=1,PDZ_PBM=0" \
  "experiments/p21_full_skempi_only_seed${SEED}"

run_one \
  "PDBbind-only full pipeline" \
  "PDBbind v2020" \
  "ATLAS=0,PDBbind v2020=1,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p21_full_pdbbind_only_seed${SEED}"

run_one \
  "SAbDab-only full pipeline" \
  "SAbDab" \
  "ATLAS=0,PDBbind v2020=0,SAbDab=1,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p21_full_sabdab_only_seed${SEED}"

run_one \
  "ATLAS-only full pipeline" \
  "ATLAS" \
  "ATLAS=1,PDBbind v2020=0,SAbDab=0,SKEMPI v2.0=0,PDZ_PBM=0" \
  "experiments/p21_full_atlas_only_seed${SEED}"
