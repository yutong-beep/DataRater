#!/usr/bin/env bash
set -euo pipefail

# Sequential helper to search teacher formulas for all three signals from a
# single existing audit run.
#
# Usage:
#   PYTHON_BIN=/home/ubuntu/DataRater/.venv/bin/python \
#   AUDIT_RUN_DIR=experiments/.../p4_mixflow_single_randomouter_YYYYMMDD_HHMMSS \
#   OUT_ROOT=experiments/p7_teacher_weight_search_suite_seed43 \
#   bash scripts/run_teacher_search_suite.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_RUN_DIR="${AUDIT_RUN_DIR:-}"
OUT_ROOT="${OUT_ROOT:-experiments/p7_teacher_weight_search_suite}"
SEED="${SEED:-43}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-8}"
TEACHER_ARCH="${TEACHER_ARCH:-multihead}"

if [[ -z "${AUDIT_RUN_DIR}" ]]; then
  echo "AUDIT_RUN_DIR is required" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "== teacher weight search: badness =="
"${PYTHON_BIN}" scripts/teacher_weight_search.py \
  --source_audit_run_dir "${AUDIT_RUN_DIR}" \
  --output_dir "${OUT_ROOT}/badness" \
  --signal badness \
  --seed "${SEED}" \
  --num_candidates "${NUM_CANDIDATES}" \
  --teacher_epochs "${TEACHER_EPOCHS}" \
  --teacher_arch "${TEACHER_ARCH}"

echo "== teacher weight search: noise_risk =="
"${PYTHON_BIN}" scripts/teacher_weight_search.py \
  --source_audit_run_dir "${AUDIT_RUN_DIR}" \
  --output_dir "${OUT_ROOT}/noise_risk" \
  --signal noise_risk \
  --seed "${SEED}" \
  --num_candidates "${NUM_CANDIDATES}" \
  --teacher_epochs "${TEACHER_EPOCHS}" \
  --teacher_arch "${TEACHER_ARCH}"

echo "== teacher weight search: ambiguity =="
"${PYTHON_BIN}" scripts/teacher_weight_search.py \
  --source_audit_run_dir "${AUDIT_RUN_DIR}" \
  --output_dir "${OUT_ROOT}/ambiguity" \
  --signal ambiguity \
  --seed "${SEED}" \
  --num_candidates "${NUM_CANDIDATES}" \
  --teacher_epochs "${TEACHER_EPOCHS}" \
  --teacher_arch "${TEACHER_ARCH}"

echo "Search suite complete -> ${OUT_ROOT}"
