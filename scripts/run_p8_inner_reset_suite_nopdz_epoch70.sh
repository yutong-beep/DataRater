#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/DataRater/.venv/bin/python}"

export PYTHON_BIN
export EPOCHS="${EPOCHS:-70}"
export RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-70}"
export BANK_EPOCHS="${BANK_EPOCHS:-5,5,6,6,7,7,8,8}"
export EXCLUDE_SOURCES="${EXCLUDE_SOURCES:-PDZ_PBM}"
export SUITE_TAG="${SUITE_TAG:-nopdz_epoch70}"

bash scripts/run_p8_inner_reset_suite.sh
