#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/DataRater/.venv/bin/python}"

export PYTHON_BIN
export EPOCHS="${EPOCHS:-20}"
export RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-20}"
export BANK_EPOCHS="${BANK_EPOCHS:-15,15,16,16,17,17,18,18}"
export SUITE_TAG="${SUITE_TAG:-epoch20_bank15_18}"

bash scripts/run_p8_inner_reset_suite.sh
