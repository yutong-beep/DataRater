#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "===== follow-up 0: no-PDZ hellcamp ====="
PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_p11_no_pdz_hellcamp.sh 2>&1 | tee experiments/_launch_logs/dr_p11_no_pdz_hellcamp_01.log

echo "===== follow-up 1: no-PDZ multi-seed ====="
PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_p11_no_pdz_multiseed.sh 2>&1 | tee experiments/_launch_logs/dr_p11_no_pdz_multiseed_01.log

echo "===== follow-up 2: no-PDZ strict deterministic ====="
PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_p11_no_pdz_strict_det.sh 43 2>&1 | tee experiments/_launch_logs/dr_p11_no_pdz_strictdet_01.log
