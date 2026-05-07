#!/usr/bin/env bash
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
cd "${REPLICATION_ROOT}"

CONFIG="configs/experiment_2.yaml"
LOG="logs/experiment_2_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1

echo "Command: scripts/run_experiment_2.sh"
echo "Config: ${CONFIG}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"

status=0
"${PYTHON_BIN}" scripts/verify_inputs.py --config "${CONFIG}" || status=$?
"${PYTHON_BIN}" scripts/run_svm_original.py --config "${CONFIG}" || status=$?
"${PYTHON_BIN}" scripts/run_cnn_original.py --config "${CONFIG}" || status=$?
"${PYTHON_BIN}" scripts/collect_results.py || status=$?
"${PYTHON_BIN}" scripts/write_replication_report.py || status=$?

echo "Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit status: ${status}"
exit "${status}"

