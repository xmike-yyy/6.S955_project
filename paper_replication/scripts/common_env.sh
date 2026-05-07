#!/usr/bin/env bash
set -euo pipefail

REPLICATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${REPLICATION_ROOT}/.." && pwd)"
PYTHON_BIN="${REPLICATION_ROOT}/.venv/bin/python"

export REPLICATION_ROOT
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/guitar_style_dataset/magcil-guitar_style_dataset-eb27d7b:${PYTHONPATH:-}"
export MPLCONFIGDIR="${REPLICATION_ROOT}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${REPLICATION_ROOT}/.cache/numba"
export XDG_CACHE_HOME="${REPLICATION_ROOT}/.cache/xdg"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "${REPLICATION_ROOT}/logs"
mkdir -p "${REPLICATION_ROOT}/results"
mkdir -p "${REPLICATION_ROOT}/work"
mkdir -p "${REPLICATION_ROOT}/pkl"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${NUMBA_CACHE_DIR}"
mkdir -p "${XDG_CACHE_HOME}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing venv Python at ${PYTHON_BIN}" >&2
  echo "Create it with: python3.11 -m venv ${REPLICATION_ROOT}/.venv" >&2
  exit 1
fi

