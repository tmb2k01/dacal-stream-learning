#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

mkdir -p "${DATA_DIR}"
curl -o "${DATA_DIR}/SEA_training_class.csv" https://raw.githubusercontent.com/vlosing/driftDatasets/refs/heads/master/artificial/sea/SEA_training_class.csv
curl -o "${DATA_DIR}/SEA_training_data.csv" https://raw.githubusercontent.com/vlosing/driftDatasets/refs/heads/master/artificial/sea/SEA_training_data.csv
