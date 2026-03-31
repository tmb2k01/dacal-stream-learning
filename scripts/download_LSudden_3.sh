#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

mkdir -p "${DATA_DIR}"
curl -o "${DATA_DIR}/LSudden_3.csv" https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/refs/heads/main/Datasets_in_CADM%2B/LSudden_3.csv
