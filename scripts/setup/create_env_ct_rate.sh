#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
conda create -n ct-rate -y python=3.10
conda install -n ct-rate -y --file envs/ct-rate.explicit.txt || true
conda run -n ct-rate pip install -r envs/ct-rate.pip.txt || true
