#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/handover}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$OUT_DIR/report_decomposition_handover_${STAMP}.tar.gz"

mkdir -p "$OUT_DIR"

tar -czf "$ARCHIVE" \
  --exclude-vcs \
  --exclude='fvlm' \
  --exclude='miniconda3' \
  --exclude='checkpoints' \
  --exclude='data/*.nii.gz' \
  --exclude='data/*.pth' \
  --exclude='output/inspect' \
  --exclude='output/tum' \
  --exclude='output/ct_rate/*.png' \
  --exclude='output/ct_rate/*.csv' \
  --exclude='output/ct_rate/*.txt' \
  -C "$PROJECT_ROOT" .

echo "Created: $ARCHIVE"
