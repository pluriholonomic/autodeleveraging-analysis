#!/usr/bin/env bash
set -euo pipefail

# Robust script dir + project root resolution (works in bash/zsh/sh and via symlinks)
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

OUT_DIR="${ROOT_DIR}/out"
FIG_DIR="${OUT_DIR}/figures_corrections"

if command -v uv >/dev/null 2>&1; then
  uv run python "${ROOT_DIR}/scripts/visualize_corrections.py" \
    --out "${OUT_DIR}" \
    --fig-dir "${FIG_DIR}"
  uv run python "${ROOT_DIR}/scripts/generate_full_report.py" \
    --out "${OUT_DIR}" \
    --fig-dir "${FIG_DIR}"
else
  python3 "${ROOT_DIR}/scripts/visualize_corrections.py" \
    --out "${OUT_DIR}" \
    --fig-dir "${FIG_DIR}"
  python3 "${ROOT_DIR}/scripts/generate_full_report.py" \
    --out "${OUT_DIR}" \
    --fig-dir "${FIG_DIR}"
fi
