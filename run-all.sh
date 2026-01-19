#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
cd "${SCRIPT_DIR}"

# Use non-interactive matplotlib backend (avoids tkinter dependency)
export MPLBACKEND=Agg

# Parse arguments
FULL_MODE=0
if [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=1
fi

if [[ "${FULL_MODE}" -eq 1 ]]; then
    echo "=== ADL Paper FULL Reproduction (from raw data) ==="
    TOTAL_STEPS=8
else
    echo "=== ADL Paper Figure Reproduction ==="
    TOTAL_STEPS=5
fi
echo ""

# Step 1: Check for uv
if ! command -v uv &>/dev/null; then
    echo "[1/${TOTAL_STEPS}] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/${TOTAL_STEPS}] uv already installed"
fi

# Step 2: Sync Python environment (including dev deps for pytest)
echo "[2/${TOTAL_STEPS}] Setting up Python environment..."
uv sync --dev

# Full mode: regenerate enriched CSV and analysis data from raw inputs
if [[ "${FULL_MODE}" -eq 1 ]]; then
    # Step 3: Clone external dependencies (HyperReplay, HyperMultiAssetedADL)
    echo "[3/${TOTAL_STEPS}] Cloning external dependencies..."
    ./scripts/clone_deps.sh

    export HYPERREPLAY_DIR="${SCRIPT_DIR}/external/HyperReplay"
    export HYPERMULTIADL_DIR="${SCRIPT_DIR}/external/HyperMultiAssetedADL"

    # Step 4: Run HyperReplay replay script (generates enriched CSV with liquidated_total_equity)
    echo "[4/${TOTAL_STEPS}] Running HyperReplay replay script (generates enriched CSV)..."
    cd "${HYPERREPLAY_DIR}"
    uv run python scripts/replay_real_time_accounts.py
    cd "${SCRIPT_DIR}"

    # Verify enriched CSV has required columns
    ENRICHED_CSV="${HYPERREPLAY_DIR}/data/canonical/adl_detailed_analysis_REALTIME.csv"
    if ! head -1 "${ENRICHED_CSV}" | grep -q "liquidated_total_equity"; then
        echo "[error] Enriched CSV missing liquidated_total_equity column"
        exit 1
    fi
    echo "       Enriched CSV generated: ${ENRICHED_CSV}"

    # Step 5: Run full OSS pipeline to generate analysis CSVs
    echo "[5/${TOTAL_STEPS}] Running full OSS analysis pipeline..."
    uv run oss-adl \
        --out "${SCRIPT_DIR}/out" \
        all \
        --gap-ms 5000 \
        --gap-seconds 5.0 \
        --follower-decay-beta 8.0 \
        --queue-deficit-select-mode total_equity \
        --queue-deficit-value-mode total_equity \
        --horizons-ms 0,500,1000,2000,5000 \
        --horizon-ms 0 \
        --trim-alpha 0.03

    FIGURE_STEP=6
    VERIFY_STEP=7
    COMPILE_STEP=8
else
    FIGURE_STEP=3
    VERIFY_STEP=4
    COMPILE_STEP=5
fi

# Generate figures
echo "[${FIGURE_STEP}/${TOTAL_STEPS}] Generating figures from data..."
mkdir -p out/figures
if [[ "${FULL_MODE}" -eq 1 ]]; then
    # Pass canonical CSV path explicitly in full mode
    uv run oss-adl --out "${SCRIPT_DIR}/out" plots \
        --canonical-csv "${HYPERREPLAY_DIR}/data/canonical/adl_detailed_analysis_REALTIME.csv"
else
    uv run oss-adl --out "${SCRIPT_DIR}/out" plots
fi

# Verify figures match reference hashes (skip in --full mode since regenerated data may differ slightly)
if [[ "${FULL_MODE}" -eq 1 ]]; then
    echo "[${VERIFY_STEP}/${TOTAL_STEPS}] Skipping hash verification (--full mode regenerates data)"
    echo "       To verify, compare figures visually or run: uv run pytest tests/test_figure_hashes.py -v"
else
    echo "[${VERIFY_STEP}/${TOTAL_STEPS}] Verifying figures match reference..."
    uv run pytest tests/test_figure_hashes.py -v
fi

# Copy figures to paper and compile
echo "[${COMPILE_STEP}/${TOTAL_STEPS}] Compiling paper..."
mkdir -p paper/oss_figures
cp out/figures/*.png paper/oss_figures/

cd paper
if command -v pdflatex &>/dev/null; then
    pdflatex -interaction=nonstopmode main_corrected.tex || true
    bibtex main_corrected || true
    pdflatex -interaction=nonstopmode main_corrected.tex || true
    pdflatex -interaction=nonstopmode main_corrected.tex || true
    echo ""
    echo "PDF: paper/main_corrected.pdf"
else
    echo "[skip] pdflatex not found, skipping PDF compilation"
    echo "       Install texlive to compile the paper"
fi

echo ""
echo "=== Done! ==="
echo "Figures: out/figures/"
echo ""
if [[ "${FULL_MODE}" -eq 1 ]]; then
    echo "Full reproduction completed from raw HyperReplay data."
else
    echo "Note: Figures were regenerated from pre-included data (out/*.csv, out/*.json)."
    echo "      For full reproduction from raw data, run: ./run-all.sh --full"
fi
