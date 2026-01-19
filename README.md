# ADL Analysis - Minimal Reproducible Setup

[![CI](https://github.com/pluriholonomic/autodeleveraging-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/pluriholonomic/autodeleveraging-analysis/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a minimal, self-contained setup to reproduce all figures in the ADL paper.

## Quick Start

```bash
./run-all.sh
```

This regenerates all figures from pre-included CSV/JSON data files and verifies they match reference hashes.

For **full reproduction from raw data** (regenerates the CSVs too):

```bash
./run-all.sh --full
```

### Default Mode (`./run-all.sh`)

1. Install uv if needed
2. Set up Python environment
3. Generate figures from pre-included data
4. Verify figures match reference hashes
5. Compile paper PDF (if pdflatex available)

### Full Mode (`./run-all.sh --full`)

1. Install uv if needed
2. Set up Python environment
3. Clone external dependencies (HyperReplay, HyperMultiAssetedADL)
4. Run HyperReplay replay script (generates enriched CSV with `liquidated_total_equity`)
5. Run full OSS analysis pipeline (generates all CSVs)
6. Generate figures
7. Skip hash verification (regenerated data may differ slightly)
8. Compile paper PDF

## Data

The repository includes pre-generated data files in `out/`:
- `headlines.json` - Headline metrics
- `policy_per_wave_metrics.csv` - Per-wave policy comparison data
- `eval_horizon_sweep_gap_ms=5000.csv` - Horizon sweep data

These are used to regenerate the figures in default mode. The `--full` mode
regenerates these CSVs from the raw HyperReplay data in `external/HyperReplay/data/raw/`.

## Docs and reports

- `methodology.md` — prose walk-through of the measurement model and how it maps to the code.
- `docs/FAQs.md` — FAQ/critique responses (markouts, observability, ledger questions).
- `out/ADL-Corrections-Full-Report.md` — generated corrections + visuals report (wealth-space vs production).
- `out/summary_totals.csv` — aggregated table backing the headlines (queue vs replay vs two-time metrics).

## Quick pointers

- Corrections report: `out/ADL-Corrections-Full-Report.md` (regen: `./scripts/visualize_corrections.sh`; full regen: `./run-all.sh --full` then rerun the script).
- Headlines and aggregates: `out/headlines.json`, `out/summary_totals.csv`.
- Figures: `out/figures/` (paper figures), `out/figures_corrections/` (corrections visuals).
- FAQs and methodology: `docs/FAQs.md`, `methodology.md`.

## Requirements

- **Python 3.10+**
- **Git** (for cloning external data)
- **Internet access** (to clone HyperReplay and HyperMultiAssetedADL)
- **LaTeX** (optional, for PDF compilation; texlive recommended)

## Manual Steps

If you prefer to run steps manually:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone external data
./scripts/clone_deps.sh

# 3. Set up Python environment
uv sync

# 4. Set environment variables
export HYPERREPLAY_DIR="$(pwd)/external/HyperReplay"
export HYPERMULTIADL_DIR="$(pwd)/external/HyperMultiAssetedADL"

# 5. Run the pipeline
uv run oss-adl --out ./out all \
    --gap-ms 5000 \
    --gap-seconds 5.0 \
    --follower-decay-beta 8.0 \
    --queue-deficit-select-mode total_equity \
    --queue-deficit-value-mode total_equity \
    --horizons-ms 0,500,1000,2000,5000 \
    --horizon-ms 0 \
    --trim-alpha 0.03

# 6. Copy figures and compile paper
cp out/figures/*.png paper/oss_figures/
cd paper && pdflatex main_corrected.tex && bibtex main_corrected && pdflatex main_corrected.tex && pdflatex main_corrected.tex
```

## Directory Structure

```
autodeleveraging-analysis/
├── README.md                    # This file
├── run-all.sh                   # Single script to reproduce everything
├── pyproject.toml               # Python dependencies
├── uv.lock                      # Locked dependencies
│
├── scripts/
│   └── clone_deps.sh            # Clone HyperReplay, HyperMultiAssetedADL
│
├── src/oss_adl/                 # Core Python package
│   ├── __init__.py
│   ├── cli.py                   # Main CLI entry point
│   ├── plots.py                 # Figure generation
│   ├── policy_per_wave.py       # Per-wave policy analysis
│   ├── two_pass_replay.py       # Two-pass replay methodology
│   ├── queue_overshoot.py       # Queue overshoot diagnostic
│   ├── bad_debt.py              # Deficit calculation
│   ├── vector_mirror_descent.py # Benchmark allocation
│   ├── adl_contract_pro_rata.py # Contract pro-rata allocation
│   └── paths.py                 # Path utilities
│
├── external/                    # Cloned by clone_deps.sh
│   ├── HyperReplay/             # Raw ADL data (pinned commit)
│   └── HyperMultiAssetedADL/    # Additional data (pinned commit)
│
├── out/                         # Generated outputs
│   ├── headlines.json
│   ├── policy_per_wave_metrics.csv
│   ├── eval_horizon_sweep_*.csv
│   └── figures/
│       ├── 01_headlines.png
│       ├── 02_overshoot_vs_horizon.png
│       ├── 05_policy_per_wave_performance.png
│       ├── 06_policy_per_wave_cumulative_overshoot.png
│       ├── 09_cumulative_regret_historical.png
│       ├── 10a_overshoot_regret.png
│       ├── 10b_fairness_regret.png
│       └── 10c_total_regret.png
│
└── paper/
    ├── main_corrected.tex       # Main paper
    ├── references.bib           # Bibliography
    ├── oss_figures/             # Figures copied from out/figures/
    └── corrected/               # Appendix tex files
```

## Figure Inventory

| Figure | File | Description |
|--------|------|-------------|
| 1 | `01_headlines.png` | Headline metrics (queue overshoot, etc.) |
| 2 | `02_overshoot_vs_horizon.png` | Overshoot vs. horizon sweep |
| 5 | `05_policy_per_wave_performance.png` | Policy per-wave performance |
| 6 | `06_policy_per_wave_cumulative_overshoot.png` | Cumulative overshoot by policy |
| 9 | `09_cumulative_regret_historical.png` | Cumulative regret over historical waves |
| 10a | `10a_overshoot_regret.png` | Overshoot regret decomposition |
| 10b | `10b_fairness_regret.png` | Fairness regret decomposition |
| 10c | `10c_total_regret.png` | Total regret decomposition |

## Pinned Dependencies

For reproducibility, external data is pinned to specific commits:

- **HyperReplay:** `a65fa2295fa98d4aef4db50499e5ba5a3e9fe9c4` (add-loser-equity-enrichment branch, includes `liquidated_total_equity`)
- **HyperMultiAssetedADL:** `79bad0fae259fc1fcd9fce960953ae3b398f2db7`

## Verification

After running `run-all.sh`, verify:

1. `out/headlines.json` exists and contains expected metrics
2. `out/policy_per_wave_metrics.csv` has 16 rows (one per wave)
3. All 8 figures exist in `out/figures/`
4. `paper/main_corrected.pdf` compiles without errors (if LaTeX installed)
5. Figures in PDF match those in `out/figures/`

## Generate the full corrections report

The consolidated corrections report (figures + markdown) lives at `out/ADL-Corrections-Full-Report.md`.

```bash
# Quick refresh from committed out/* data
./scripts/visualize_corrections.sh

# Full regeneration from raw data, then rebuild the report
./run-all.sh --full
./scripts/visualize_corrections.sh
```
