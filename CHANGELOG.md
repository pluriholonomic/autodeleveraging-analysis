# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-17

### Added

- Initial public release
- Minimal reproducible setup for ADL paper figures
- Pre-generated data files for figure reproduction without enriched HyperReplay data
- CLI tool (`oss-adl`) for running the analysis pipeline
- GitHub Actions CI workflow for linting and testing
- Test suite for figure generation and CLI
- MIT License
- Contributing guidelines
- Code of Conduct

### Figures

- `01_headlines.png` - Headline metrics bar chart
- `02_overshoot_vs_horizon.png` - Overshoot vs horizon sweep
- `05_policy_per_wave_performance.png` - Policy per-wave performance comparison
- `06_policy_per_wave_cumulative_overshoot.png` - Cumulative overshoot by policy
- `09_cumulative_regret_historical.png` - Cumulative regret over historical waves
- `10a_overshoot_regret.png` - Overshoot regret decomposition
- `10b_fairness_regret.png` - Fairness regret decomposition
- `10c_total_regret.png` - Total regret decomposition

### Dependencies

- Python 3.10+
- pandas, matplotlib, numpy for data analysis
- ortools for MIP solver benchmarks
- ruff for linting

### External Data

- HyperReplay pinned to commit `b8d258b5b6d1967538f61bdbf97995d3852e643a`
- HyperMultiAssetedADL pinned to commit `79bad0fae259fc1fcd9fce960953ae3b398f2db7`
