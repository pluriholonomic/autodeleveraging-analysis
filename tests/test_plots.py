"""Tests for figure generation."""

import pytest
from pathlib import Path

from oss_adl.plots import (
    plot_headline_bars,
    plot_overshoot_vs_horizon,
    plot_policy_per_wave_performance,
    plot_policy_cumulative_overshoot,
    plot_cumulative_regret_historical,
)


def test_plot_headline_bars(headlines_json: Path, tmp_path: Path):
    """Test headline bar chart generation."""
    if not headlines_json.exists():
        pytest.skip("headlines.json not found")

    out_png = tmp_path / "headlines.png"
    plot_headline_bars(headlines_json=headlines_json, out_png=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_overshoot_vs_horizon(out_dir: Path, tmp_path: Path):
    """Test overshoot vs horizon plot generation."""
    sweep_csv = out_dir / "eval_horizon_sweep_gap_ms=5000.csv"
    if not sweep_csv.exists():
        pytest.skip("eval_horizon_sweep CSV not found")

    out_png = tmp_path / "overshoot.png"
    plot_overshoot_vs_horizon(horizon_sweep_csv=sweep_csv, out_png=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_policy_per_wave_performance(policy_csv: Path, tmp_path: Path):
    """Test policy performance plot generation."""
    if not policy_csv.exists():
        pytest.skip("policy_per_wave_metrics.csv not found")

    out_png = tmp_path / "performance.png"
    plot_policy_per_wave_performance(policy_per_wave_csv=policy_csv, out_png=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_policy_cumulative_overshoot(policy_csv: Path, tmp_path: Path):
    """Test cumulative overshoot plot generation."""
    if not policy_csv.exists():
        pytest.skip("policy_per_wave_metrics.csv not found")

    out_png = tmp_path / "cumulative.png"
    plot_policy_cumulative_overshoot(policy_per_wave_csv=policy_csv, out_png=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_cumulative_regret_historical(policy_csv: Path, tmp_path: Path):
    """Test cumulative regret plot generation."""
    if not policy_csv.exists():
        pytest.skip("policy_per_wave_metrics.csv not found")

    out_png = tmp_path / "regret.png"
    plot_cumulative_regret_historical(policy_per_wave_csv=policy_csv, out_png=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0
