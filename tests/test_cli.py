"""Tests for CLI commands."""

import subprocess
import sys


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        [sys.executable, "-m", "oss_adl.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "oss-adl" in result.stdout or "usage" in result.stdout.lower()


def test_cli_plots_subcommand():
    """Test plots subcommand help."""
    result = subprocess.run(
        [sys.executable, "-m", "oss_adl.cli", "plots", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
