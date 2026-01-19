"""Shared test fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def out_dir(project_root: Path) -> Path:
    """Return the out directory with pre-generated data."""
    return project_root / "out"


@pytest.fixture
def headlines_json(out_dir: Path) -> Path:
    """Return path to headlines.json."""
    return out_dir / "headlines.json"


@pytest.fixture
def policy_csv(out_dir: Path) -> Path:
    """Return path to policy_per_wave_metrics.csv."""
    return out_dir / "policy_per_wave_metrics.csv"
