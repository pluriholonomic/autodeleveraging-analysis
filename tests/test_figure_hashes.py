"""Test that generated figures match reference hashes."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


def pixel_hash(image_path: Path) -> str:
    """Compute SHA256 hash of image pixel data only."""
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        pixel_data = img.tobytes()
        return hashlib.sha256(pixel_data).hexdigest()


@pytest.fixture
def reference_hashes() -> dict[str, str]:
    """Load reference hashes."""
    hashes_file = Path(__file__).parent / "reference_hashes.json"
    if not hashes_file.exists():
        pytest.skip("reference_hashes.json not found")
    with open(hashes_file) as f:
        return json.load(f)


@pytest.fixture
def figures_dir(project_root: Path) -> Path:
    """Return the figures directory."""
    return project_root / "out" / "figures"


def test_figure_hashes_match(reference_hashes: dict[str, str], figures_dir: Path):
    """Verify all generated figures match their reference hashes."""
    if not figures_dir.exists():
        pytest.skip("out/figures/ not found")

    mismatches = []
    missing = []

    for filename, expected_hash in reference_hashes.items():
        figure_path = figures_dir / filename
        if not figure_path.exists():
            missing.append(filename)
            continue

        actual_hash = pixel_hash(figure_path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{filename}: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )

    if missing:
        pytest.fail(f"Missing figures: {missing}")

    if mismatches:
        pytest.fail(f"Hash mismatches:\n" + "\n".join(mismatches))


def test_no_extra_figures(reference_hashes: dict[str, str], figures_dir: Path):
    """Check for unexpected figures not in reference."""
    if not figures_dir.exists():
        pytest.skip("out/figures/ not found")

    actual_figures = {p.name for p in figures_dir.glob("*.png")}
    expected_figures = set(reference_hashes.keys())
    extra = actual_figures - expected_figures

    if extra:
        pytest.fail(f"Unexpected figures not in reference: {extra}")
