#!/usr/bin/env python3
"""Generate pixel-data hashes for reference figures.

Hashes only the pixel data (ignoring PNG metadata like timestamps),
so figures can be compared for visual equivalence.
"""

import hashlib
import json
from pathlib import Path

from PIL import Image


def pixel_hash(image_path: Path) -> str:
    """Compute SHA256 hash of image pixel data only."""
    with Image.open(image_path) as img:
        # Convert to consistent format (RGBA) to ensure comparable hashes
        img = img.convert("RGBA")
        pixel_data = img.tobytes()
        return hashlib.sha256(pixel_data).hexdigest()


def main():
    figures_dir = Path(__file__).parent.parent / "out" / "figures"
    hashes = {}

    for png in sorted(figures_dir.glob("*.png")):
        h = pixel_hash(png)
        hashes[png.name] = h
        print(f"{png.name}: {h[:16]}...")

    output_file = Path(__file__).parent.parent / "tests" / "reference_hashes.json"
    with open(output_file, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"\nWrote {len(hashes)} hashes to {output_file}")


if __name__ == "__main__":
    main()
