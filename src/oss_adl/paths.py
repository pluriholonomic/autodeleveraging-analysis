from __future__ import annotations

"""
Path resolution for the OSS reproduction.

Design goal:
- Be explicit about what files we require from upstream repos
- Be robust to minor upstream layout differences (canonical vs "cash-only balances ..." folders)

We intentionally DO NOT download anything automatically here; cloning is handled by
`OSS/scripts/clone_deps.sh`.
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _csv_header_has_any(p: Path, needles: Sequence[str]) -> bool:
    """
    Lightweight CSV header probe (avoids importing pandas during path resolution).
    """
    try:
        with p.open("r", encoding="utf-8") as f:
            header = f.readline()
    except OSError:
        return False
    return any(n in header for n in needles)


@dataclass(frozen=True)
class RepoPaths:
    """
    Resolved locations for upstream repos and key input files.

    Environment variable overrides:
      - HYPERREPLAY_DIR
      - HYPERMULTIADL_DIR
    """

    hyperreplay_dir: Path
    hypermultiadl_dir: Path

    # HyperReplay key inputs
    hyperreplay_raw_dir: Path
    hyperreplay_canonical_realtime_csv: Path

    # HyperMultiAssetedADL key inputs (queue-overshoot pipeline)
    hypermultiadl_winners_csv: Path
    hypermultiadl_liquidations_csv: Path | None


def resolve_repo_paths(*, oss_root: Path) -> RepoPaths:
    """
    Resolve repo roots + the specific data files the OSS pipeline consumes.
    """
    oss_root = Path(oss_root).resolve()

    # Default to OSS/external/*, but allow env overrides.
    # Prefer a vendored top-level HyperReplay checkout (monorepo layout) *iff* it includes the
    # enriched loser-side columns needed by `oss_adl.bad_debt.compute_loser_waves`.
    if "HYPERREPLAY_DIR" in os.environ:
        hyperreplay_dir = Path(os.environ["HYPERREPLAY_DIR"]).resolve()
    else:
        external_hr = (oss_root / "external" / "HyperReplay").resolve()
        vendored_hr = (oss_root.parent / "HyperReplay").resolve()
        vendored_csv = vendored_hr / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"
        if _csv_header_has_any(vendored_csv, ["liquidated_total_equity", "liquidated_total_unrealized_pnl"]):
            hyperreplay_dir = vendored_hr
        else:
            hyperreplay_dir = external_hr
    hypermultiadl_dir = Path(
        os.environ.get("HYPERMULTIADL_DIR", oss_root / "external" / "HyperMultiAssetedADL")
    ).resolve()

    # --- HyperReplay paths ---
    hyperreplay_raw_dir = hyperreplay_dir / "data" / "raw"
    hyperreplay_canonical_realtime_csv = hyperreplay_dir / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"

    # --- HyperMultiAssetedADL paths ---
    # We try a small set of plausible locations; upstream has moved these around.
    winners_candidates = [
        hypermultiadl_dir
        / "data"
        / "canonical"
        / "cash-only balances ADL event orderbook 2025-10-10"
        / "adl_detailed_analysis_REALTIME.csv",
        hypermultiadl_dir / "cash-only balances ADL event orderbook 2025-10-10" / "adl_detailed_analysis_REALTIME.csv",
        hypermultiadl_dir / "adl_detailed_analysis_REALTIME.csv",
    ]
    liq_candidates = [
        hypermultiadl_dir
        / "data"
        / "canonical"
        / "cash-only balances ADL event orderbook 2025-10-10"
        / "liquidations_full_12min.csv",
        hypermultiadl_dir / "cash-only balances ADL event orderbook 2025-10-10" / "liquidations_full_12min.csv",
        hypermultiadl_dir / "liquidations_full_12min.csv",
    ]

    winners_csv = _first_existing(winners_candidates)
    liq_csv = _first_existing(liq_candidates)

    missing = []
    if not hyperreplay_raw_dir.exists():
        missing.append(f"HyperReplay raw dir not found: {hyperreplay_raw_dir}")
    if not hyperreplay_canonical_realtime_csv.exists():
        missing.append(f"HyperReplay canonical CSV not found: {hyperreplay_canonical_realtime_csv}")
    if winners_csv is None:
        missing.append(
            "HyperMultiAssetedADL winners CSV not found. Tried:\n  - "
            + "\n  - ".join(str(p) for p in winners_candidates)
        )
    # Liquidations are optional because some upstream layouts ship a single combined CSV that
    # contains both winners and loser-side rows (identified by `is_negative_equity`).
    # In that case, the queue reproduction can run without `liquidations_full_12min.csv`.

    if missing:
        raise FileNotFoundError("\n".join(missing))

    return RepoPaths(
        hyperreplay_dir=hyperreplay_dir,
        hypermultiadl_dir=hypermultiadl_dir,
        hyperreplay_raw_dir=hyperreplay_raw_dir,
        hyperreplay_canonical_realtime_csv=hyperreplay_canonical_realtime_csv,
        hypermultiadl_winners_csv=winners_csv,
        hypermultiadl_liquidations_csv=liq_csv,
    )
