from __future__ import annotations

"""
Loser-side deficit (bad debt) wave construction.

This is a minimal OSS port of `scripts/compute_bad_debt_from_losers.py`.

Core point (addendum-aligned):
  D_t must be computed from *losers* (liquidated accounts), not the ADL’d winners.

We rely on the canonical HyperReplay ADL-fill table including:
  - liquidated_user
  - liquidated_total_equity   (or, at minimum, liquidated_total_unrealized_pnl)

IMPORTANT (common source of confusion):
  HyperReplay and HyperMultiAssetedADL both ship a file named `adl_detailed_analysis_REALTIME.csv`,
  but they are different exports.
  - The HyperReplay canonical CSV (`HyperReplay/data/canonical/adl_detailed_analysis_REALTIME.csv`)
    contains the loser-side `liquidated_*` fields used here.
  - The HyperMultiAssetedADL winners CSV typically does NOT include `liquidated_total_equity`;
    loser rows may instead be represented via `is_negative_equity` (combined mode) or via a separate
    `liquidations_full_12min.csv` export (separate mode). That schema is handled by `oss_adl.queue_overshoot`,
    not by this module.

See the addendum draft:
  https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def cluster_global_time(times_ms: np.ndarray, *, gap_ms: int) -> np.ndarray:
    """
    Global time clustering: start a new wave when the time gap exceeds gap_ms.
    """
    t = np.asarray(times_ms, dtype=np.int64)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    cut = np.zeros(t.size, dtype=np.int64)
    cut[0] = 1
    if t.size > 1:
        cut[1:] = (t[1:] - t[:-1] > int(gap_ms)).astype(np.int64)
    return np.cumsum(cut)


@dataclass(frozen=True)
class LoserWaves:
    waves: pd.DataFrame
    summary: dict


def compute_loser_waves(
    *,
    canonical_realtime_csv: Path,
    gap_ms: int = 5000,
    prefer_equity: bool = True,
) -> LoserWaves:
    """
    Compute loser deficit waves from the canonical ADL table.

    Columns expected (best case):
      - time
      - liquidated_user
      - liquidated_total_equity

    Fallback (if equity unavailable):
      - liquidated_total_unrealized_pnl
    """
    # Read only the header first so we can emit a helpful error if the caller points us at the
    # wrong "REALTIME.csv" (HyperMultiAssetedADL vs HyperReplay have different schemas).
    header_cols = list(pd.read_csv(canonical_realtime_csv, nrows=0).columns)
    have_liq_equity = "liquidated_total_equity" in header_cols
    have_liq_upnl = "liquidated_total_unrealized_pnl" in header_cols
    if not have_liq_equity and not have_liq_upnl:
        # This usually means the user pointed at the HyperMultiAssetedADL winners CSV, which has
        # `liquidated_user` but not the loser-side `liquidated_*` value columns.
        if "liquidated_user" in header_cols and "total_equity" in header_cols:
            raise ValueError(
                "compute_loser_waves expects an *enriched* HyperReplay canonical REALTIME table with "
                "loser-side `liquidated_*` value fields (e.g. `liquidated_total_equity`).\n\n"
                f"Provided: {canonical_realtime_csv}\n"
                "This file has `liquidated_user` but is missing both `liquidated_total_equity` and "
                "`liquidated_total_unrealized_pnl`, so it looks like the *winner-side* ADL table rather "
                "than an enriched loser-metrics table.\n\n"
                "This can happen if:\n"
                "- You passed the HyperMultiAssetedADL winners export (same filename, different schema), or\n"
                "- You passed a HyperReplay checkout that has not been enriched to add loser-side "
                "`liquidated_total_*` columns.\n\n"
                "Fix:\n"
                "- Point `HYPERREPLAY_DIR` at a HyperReplay checkout whose "
                "`data/canonical/adl_detailed_analysis_REALTIME.csv` includes `liquidated_total_equity` "
                "(or regenerate it by running HyperReplay’s replay script).\n"
                "- If you are working with the HyperMultiAssetedADL combined/separate liquidation schemas, "
                "use `oss-adl queue` (queue overshoot) instead; it supports `is_negative_equity` combined mode "
                "and `liquidations_full_12min.csv` separate mode."
            )
        raise ValueError(
            "canonical_realtime_csv missing both liquidated_total_equity and liquidated_total_unrealized_pnl"
        )

    # Only request the columns that exist (some HyperReplay versions may omit one of the two).
    usecols = ["time", "liquidated_user"]
    if have_liq_equity:
        usecols.append("liquidated_total_equity")
    if have_liq_upnl:
        usecols.append("liquidated_total_unrealized_pnl")

    df = pd.read_csv(canonical_realtime_csv, usecols=usecols)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(np.int64)
    df = df.sort_values("time").reset_index(drop=True)
    df["liquidated_user"] = df["liquidated_user"].fillna("").astype(str)

    use_equity = prefer_equity and ("liquidated_total_equity" in df.columns)
    if use_equity:
        v = pd.to_numeric(df["liquidated_total_equity"], errors="coerce").fillna(0.0).astype(float)
    else:
        if "liquidated_total_unrealized_pnl" not in df.columns:
            raise ValueError(
                "canonical_realtime_csv missing both liquidated_total_equity and liquidated_total_unrealized_pnl"
            )
        v = pd.to_numeric(df["liquidated_total_unrealized_pnl"], errors="coerce").fillna(0.0).astype(float)

    df["_wave"] = cluster_global_time(df["time"].to_numpy(np.int64), gap_ms=int(gap_ms))

    rows = []
    for wave_id, sub in df.groupby("_wave", sort=True):
        losers = sub[sub["liquidated_user"].str.len() > 0][["liquidated_user"]].copy()
        if losers.empty:
            continue

        # For each unique liquidated_user, take the minimum observed equity (or pnl) in the wave.
        idx = losers.index
        mins = pd.Series(v.loc[idx].to_numpy(float), index=idx).groupby(df.loc[idx, "liquidated_user"]).min()
        deficit = float((-mins[mins < 0.0]).sum())
        max_loss = float((-mins[mins < 0.0]).max()) if (mins < 0.0).any() else 0.0
        if deficit <= EPS:
            continue

        rows.append(
            {
                "wave": int(wave_id),
                "t_start_ms": int(sub["time"].min()),
                "t_end_ms": int(sub["time"].max()),
                "n_rows": int(len(sub)),
                "unique_losers": int(len(mins)),
                "deficit_usd": float(deficit),
                "max_loss_usd": float(max_loss),
            }
        )

    out = pd.DataFrame(rows)
    summary = {
        "input_csv": str(Path(canonical_realtime_csv)),
        "gap_ms": int(gap_ms),
        "prefer_equity": bool(prefer_equity),
        "deficit_measure": "liquidated_total_equity" if use_equity else "liquidated_total_unrealized_pnl",
        "num_waves_with_positive_deficit": int(len(out)),
        "total_deficit_usd": float(out["deficit_usd"].sum()) if len(out) else 0.0,
        "max_wave_deficit_usd": float(out["deficit_usd"].max()) if len(out) else 0.0,
        "max_wave_max_loss_usd": float(out["max_loss_usd"].max()) if len(out) else 0.0,
    }
    return LoserWaves(waves=out, summary=summary)


def write_loser_waves(*, out_dir: Path, loser_waves: LoserWaves) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bad_debt_loser_waves.csv"
    out_json = out_dir / "bad_debt_loser_waves.json"
    loser_waves.waves.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(loser_waves.summary, indent=2, sort_keys=True) + "\n")
    return out_csv, out_json
