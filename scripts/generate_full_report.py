#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def usd(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}\\${v:,.2f}"


def fmt_ratio(x: object, decimals: int = 2) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "n/a"


def fmt_int(x: object) -> str:
    try:
        v = int(x)
    except Exception:
        return "n/a"
    return f"{v:,}"


def fmt_latency(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if v >= 60000.0:
        return f"{v / 60000.0:.1f} min"
    if v >= 1000.0:
        return f"{v / 1000.0:.1f} s"
    return f"{int(round(v)):,} ms"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_reproduce_args(paths: List[Path]) -> dict:
    text = ""
    for path in paths:
        if path.exists():
            text = path.read_text()
            break
    if not text:
        return {}
    mapping = {
        "--gap-ms": "gap_ms",
        "--gap-seconds": "gap_seconds",
        "--follower-decay-beta": "follower_decay_beta",
        "--queue-deficit-select-mode": "queue_deficit_select_mode",
        "--queue-deficit-value-mode": "queue_deficit_value_mode",
        "--horizons-ms": "horizons_ms",
        "--horizon-ms": "horizon_ms",
        "--trim-alpha": "trim_alpha",
        "--canonical-csv": "canonical_csv",
    }
    cfg = {v: "n/a" for v in mapping.values()}
    for flag, key in mapping.items():
        match = re.search(rf"{re.escape(flag)}(?:=|\\s+)([^\\s\\\\]+)", text)
        if match:
            cfg[key] = match.group(1).strip("\"'")
    return cfg




def summarize_gap_sweeps(out_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(out_dir.glob("eval_horizon_sweep_gap_ms=*.csv")):
        match = re.search(r"gap_ms=(\\d+)", path.name)
        if not match:
            continue
        gap_ms = int(match.group(1))
        df = load_csv(path)
        if df.empty:
            continue
        if "eval_horizon_ms" in df.columns:
            df = df.sort_values("eval_horizon_ms")
            row0 = df[df["eval_horizon_ms"] == 0]
            row = row0.iloc[0] if not row0.empty else df.iloc[0]
        else:
            row = df.iloc[0]
        rows.append(
            {
                "gap_ms": gap_ms,
                "num_waves": row.get("num_waves"),
                "total_budget_needed_usd": row.get("total_budget_needed_usd"),
                "total_H_prod_usd": row.get("total_H_prod_usd"),
                "total_overshoot_prod_minus_needed_usd": row.get("total_overshoot_prod_minus_needed_usd"),
            }
        )
    return rows


def horizon_value(sweep: pd.DataFrame, horizon_ms: int) -> Optional[float]:
    if sweep.empty:
        return None
    if "eval_horizon_ms" not in sweep.columns:
        return None
    if "total_overshoot_prod_minus_needed_usd" not in sweep.columns:
        return None
    df = sweep.copy()
    df["eval_horizon_ms"] = pd.to_numeric(df["eval_horizon_ms"], errors="coerce")
    row = df[df["eval_horizon_ms"] == float(horizon_ms)]
    if row.empty:
        return None
    return float(row.iloc[0]["total_overshoot_prod_minus_needed_usd"])


def compute_latency_summary(out_dir: Path) -> List[dict]:
    per_user_path = out_dir / "two_time" / "per_user_reaction_metrics.csv"
    per_event_path = out_dir / "two_time" / "per_adl_event_metrics.csv"
    if not per_user_path.exists() or not per_event_path.exists():
        return []
    per_user = load_csv(per_user_path)
    per_event = load_csv(per_event_path)
    if per_user.empty or per_event.empty:
        return []

    x_col = "share_any_nonadl_fill_60000ms"
    y_col = "mean_undo_frac_60000ms"
    if x_col not in per_user.columns or y_col not in per_user.columns:
        return []

    x = pd.to_numeric(per_user[x_col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(per_user[y_col], errors="coerce").fillna(0.0)
    cohort = pd.Series("other", index=per_user.index)
    cohort[(x >= 0.5) & (y >= 0.25)] = "strategic"
    cohort[(x < 0.5) & (y < 0.25)] = "passive"
    per_user = per_user.assign(_cohort=cohort)

    per_event = per_event.copy()
    if "latency_to_first_nonadl_fill_ms" not in per_event.columns or "user" not in per_event.columns:
        return []
    per_event["latency_to_first_nonadl_fill_ms"] = pd.to_numeric(
        per_event["latency_to_first_nonadl_fill_ms"], errors="coerce"
    )
    per_event = per_event.dropna(subset=["latency_to_first_nonadl_fill_ms"])

    cohort_map = per_user.set_index("user")["_cohort"]
    per_event["_cohort"] = per_event["user"].map(cohort_map)
    per_event = per_event[per_event["_cohort"].isin(["strategic", "passive"])]
    if per_event.empty:
        return []

    rows: List[dict] = []
    for name in ["strategic", "passive"]:
        sub = per_event[per_event["_cohort"] == name]
        lat = sub["latency_to_first_nonadl_fill_ms"].to_numpy(float)
        if lat.size == 0:
            continue
        rows.append(
            {
                "cohort": name,
                "n_users": int(per_user[per_user["_cohort"] == name]["user"].nunique()),
                "n_events": int(len(sub)),
                "median_ms": float(np.median(lat)),
                "p10_ms": float(np.percentile(lat, 10)),
                "p90_ms": float(np.percentile(lat, 90)),
            }
        )
    return rows


def image_block(
    lines: List[str],
    *,
    title: str,
    img_path: Path,
    rel_path: Path,
    desc: str,
    fig_label: Optional[str] = None,
) -> None:
    lines.append(f"### {title}")
    if img_path.exists():
        lines.append(f"![]({rel_path.as_posix()})")
    else:
        lines.append(f"_Missing image: {rel_path}_")
    lines.append("")
    if fig_label:
        clean_title = title.split(". ", 1)[1] if ". " in title else title
        lines.append(f"Figure {fig_label}. {clean_title}")
        lines.append("")
    lines.append(desc)
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a consolidated ADL report with inline visuals.")
    parser.add_argument("--out", default="out", help="Output directory (default: out)")
    parser.add_argument("--fig-dir", default=None, help="Figures directory (default: <out>/figures_corrections)")
    parser.add_argument(
        "--output",
        default=None,
        help="Report path (default: <out>/ADL-Corrections-Full-Report.md)",
    )
    parser.add_argument("--old-queue-overshoot", type=float, default=None, help="Optional legacy queue overshoot headline")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir / "figures_corrections"
    output_path = Path(args.output) if args.output else out_dir / "ADL-Corrections-Full-Report.md"
    if output_path.name == "ADL-Corrections-Full-Report.md":
        from generate_narrative_report import main as narrative_main

        return narrative_main()

    headlines = load_json(out_dir / "headlines.json")
    h = headlines.get("headline", {})
    d = headlines.get("derived", {})
    oc = h.get("winner_overcollateralization_equity_over_pnl", {}) if isinstance(h, dict) else {}

    queue_overshoot = usd(h.get("queue_total_overshoot_usd", 0.0))
    prod_overshoot = usd(h.get("prod_overshoot_vs_needed_usd", 0.0))
    naive_expected = usd(d.get("naive_expected_pnl_closed_usd", 0.0))
    factor = d.get("factor_naive_expected_over_observed", "n/a")
    oc_trim = fmt_ratio(oc.get("trimmed_mean", "n/a"), 2)
    factor_fmt = fmt_ratio(factor, 2)
    gap_summaries = summarize_gap_sweeps(out_dir)
    sweep = load_csv(out_dir / "eval_horizon_sweep_gap_ms=5000.csv")
    h0 = horizon_value(sweep, 0) if sweep is not None else None
    h500 = horizon_value(sweep, 500) if sweep is not None else None
    h1000 = horizon_value(sweep, 1000) if sweep is not None else None
    latency_rows = compute_latency_summary(out_dir)

    lines: List[str] = []
    lines.append("# ADL Corrections and Analysis Report")
    lines.append("")
    lines.append("This report consolidates the corrected definitions, headline results, visualizations, and regeneration steps.")
    lines.append("It reflects the updated ADL code and `methodology.md` (the paper is not yet updated).")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The core correction is that queue overshoot must use **PnL-only capacity**, not equity + principal.")
    lines.append("With that fix, the wealth-space queue overshoot (equity USD) remains large, while the production")
    lines.append("overshoot-vs-needed (PnL USD) is materially smaller, and the gap is explained by overcollateralization")
    lines.append("and two-time behavior.")
    lines.append("")

    lines.append("## Key Headlines")
    lines.append("")
    lines.append(f"- Queue overshoot (equity USD): {queue_overshoot}")
    lines.append(f"- Production overshoot vs needed (PnL USD, horizon=0): {prod_overshoot}")
    lines.append(f"- Winner overcollateralization (trimmed mean): {oc_trim}x")
    lines.append(f"- Naive expected PnL closed (queue / ratio): {naive_expected}")
    lines.append(f"- Naive expected over observed factor: {factor_fmt}x")
    lines.append("")
    lines.append("Source: `out/headlines.json` and `out/analysis_summary.md`.")
    lines.append("")
    if h500 is not None or h1000 is not None:
        parts = []
        if h500 is not None:
            parts.append(f"500ms: {usd(h500)}")
        if h1000 is not None:
            parts.append(f"1000ms: {usd(h1000)}")
        if parts:
            lines.append(f"Note: headline uses horizon=0 ({usd(h0) if h0 is not None else 'n/a'}).")
            lines.append(f"Horizon sweep examples: {', '.join(parts)}.")
            lines.append("")

    if gap_summaries:
        base = gap_summaries[0]
        try:
            q_val = float(h.get("queue_total_overshoot_usd", 0.0))
        except Exception:
            q_val = None
        try:
            h_prod_val = float(base.get("total_H_prod_usd"))
        except Exception:
            h_prod_val = None
        equity_gap = usd(q_val - h_prod_val) if q_val is not None and h_prod_val is not None else "n/a"

        lines.append("## Equity USD → PnL USD Reconciliation (reader-friendly)")
        lines.append("")
        lines.append("This table shows how the equity-USD queue headline relates to the PnL-USD replay objects.")
        lines.append("These are **not** the same units, so the final row is illustrative only.")
        lines.append("")
        lines.append("| item | amount | note |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| Queue overshoot (equity USD) | {queue_overshoot} | Wealth-space queue headline |")
        lines.append(f"| Naive expected PnL closed | {naive_expected} | Queue overshoot / overcollateralization ratio |")
        lines.append(f"| Needed budget $B_{{\\text{{needed}}}}$ | {usd(base.get('total_budget_needed_usd'))} | Bankruptcy-gap proxy |")
        lines.append(f"| Production haircut $H_{{\\text{{prod}}}}$ | {usd(base.get('total_H_prod_usd'))} | Two-pass replay counterfactual |")
        lines.append(
            f"| Overshoot vs needed $H_{{\\text{{prod}}}}-B_{{\\text{{needed}}}}$ | "
            f"{usd(base.get('total_overshoot_prod_minus_needed_usd'))} | PnL USD headline |"
        )
        lines.append(f"| Equity-to-PnL gap (illustrative) | {equity_gap} | Not a missing PnL; different objects |")
        lines.append("")

    lines.append("## Definitions")
    lines.append("")
    lines.append("- **Queue overshoot (equity USD)**: wealth-space overshoot from a stylized queue using PnL-only capacity.")
    lines.append("- **Loser deficit** $D_t$: total negative equity on liquidated accounts within a wave.")
    lines.append("- **Needed budget** $B_{\\text{needed}}$: bankruptcy-gap proxy from ADL fills ($|\\mathrm{markPx}-\\mathrm{execPx}|\\times|\\mathrm{size}|$).")
    lines.append("- **Production haircut** $H_{\\text{prod}}$: two-pass counterfactual equity loss (ADL-on vs ADL-off).")
    lines.append("- **Overshoot vs needed**: $H_{\\text{prod}} - B_{\\text{needed}}$ (PnL USD headline).")
    lines.append("- **Evaluation horizon** $\\Delta$: post-wave price-path window used to measure opportunity-cost effects.")
    lines.append("")

    lines.append("## What this is not")
    lines.append("")
    lines.append("- The queue overshoot is **not** a production replay; it is a wealth-space abstraction.")
    lines.append("- The two-pass replay fixes the price path and removes ADL state updates; it is a counterfactual object.")
    lines.append("- Per-coin clustering (queue) and global waves (two-pass) are different objects and should not be mixed.")
    lines.append("")

    lines.append("## What Was Fixed (and where)")
    lines.append("")
    lines.append("- **Queue overshoot capacity now uses PnL-only** (`position_unrealized_pnl`), not equity + principal.")
    lines.append("  - Code: `src/oss_adl/queue_overshoot.py:74`")
    lines.append("- **Queue overshoot deficit proxy is loser-side**, not winner equity.")
    lines.append("  - Code: `src/oss_adl/queue_overshoot.py:179`, `src/oss_adl/queue_overshoot.py:292`")
    lines.append("- **Production overshoot is reconstructed with two-pass replay** (ADL-on vs ADL-off on the same price path).")
    lines.append("  - Code: `src/oss_adl/two_pass_replay.py:14`, `src/oss_adl/two_pass_replay.py:423`")
    lines.append("- **Loser deficit waves use loser-side equity** and global time clustering.")
    lines.append("  - Code: `src/oss_adl/bad_debt.py:30`, `src/oss_adl/bad_debt.py:80`")
    lines.append("")

    lines.append("## Methodology Map (paper to code)")
    lines.append("")
    lines.append("- Queue overshoot (wealth-space queue): `methodology.md` section 3.7; `src/oss_adl/queue_overshoot.py`.")
    lines.append("- Loser deficit waves $D_t$: `methodology.md` section 3.2; `src/oss_adl/bad_debt.py`.")
    lines.append("- Needed budget $B_{\\text{needed}}$: `methodology.md` section 3.3; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Production haircut $H_{\\text{prod}}$: `methodology.md` section 3.4; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Horizon sweep $\\Delta$: `methodology.md` section 3.6; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Winner overcollateralization: `methodology.md` section 3.8; `src/oss_adl/queue_overshoot.py`.")
    lines.append("- Two-time behavior (undo fraction): `methodology.md` section 3.9; `src/oss_adl/two_time_behavior.py`.")
    lines.append("")

    lines.append("## Visualizations")
    lines.append("")
    def rel(name: str) -> Path:
        img_path = fig_dir / name
        try:
            return img_path.relative_to(out_dir)
        except ValueError:
            return img_path

    image_block(
        lines,
        title="00. Overview Tiles",
        img_path=fig_dir / "00_overview_tiles.png",
        rel_path=rel("00_overview_tiles.png"),
        fig_label="00",
        desc=(
            "Quick read of the three headline quantities: corrected queue overshoot, "
            "production overshoot vs needed, and overcollateralization ratio."
        ),
    )
    image_block(
        lines,
        title="01. Definition Diff (the core fix)",
        img_path=fig_dir / "01_definition_diff.png",
        rel_path=rel("01_definition_diff.png"),
        fig_label="01",
        desc=(
            "Shows the correction: queue capacity must be PnL-only, not equity + principal. "
            "This is the main definitional fix."
        ),
    )
    image_block(
        lines,
        title="02. Mapping Bars (why the gap exists)",
        img_path=fig_dir / "02_mapping_bars.png",
        rel_path=rel("02_mapping_bars.png"),
        fig_label="02",
        desc=(
            "Explains why a large equity USD overshoot does not map 1:1 to PnL USD overshoot. "
            "The naive mapping still overpredicts."
        ),
    )
    image_block(
        lines,
        title="03. Horizon Robustness",
        img_path=fig_dir / "03_horizon_robustness.png",
        rel_path=rel("03_horizon_robustness.png"),
        fig_label="03",
        desc=(
            "Overshoot vs evaluation horizon (opportunity-cost channel). "
            "This is a robustness view, not a new headline."
        ),
    )
    lines.append(
        "Interpretation: the curve's slope is a proxy for time sensitivity. Strategic users care more about "
        "the gradient (short-horizon opportunity cost), while passive users care more about the point estimate."
    )
    lines.append("")
    image_block(
        lines,
        title="04. Queue Concentration",
        img_path=fig_dir / "04_queue_concentration.png",
        rel_path=rel("04_queue_concentration.png"),
        fig_label="04",
        desc=(
            "Shows whether queue overshoot (equity USD) is broad-based or dominated by a few coins."
        ),
    )
    image_block(
        lines,
        title="05. Two-Time Scatter (strategic vs passive)",
        img_path=fig_dir / "05_two_time_scatter.png",
        rel_path=rel("05_two_time_scatter.png"),
        fig_label="05",
        desc=(
            "Behavioral classification using undo fraction vs trading share; "
            "helps explain the equity USD vs PnL USD gap."
        ),
    )
    lines.append("Undo fraction is an interpretable proxy for rapid reversal (strategic intent).")
    lines.append("")
    image_block(
        lines,
        title="06. Two-Time Quadrant Shares",
        img_path=fig_dir / "06_two_time_quadrant_shares.png",
        rel_path=rel("06_two_time_quadrant_shares.png"),
        fig_label="06",
        desc=(
            "Share of users in each strategic/passive quadrant (user mass), making the distribution explicit."
        ),
    )
    image_block(
        lines,
        title="07. Two-Time Quadrant Shares (Volume-weighted)",
        img_path=fig_dir / "07_two_time_quadrant_shares_volume.png",
        rel_path=rel("07_two_time_quadrant_shares_volume.png"),
        fig_label="07",
        desc=(
            "Shares weighted by non-ADL volume; this is the weighting where the strategic quadrant can dominate."
        ),
    )
    if latency_rows:
        lines.append("### 08. Response-Time Proxy (strategic vs passive)")
        lines.append("")
        lines.append("Latency is computed from per-event time to first non-ADL fill (60s window).")
        lines.append("")
        lines.append("| cohort | n_users | n_events | median | p10 | p90 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in latency_rows:
            lines.append(
                f"| {row['cohort']} | {fmt_int(row['n_users'])} | {fmt_int(row['n_events'])} | "
                f"{fmt_latency(row['median_ms'])} | {fmt_latency(row['p10_ms'])} | {fmt_latency(row['p90_ms'])} |"
            )
        lines.append("")
    lines.append(
        "Per-symbol comparisons below use a PnL-closed proxy: the per-coin needed budget "
        "from ADL fill gaps ($|\\mathrm{markPx}-\\mathrm{execPx}|\\times|\\mathrm{size}|$) at horizon=0."
    )
    lines.append("")
    image_block(
        lines,
        title="09. Equity vs PnL Proxy (per symbol)",
        img_path=fig_dir / "08_equity_vs_needed_scatter.png",
        rel_path=rel("08_equity_vs_needed_scatter.png"),
        fig_label="09",
        desc=(
            "Each point is a symbol. Compares equity overshoot per coin to the per-coin needed-budget proxy. "
            "Color encodes non-ADL volume as a liquidity proxy (no minimum volume filter)."
        ),
    )
    image_block(
        lines,
        title="10. Gap Ratio vs Liquidity",
        img_path=fig_dir / "09_gap_ratio_vs_liquidity.png",
        rel_path=rel("09_gap_ratio_vs_liquidity.png"),
        fig_label="10",
        desc=(
            "Scatter of equity/PnL-proxy ratio against non-ADL volume. "
            "Downward slope would indicate lower-liquidity coins get hit worse."
        ),
    )
    image_block(
        lines,
        title="11. Largest Equity/PnL Gaps",
        img_path=fig_dir / "10_gap_ratio_bars.png",
        rel_path=rel("10_gap_ratio_bars.png"),
        fig_label="11",
        desc=(
            "Top symbols by equity/PnL-proxy ratio (filtered to avoid tiny denominators)."
        ),
    )
    image_block(
        lines,
        title="12. Undo Fraction vs Impact Proxy (per coin)",
        img_path=fig_dir / "12_undo_vs_impact_per_coin.png",
        rel_path=rel("12_undo_vs_impact_per_coin.png"),
        fig_label="12",
        desc=(
            "Per-coin, volume-weighted undo fraction vs impact proxy (gap per contract). "
            "Color encodes non-ADL volume as a liquidity proxy."
        ),
    )
    lines.append(
        "Note: the aggregate relationship is noisy and likely low-SNR; this plot is most useful as a per-coin diagnostic."
    )
    lines.append("")
    lines.append(
        "These per-asset gaps are relevant for considering asset-specific ADL mechanisms based on liquidity."
    )
    lines.append("")

    cfg = parse_reproduce_args([Path("run-all.sh")])
    if any(v != "n/a" for v in cfg.values()):
        lines.append("## Run Configuration (repro script)")
        lines.append("")
        lines.append(f"- gap_ms: {cfg.get('gap_ms', 'n/a')}")
        lines.append(f"- gap_seconds: {cfg.get('gap_seconds', 'n/a')}")
        lines.append(f"- follower_decay_beta: {cfg.get('follower_decay_beta', 'n/a')}")
        lines.append(f"- queue_deficit_select_mode: {cfg.get('queue_deficit_select_mode', 'n/a')}")
        lines.append(f"- queue_deficit_value_mode: {cfg.get('queue_deficit_value_mode', 'n/a')}")
        lines.append(f"- horizons_ms: {cfg.get('horizons_ms', 'n/a')}")
        lines.append(f"- horizon_ms: {cfg.get('horizon_ms', 'n/a')}")
        lines.append(f"- trim_alpha: {cfg.get('trim_alpha', 'n/a')}")
        lines.append(f"- canonical_csv: {cfg.get('canonical_csv', 'n/a')}")
        lines.append("")

    if args.old_queue_overshoot is not None:
        lines.append("## Correction Impact")
        lines.append("")
        delta = float(args.old_queue_overshoot) - float(h.get("queue_total_overshoot_usd", 0.0))
        lines.append(f"- Old queue overshoot: {usd(args.old_queue_overshoot)}")
        lines.append(f"- Corrected queue overshoot: {queue_overshoot}")
        lines.append(f"- Absolute change: {usd(delta)}")
        lines.append("")

    rob = load_json(out_dir / "overshoot_robustness.json")
    if rob:
        lines.append("## Sensitivity Summary (horizon and discounting)")
        lines.append("")
        lines.append("Definitions:")
        lines.append("- $O_0$: overshoot vs needed at horizon $\\Delta=0$ (no opportunity-cost window).")
        lines.append("- $O_{\\mathrm{disc}}$: exponentially time-discounted average of overshoot across horizons.")
        lines.append("- $\\Delta O_{\\mathrm{disc}}$: uplift over $O_0$ from longer-horizon effects.")
        lines.append("")
        if "O0" in rob:
            lines.append(f"- baseline $O_0$ (horizon=0): {usd(rob.get('O0'))}")
        if "O_min_band" in rob and "O_max_band" in rob:
            band_ms = rob.get("band_max_ms")
            try:
                band_label = f"0..{int(band_ms)} ms" if band_ms is not None else "0..band_max_ms"
            except Exception:
                band_label = "0..band_max_ms"
            lines.append(
                f"- horizon band ({band_label}): "
                f"{usd(rob.get('O_min_band'))} to {usd(rob.get('O_max_band'))}"
            )
        discounted = rob.get("discounted", {})
        if discounted:
            def sort_key(val: str) -> int:
                try:
                    return int(val)
                except Exception:
                    return 10**12

            for key in sorted(discounted.keys(), key=sort_key):
                entry = discounted.get(key, {})
                lines.append(
                    f"- half_life_ms={key}: $O_{{\\mathrm{{disc}}}}$={usd(entry.get('O_disc'))}, "
                    f"$\\Delta O_{{\\mathrm{{disc}}}}$={usd(entry.get('DeltaO_disc'))}"
                )
        lines.append("")

    if gap_summaries:
        lines.append("## Sensitivity Coverage (gap_ms / wave clustering)")
        lines.append("")
        lines.append(
            "Sensitivity coverage is thin: we show horizon discounting, but not sensitivity "
            "to `gap_ms`, wave clustering, or other replay parameters. This is important for robustness."
        )
        lines.append("")
        lines.append("Current gap_ms results (horizon=0):")
        lines.append("")
        lines.append("| gap_ms | num_waves | $B_{\\text{needed}}$ | $H_{\\text{prod}}$ | $H_{\\text{prod}}-B_{\\text{needed}}$ |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in sorted(gap_summaries, key=lambda r: int(r.get("gap_ms", 0))):
            lines.append(
                f"| {row.get('gap_ms')} | {row.get('num_waves')} | "
                f"{usd(row.get('total_budget_needed_usd'))} | {usd(row.get('total_H_prod_usd'))} | "
                f"{usd(row.get('total_overshoot_prod_minus_needed_usd'))} |"
            )
        lines.append("")

    lines.append("## Coverage / Sanity Checks")
    lines.append("")
    waves = load_csv(out_dir / "bad_debt_loser_waves.csv")
    lines.append(f"- loser_waves_count: {len(waves)}")
    if not waves.empty and "deficit_usd" in waves.columns:
        lines.append(f"- total_deficit_usd: {usd(waves['deficit_usd'].sum())}")
    per_shock = load_csv(out_dir / "per_shock_metrics_queue.csv")
    lines.append(f"- queue_clusters: {len(per_shock)}")
    if not per_shock.empty and "coin" in per_shock.columns:
        lines.append(f"- queue_coins: {per_shock['coin'].nunique()}")
    lines.append("")

    lines.append("## Interpretation Warnings")
    lines.append("")
    lines.append("- Do not equate queue overshoot (equity USD) with production overshoot (PnL USD).")
    lines.append("- Queue metrics are per-coin; replay metrics are global time waves.")
    lines.append("- Overcollateralization is heavy-tailed; use trimmed summaries, not raw means.")
    lines.append("")

    lines.append("## Known Approximations / Limitations")
    lines.append("")
    lines.append("- Queue overshoot is per-coin; production replay uses global waves.")
    lines.append("- Two-pass replay fixes the realized price path; it is a counterfactual.")
    lines.append("- Position cost-basis is approximated; equity deltas are informative but not exact.")
    lines.append("- The replay does not model alternative trader behavior under different ADL rules.")
    lines.append("")

    lines.append("## Recommended edits to the paper")
    lines.append("")
    lines.append("### 1. Make the regime condition explicit (interpretation, not a redefinition)")
    lines.append("")
    lines.append(
        "Keep Assumption J.3 as written, but add a short interpretation note (see point `9`) below that ties "
        "$\\mu_\\Phi$ to a sustainable diversion rate (e.g., via an LTV sensitivity constraint), without "
        "redefining the assumption."
    )
    lines.append("")
    lines.append("Then state plainly: the trilemma is conditional; it is silent when $\\mu_- \\le \\mu_\\Phi$.")
    lines.append("")

    lines.append("### 2. Cleanly separate policy families")
    lines.append("")
    lines.append("- The queue analyzed in the theory is a wealth-space queue.")
    lines.append("- Hyperliquid's implementation is a contract-space queue that induces a haircut vector via equity differences.")
    lines.append("- Wealth-pro-rata and contracts-pro-rata are different mechanisms with different fairness meanings.")
    lines.append("")
    lines.append(
        "Without this separation, readers will conflate wealth-space theory results with contract-space "
        "implementation and misread empirical 'overshoot' as a production ledger claim."
    )
    lines.append("")

    lines.append("### 3. Add a short empirical regime check")
    lines.append("")
    lines.append(
        "If the empirical section invokes the structural deficit regime, add a short paragraph that estimates "
        "whether the event satisfies $\\mu_- \\gg \\mu_\\Phi$ for the relevant window."
    )
    lines.append("")
    lines.append("This check is an order-of-magnitude sanity test, not a precise identification of $\\mu_-$ or $\\mu_\\Phi$.")
    lines.append("")

    lines.append("### 4. Qualify the \"$653M\" headline as wealth-space overshoot")
    lines.append("")
    lines.append(
        "Add one sentence immediately after the first \"$653M\" claim in the abstract/intro, and the same "
        "qualifier after the \"$653.6M overshoot\" sentence in §9.2:"
    )
    lines.append("")
    lines.append(
        "> This figure is a wealth-space overshoot diagnostic computed under the paper's $\\mathrm{PnL}$-only "
        "haircut capacity definition. It is not a direct claim that this amount of collateral/principal could "
        "have been saved in production, because execution/settlement mechanics are not fully observable from "
        "public data under the stated observation model."
    )
    lines.append("")

    lines.append("### 5. Add a scope note in §2.5 (theory vs measurement)")
    lines.append("")
    lines.append("Insert a short paragraph right after the trilemma statement (Proposition 2.1):")
    lines.append("")
    lines.append(
        "> Scope note. The ADL trilemma is a statement about an abstract policy model (static ADL families under "
        "Assumptions J.1–J.3). It does not rely on any empirical \"two-party zero-sum fill\" identity or on "
        "observing a complete execution/settlement ledger. Empirical sections map public observations onto model "
        "objects; that mapping is separate from the theorem."
    )
    lines.append("")

    lines.append("### 6. Add a metric-agnostic observation-model note in §9.1")
    lines.append("")
    lines.append("At the start of §9.1, add:")
    lines.append("")
    lines.append(
        "> Observation model (public data). Our reconstruction uses publicly available fills, marks, and "
        "account-level aggregates. It is not a full execution/clearing ledger: we do not observe the complete "
        "settlement logic, internal netting, fees/buffers, or all timing details. This is sufficient to compute "
        "the paper's empirical objectives (deficits and winner $\\mathrm{PnL}$ capacity) under stated definitions; "
        "as a result, we treat strict accounting invariants as diagnostics rather than hard identities."
    )
    lines.append("")
    lines.append(
        "If §9.1 claims the dataset provides an “exact sequence enabling counterfactual analysis,” soften it to: "
        "we observe the realized sequence of fills/marks/account aggregates sufficient to define a replay under "
        "the public-data observation model."
    )
    lines.append("")
    lines.append(
        "Also soften phrases like “high-fidelity replay” to “reconstruction under partial observability” where "
        "appropriate."
    )
    lines.append("")

    lines.append("### 7. Make PnL-only capacity explicit (and align deficit vs capacity)")
    lines.append("")
    lines.append(
        "- Replace \"winner equity vector\" / \"maximum haircut mass\" with \"winner positive $\\mathrm{PnL}$ "
        "capacity vector\" / \"maximum $\\mathrm{PnL}$-only haircut capacity\" in §9."
    )
    lines.append("- Add one sentence immediately after the first capacity definition:")
    lines.append(
        "  > Here, \"capacity\" refers to positive $\\mathrm{PnL}$ (profits) available to be socialized; "
        "principal/collateral is not part of haircut capacity in this measurement model."
    )
    lines.append("- Add an alignment note in Appendix I (Account aggregates):")
    lines.append(
        "  > Alignment note. In this empirical model, deficit is computed from losers' negative equity, while "
        "capacity is computed from winners' positive $\\mathrm{PnL}$; this is a model choice and matches the "
        "corrected wealth-space haircut capacity definition used throughout §9."
    )
    lines.append("")

    lines.append("### 8. Add a diagnostic markout decomposition in Appendix I (not §9’s overshoot metric)")
    lines.append("")
    lines.append(
        "Add a short Appendix I subsection (between I.2 and I.3) titled “Diagnostic decomposition (not the §9 "
        "overshoot metric)”:"
    )
    lines.append("")
    lines.append(
        "> This subsection is a diagnostic for interpreting execution timing and for answering common \"markout\" "
        "questions. It is not the overshoot metric used in §9.  \n"
        ">  \n"
        "> Sign convention: $Q>0$ buys, $Q<0$ sells; cost is negative when the ADLed user is worse off.  \n"
        ">  \n"
        "> From the ADLed user’s perspective, define signed quantity, execution price, the nearest mark at "
        "execution time, and the mark at a chosen horizon later. Then the total markout vs execution decomposes "
        "into an immediate execution-vs-mark term plus an opportunity term from mark movement after forced "
        "flattening.  \n"
        ">  \n"
        "> Worked sign check: sell 10 at $5 and mark $\\Delta$ later $6$, then $Q=-10$ and "
        "$C_{\\text{tot}}=(-10)(6-5)=-10$ (a $10 loss under the sign convention).  \n"
        ">  \n"
        "> If desired, this diagnostic can be evaluated across a horizon sweep (e.g., 0ms, 0.5s, 1s, 2s, 5s...).  \n"
        ">  \n"
        "> Because this diagnostic combines executions with time-indexed marks and public-data replay lacks full "
        "clearing/settlement observability, its aggregate is not expected to satisfy a strict two-party zero-sum "
        "identity in raw data. Any residual should be interpreted as a combination of imperfect fill pairing, "
        "timestamp alignment effects, and potential system wedges (fees/buffers/insurance/bankruptcy rules)."
    )
    lines.append("")

    lines.append("### 9. Appendix J remarks after Theorem J.7 (observability + regime boundary)")
    lines.append("")
    lines.append("Add two short remarks immediately after Theorem J.7:")
    lines.append("")
    lines.append(
        "> Remark (interpretation of $\\mu_\\Phi$). In applications, $\\mu_\\Phi$ can be interpreted as the largest "
        "fee diversion rate that remains compatible with non-declining long-run venue value (e.g., via an LTV "
        "sensitivity constraint). This is an interpretation of the regime boundary, not a change to the formal "
        "assumption used in the proof.  \n"
        ">  \n"
        "> Remark (theorem vs observability). Theorem J.7 is proved in the policy model and does not assume any "
        "particular empirical observability of execution/settlement cashflows. Empirical tests necessarily operate "
        "under an observation model and should be interpreted as measurements of the paper's defined objects, not "
        "as complete ledger identities."
    )
    lines.append("")

    lines.append("### 10. Optional: rename §9 to preempt “ledger invariant” critiques")
    lines.append("")
    lines.append("Rename §9 to: “Empirical Analysis under Partial Observability: The October 10 Event.”")
    lines.append("")

    lines.append("### 11. Global guardrail for headline numbers (search-and-patch)")
    lines.append("")
    lines.append(
        "Action: in the ADL paper, search for `653`, `658`, `28x`, `28 x`, `28×` and apply the same "
        "wealth-space/PnL-only qualifier wherever these appear. Also search for `unnecessary` and `inefficiency`, "
        "those are often where the \"production savings\" implication sneaks back in without the numbers."
    )
    lines.append("")
    lines.append("Supplementary: a short Q&A addressing markout accounting and observability is in `docs/FAQs.md`.")
    lines.append("")

    lines.append("## Mermaid diagram: how the pieces fit")
    lines.append("")
    lines.append(
        "Diagram summary: maps the empirical inputs (fills/marks) to reconstructed deficits, induced haircuts, "
        "and the overshoot metric used in this report."
    )
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append("  A[Production event data<br/>positions, fills, prices] --> B[Wave decomposition t=1..T]")
    lines.append("  B --> C[Compute shortfall D_t from equities]")
    lines.append("  B --> D[Replay / counterfactual<br/>compute e_t no-ADL]")
    lines.append("  B --> E[Observed outcome<br/>compute e_t ADL]")
    lines.append("  D --> F[Induced haircuts h_t prod]")
    lines.append("  E --> F")
    lines.append("  F --> G[Winner extraction H_t prod]")
    lines.append("  C --> H[Comparator policy family<br/>compute H_t star]")
    lines.append("  G --> I[Overshoot E_t = H_t prod - H_t star]")
    lines.append("  I --> J[Aggregate metrics<br/>sum_t E_t, ratio kappa]")
    lines.append("```")
    lines.append("")

    lines.append("## Summary Tables and Outputs")
    lines.append("")
    lines.append("- Headline summary: `out/headlines.json`")
    lines.append("- Tables: `out/analysis_summary.md`")
    lines.append("- Queue detail: `out/per_shock_metrics_queue.csv`")
    lines.append("- Loser waves: `out/bad_debt_loser_waves.csv`")
    lines.append("- Horizon sweep: `out/eval_horizon_sweep_gap_ms=5000.csv`")
    lines.append("- Robustness bundle: `out/overshoot_robustness.json`")
    lines.append("- Two-time report: `out/two_time/report.md`")
    lines.append("")

    lines.append("## How to Regenerate")
    lines.append("")
    lines.append("1) Run the pipeline:")
    lines.append("")
    lines.append("```bash")
    lines.append("./run-all.sh          # uses bundled out/* data")
    lines.append("./run-all.sh --full   # regenerates from raw HyperReplay")
    lines.append("```")
    lines.append("")
    lines.append("2) Generate the corrected visuals + this report:")
    lines.append("")
    lines.append("```bash")
    lines.append("./scripts/visualize_corrections.sh")
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"[info] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
