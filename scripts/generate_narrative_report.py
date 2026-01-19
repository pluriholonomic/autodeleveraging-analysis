#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def usd(x: object) -> str:
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


def policy_overshoot_summary(out_dir: Path) -> List[tuple[str, float]]:
    df = load_csv(out_dir / "policy_per_wave_metrics.csv")
    if df.empty:
        return []
    col_map = [
        ("production (queue)", "overshoot_prod"),
        ("pro-rata", "overshoot_pr"),
        ("vector (mirror descent)", "overshoot_vector"),
        ("drift pro-rata (integer)", "overshoot_drift_pr_integer"),
        ("pro-rata ILP (integer)", "overshoot_pr_ilp_integer"),
        ("fixed-point ILP (integer)", "overshoot_fixed_point_ilp_integer"),
    ]
    rows: List[tuple[str, float]] = []
    for label, col in col_map:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        rows.append((label, float(vals.sum())))
    return rows


def image_block(lines: List[str], *, title: str, rel_path: Path, fig_label: str, desc: str) -> None:
    lines.append(f"### {title}")
    lines.append(f"![]({rel_path.as_posix()})")
    lines.append("")
    clean_title = title.split(". ", 1)[1] if ". " in title else title
    lines.append(f"Figure {fig_label}. {clean_title}")
    lines.append("")
    lines.append(desc)
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a full narrative ADL corrections report.")
    parser.add_argument("--out", default="out", help="Output directory (default: out)")
    parser.add_argument("--fig-dir", default=None, help="Figures directory (default: <out>/figures_corrections)")
    parser.add_argument(
        "--output",
        default=None,
        help="Report path (default: <out>/ADL-Corrections-Full-Report.md)",
    )
    parser.add_argument("--old-queue-overshoot", type=float, default=None, help="Optional legacy queue overshoot")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir / "figures_corrections"
    output_path = Path(args.output) if args.output else out_dir / "ADL-Corrections-Full-Report.md"

    headlines = load_json(out_dir / "headlines.json")
    h = headlines.get("headline", {})
    d = headlines.get("derived", {})
    oc = h.get("winner_overcollateralization_equity_over_pnl", {}) if isinstance(h, dict) else {}

    queue_overshoot = usd(h.get("queue_total_overshoot_usd", 0.0))
    prod_overshoot = usd(h.get("prod_overshoot_vs_needed_usd", 0.0))
    naive_expected = usd(d.get("naive_expected_pnl_closed_usd", 0.0))
    oc_trim = fmt_ratio(oc.get("trimmed_mean", "n/a"), 2)
    factor_fmt = fmt_ratio(d.get("factor_naive_expected_over_observed", "n/a"), 2)

    sweep = load_csv(out_dir / "eval_horizon_sweep_gap_ms=5000.csv")
    h0 = horizon_value(sweep, 0) if not sweep.empty else None
    h500 = horizon_value(sweep, 500) if not sweep.empty else None
    h1000 = horizon_value(sweep, 1000) if not sweep.empty else None
    latency_rows = compute_latency_summary(out_dir)
    policy_rows = policy_overshoot_summary(out_dir)

    def rel(name: str) -> Path:
        img_path = fig_dir / name
        try:
            return img_path.relative_to(out_dir)
        except ValueError:
            return img_path

    lines: List[str] = []
    lines.append("# ADL Corrections Full Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        "This note is a companion to two blog posts in this series "
        "([post 1](https://thogiti.github.io/2025/12/11/Autodeleveraging-Hyperliquid-653M-debate.html), "
        "[post 2](https://thogiti.github.io/2025/12/14/ADL-Trilemma-Dan-Critique-Tarun-paper-fixes.html)) "
        "and the related threads "
        "([x.com 1](https://x.com/tarunchitra/status/1998985133303701673), "
        "[x.com 2](https://x.com/tarunchitra/status/2001231023364038796), "
        "[x.com 3](https://x.com/tarunchitra/status/1998451762232177041)). It consolidates the technical "
        "corrections and the empirical methodology into one place so a reader can audit the assumptions, "
        "the unit conventions (contracts vs equity), and the meaning of the headline numbers (overshoot, "
        "\"\\$653m\", \"28x\", etc.)."
    )
    lines.append("")
    lines.append("The core message is narrow:")
    lines.append("")
    lines.append("1. Production ADL on venues like Hyperliquid is executed in contract space.")
    lines.append("2. The theoretical analysis in Tarun's paper is written in wealth space (equity haircuts).")
    lines.append("3. To compare production ADL to model ADL you need an explicit mapping.")
    lines.append(
        "4. Once the mapping is explicit, you can ask an apples-to-apples question: how much winner equity "
        "was removed by production ADL versus a comparator policy that covers the same deficit?"
    )
    lines.append("")
    lines.append("**Key headlines:**")
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
        lines.append(f"Note: headline uses horizon=0 ({usd(h0) if h0 is not None else 'n/a'}).")
        parts = []
        if h500 is not None:
            parts.append(f"500ms: {usd(h500)}")
        if h1000 is not None:
            parts.append(f"1000ms: {usd(h1000)}")
        lines.append(f"Horizon sweep examples: {', '.join(parts)}.")
        lines.append("")
    lines.append("**Sanity bound (naive inefficiency; no markouts / no impact).**")
    lines.append(
        "Readers often have a simple sanity-check model in mind: in each ADL wave, the deficit proxy "
        "$B_{\\text{needed}}$ is matched 1-to-1 against winner PnL at the clearing price, with no markout "
        "horizon and no market impact model. In our framework, that is exactly the horizon = 0 construction."
    )
    lines.append("")
    lines.append(
        f"Under that naive model, the lower bound on inefficiency is \\$0, and the upper bound is the "
        f"horizon = 0 overshoot vs the deficit proxy: {prod_overshoot}."
    )
    lines.append("")
    lines.append(
        "Figure 03 shows how overshoot changes when you choose a non-zero evaluation horizon "
        "(a scenario choice, not statistical robustness)."
    )
    lines.append("")

    lines.append("## Glossary / Thesaurus (paper vs debate vs code)")
    lines.append("")
    lines.append(
        "This note involves three overlapping vocabularies. The same words (\"haircut\", \"capacity\", "
        "\"overshoot\", \"queue\") can mean different objects depending on whether we are talking about: "
        "(i) the abstract model in the paper, (ii) production ADL implementations, or (iii) the "
        "reconstruction/measurement model in this repo."
    )
    lines.append("")
    lines.append("### A) Spaces and their primitives")
    lines.append("")
    lines.append("**Wealth space (paper / theory).**")
    lines.append("")
    lines.append("- State variable: equity per account/position, $e_{i,t}$.")
    lines.append("- Equity decomposition: $e_{i,t} = c_{i,t} + \\mathrm{PnL}_{i,t}$.")
    lines.append(
        "- ADL action: directly reduces winner wealth via a haircut fraction $h_i$ applied to a chosen base "
        "(paper: often $(e_{i,t})_+$; corrected model: $(\\mathrm{PnL}_{i,t})_+$)."
    )
    lines.append("")
    lines.append("**Contract space (production / engine).**")
    lines.append("")
    lines.append("- State variable: contracts/position size $q_{i,t}$ and collateral/margin rules.")
    lines.append(
        "- ADL action: forces position reductions $\\Delta q_{i,t}$ based on a ranking (often a queue) and "
        "executes trades at some execution price under matching/settlement rules."
    )
    lines.append("")
    lines.append("**Observation space (public-data reconstruction / this repo).**")
    lines.append("")
    lines.append("- Observables: fills, marks, and account aggregates from public endpoints.")
    lines.append(
        "- Missing: full internal clearing/settlement logic, netting, fee/buffer application, exact "
        "timing/priority details."
    )
    lines.append("- \"Haircut\" is inferred from equity differences under a replay, not directly observed as a ledger transfer.")
    lines.append("")
    lines.append("### B) Core objects and \"same word, different meaning\"")
    lines.append("")
    lines.append("#### 1) Deficit / shortfall / bad debt")
    lines.append("")
    lines.append("**Paper (wealth space):**")
    lines.append("")
    lines.append("- Deficit (bad debt) is loser-side negative equity:")
    lines.append("")
    lines.append("$$")
    lines.append("D_t = \\sum_i (-e_{i,t})_+.")
    lines.append("$$")
    lines.append("")
    lines.append("**Production (contract space):**")
    lines.append("")
    lines.append("- Deficit is whatever the clearing engine cannot collect after liquidation(s), including settlement rules, buffers, and insurance behavior.")
    lines.append("")
    lines.append("**Repo (observation model):**")
    lines.append("")
    lines.append("- Deficit waves are reconstructed from loser-side equity aggregates (public data).")
    lines.append("")
    lines.append("#### 2) Haircut capacity / winner resources")
    lines.append("")
    lines.append("**Paper (original phrasing):**")
    lines.append("")
    lines.append("- Often described as winner equity mass, $W_t = \\sum_i (e_{i,t})_+$.")
    lines.append("")
    lines.append("**Corrected / mapping model (repo + fixes):**")
    lines.append("")
    lines.append("- Haircut capacity is positive PnL only:")
    lines.append("")
    lines.append("$$")
    lines.append("U_t = \\sum_i (\\mathrm{PnL}_{i,t})_+.")
    lines.append("$$")
    lines.append("")
    lines.append("- Interpretation: in the mapping model, ADL seizes profits, not principal/collateral.")
    lines.append("")
    lines.append("**Why this matters:**")
    lines.append("")
    lines.append("- Under PnL-only capacity, $W_t > D_t$ does not imply the deficit is coverable in one round.")
    lines.append("- The correct feasibility check is $U_t \\ge \\theta D_t$ (or $U_t \\ge D_t$ for full coverage).")
    lines.append("")
    lines.append("#### 3) Queue")
    lines.append("")
    lines.append("**Paper (wealth-space queue):**")
    lines.append("")
    lines.append("- A selection rule over winners in wealth space (who gets haircutted and by how much), operating on an endowment vector (originally $(e_{i,t})_+$, corrected to $(\\mathrm{PnL}_{i,t})_+$).")
    lines.append("")
    lines.append("**Production (contract-space queue):**")
    lines.append("")
    lines.append("- A ranking of positions/accounts by some score (PnL, leverage, etc.), determining which contracts are reduced first. The resulting equity impact depends on price/execution and on how much equity is tied to the closed contracts.")
    lines.append("")
    lines.append("**Repo usage:**")
    lines.append("")
    lines.append("- \"Queue\" always needs a qualifier: wealth-space queue (theory comparator) vs contract-space queue (observed implementation).")
    lines.append("")
    lines.append("#### 4) Overshoot / unnecessary haircuts")
    lines.append("")
    lines.append("**Paper (wealth space, as an abstract diagnostic):**")
    lines.append("")
    lines.append("- Overshoot can mean excess extraction relative to a comparator policy for the same deficit.")
    lines.append("")
    lines.append("**Repo (two distinct metrics):**")
    lines.append("")
    lines.append("- Wealth-space overshoot (equity USD): a diagnostic comparing induced wealth removed under observed ADL to a comparator wealth-space policy under the same deficit definition.")
    lines.append("- Production overshoot vs needed (PnL USD, horizon=0): a reconstruction objective that compares observed PnL seized vs a deficit proxy $B_{\\text{needed}}$ under the horizon=0 (no-markout) construction.")
    lines.append("")
    lines.append("**Important guardrail:**")
    lines.append("")
    lines.append("- Any headline like USD 653m overshoot is a wealth-space diagnostic under a PnL-only capacity definition, not a direct statement that collateral/principal was \"saved\" in production.")
    lines.append("")
    lines.append("#### 5) Markout / opportunity cost")
    lines.append("")
    lines.append("**Not used as the overshoot metric in this report.**")
    lines.append("")
    lines.append("- Markout-style costs compare execution to a mark price at horizon $\\Delta$: $Q(M_{\\Delta} - P_{\\text{exec}})$.")
    lines.append("- This is a diagnostic tool and is not expected to be perfectly zero-sum in public data due to pairing/timestamp mismatch and possible system wedges (fees/buffers/insurance/bankruptcy rules).")
    lines.append("")
    lines.append("### C) Quick mapping table (where to look in code)")
    lines.append("")
    lines.append("- Loser deficit waves ($D_t$): `src/oss_adl/bad_debt.py`")
    lines.append("- PnL-only capacity ($U_t$): `src/oss_adl/queue_overshoot.py`")
    lines.append("- Two-pass replay haircuts / induced wealth removed: `src/oss_adl/two_pass_replay.py`")
    lines.append("- Policy comparisons (pro-rata / mirror descent / etc.): `policy_per_wave_metrics.csv`")
    lines.append("- Markout diagnostic Q&A: `docs/FAQs.md`")
    lines.append("")

    lines.append("## Background and problem statement")
    lines.append("")
    lines.append("### What ADL is doing")
    lines.append("")
    lines.append(
        "When liquidations fail (latency, oracle jumps, thin liquidity, adversarial orderflow), the system "
        "can end up with a shortfall: some accounts have negative equity that cannot be collected. ADL is "
        "one way to restore solvency by transferring wealth from solvent accounts to cover that deficit."
    )
    lines.append("")
    lines.append("We will use the following wealth-space aggregates over a position set $P_n$ at a wave/end time:")
    lines.append("")
    lines.append("Total shortfall (bad debt):")
    lines.append("")
    lines.append("$$")
    lines.append("D_T(P_n) = \\sum_{p \\in P_n} \\bigl(-e_T(p)\\bigr)_+.")
    lines.append("$$")
    lines.append("")
    lines.append("Total winner equity:")
    lines.append("")
    lines.append("$$")
    lines.append("W_T(P_n) = \\sum_{p \\in P_n} e_T(p)^+.")
    lines.append("$$")
    lines.append("")
    lines.append("### The contracts-vs-wealth mismatch")
    lines.append("")
    lines.append("A lot of the public disagreement came from mixing two different spaces:")
    lines.append("")
    lines.append("- Contract space (production): the engine selects positions by a ranking score and forces contract-level reductions.")
    lines.append("- Wealth space (theory): a policy selects haircut fractions $h(p) \\in [0,1]$ and directly reduces winner equities.")
    lines.append("")
    lines.append(
        "A close in contracts is not the same statement as haircut 100% of equity. The economic effect depends on "
        "prices and on how much of the account's equity is attributable to the seized contracts."
    )
    lines.append("")
    lines.append("### What this report is correcting")
    lines.append("")
    lines.append("This note is focused on three corrections and clarifications that make the model-to-production comparison well-defined:")
    lines.append("")
    lines.append("1. Queue interpretation: treat the queue in the theory as a wealth-space abstraction.")
    lines.append("2. Pro-rata definition: separate wealth-pro-rata from contracts-pro-rata.")
    lines.append("3. The headline number: define excess haircuts as an equity-dollar difference against a benchmark.")
    lines.append("")

    lines.append("## Corrections implemented")
    lines.append("")
    lines.append("- Queue overshoot capacity now uses PnL-only (`position_unrealized_pnl`), not equity + principal.")
    lines.append("  - Code: `src/oss_adl/queue_overshoot.py:74`")
    lines.append("- Queue overshoot deficit proxy is loser-side, not winner equity.")
    lines.append("  - Code: `src/oss_adl/queue_overshoot.py:179`, `src/oss_adl/queue_overshoot.py:292`")
    lines.append("- Production overshoot is reconstructed with two-pass replay (ADL-on vs ADL-off on the same price path).")
    lines.append("  - Code: `src/oss_adl/two_pass_replay.py:14`, `src/oss_adl/two_pass_replay.py:423`")
    lines.append("- Loser deficit waves use loser-side equity and global time clustering.")
    lines.append("  - Code: `src/oss_adl/bad_debt.py:30`, `src/oss_adl/bad_debt.py:80`")
    lines.append("")

    lines.append("## Methodology map from paper to analysis code")
    lines.append("")
    lines.append("- Queue overshoot (wealth-space queue): `methodology.md` section 3.7; `src/oss_adl/queue_overshoot.py`.")
    lines.append("- Loser deficit waves $D_t$: `methodology.md` section 3.2; `src/oss_adl/bad_debt.py`.")
    lines.append("- Needed budget $B_{\\text{needed}}$: `methodology.md` section 3.3; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Production haircut $H_{\\text{prod}}$: `methodology.md` section 3.4; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Horizon sweep $\\Delta$: `methodology.md` section 3.6; `src/oss_adl/two_pass_replay.py`.")
    lines.append("- Winner overcollateralization: `methodology.md` section 3.8; `src/oss_adl/queue_overshoot.py`.")
    lines.append("- Two-time behavior (undo fraction): `methodology.md` section 3.9; `src/oss_adl/two_time_behavior.py`.")
    lines.append("")

    lines.append("## Visualizations and what each is showing")
    lines.append("")
    image_block(
        lines,
        title="00. Overview Tiles",
        rel_path=rel("00_overview_tiles.png"),
        fig_label="00",
        desc="Quick read of the three headline quantities and the naive expected PnL closed.",
    )
    image_block(
        lines,
        title="01. Definition Diff (the core fix)",
        rel_path=rel("01_definition_diff.png"),
        fig_label="01",
        desc="Queue capacity must be PnL-only, not equity + principal.",
    )
    image_block(
        lines,
        title="02. Mapping Bars (why the gap exists)",
        rel_path=rel("02_mapping_bars.png"),
        fig_label="02",
        desc="Naive equity-to-PnL mapping still overpredicts observed PnL overshoot.",
    )
    image_block(
        lines,
        title="03. Horizon Robustness",
        rel_path=rel("03_horizon_robustness.png"),
        fig_label="03",
        desc=(
            "Overshoot vs evaluation horizon. This is a counterfactual choice, not statistical robustness; "
            "the slope reflects time sensitivity."
        ),
    )
    image_block(
        lines,
        title="04. Queue Concentration",
        rel_path=rel("04_queue_concentration.png"),
        fig_label="04",
        desc="Shows whether queue overshoot is broad-based or dominated by a few coins.",
    )
    image_block(
        lines,
        title="05. Two-Time Scatter (strategic vs passive)",
        rel_path=rel("05_two_time_scatter.png"),
        fig_label="05",
        desc="Behavioral classification using undo fraction vs trading share.",
    )
    image_block(
        lines,
        title="06. Two-Time Quadrant Shares",
        rel_path=rel("06_two_time_quadrant_shares.png"),
        fig_label="06",
        desc="User-weighted quadrant shares.",
    )
    image_block(
        lines,
        title="07. Two-Time Quadrant Shares (Volume-weighted)",
        rel_path=rel("07_two_time_quadrant_shares_volume.png"),
        fig_label="07",
        desc="Volume-weighted quadrant shares (strategic share can dominate by volume).",
    )

    lines.append("### 08. Response-Time Proxy (strategic vs passive)")
    lines.append("")
    lines.append("Latency is computed from per-event time to first non-ADL fill (60s window).")
    lines.append("")
    lines.append("| cohort | n_users | n_events | median | p10 | p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if latency_rows:
        for row in latency_rows:
            lines.append(
                f"| {row['cohort']} | {fmt_int(row['n_users'])} | {fmt_int(row['n_events'])} | "
                f"{fmt_latency(row['median_ms'])} | {fmt_latency(row['p10_ms'])} | {fmt_latency(row['p90_ms'])} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")

    lines.append(
        "Per-symbol comparisons below use a PnL-closed proxy: the per-coin needed budget from ADL fill gaps "
        "($|\\mathrm{markPx}-\\mathrm{execPx}|\\times|\\mathrm{size}|$) at horizon=0."
    )
    lines.append("")

    image_block(
        lines,
        title="09. Equity vs PnL Proxy (per symbol)",
        rel_path=rel("08_equity_vs_needed_scatter.png"),
        fig_label="09",
        desc="Each point is a symbol; color encodes non-ADL volume (liquidity proxy).",
    )
    image_block(
        lines,
        title="10. Gap Ratio vs Liquidity",
        rel_path=rel("09_gap_ratio_vs_liquidity.png"),
        fig_label="10",
        desc="Weak/no trend indicates volume alone does not explain the gap.",
    )
    image_block(
        lines,
        title="11. Largest Equity/PnL Gaps",
        rel_path=rel("10_gap_ratio_bars.png"),
        fig_label="11",
        desc="Largest equity/PnL proxy gaps (filtered to avoid tiny denominators).",
    )
    image_block(
        lines,
        title="12. Undo Fraction vs Impact Proxy (per coin)",
        rel_path=rel("12_undo_vs_impact_per_coin.png"),
        fig_label="12",
        desc="Per-coin undo fraction vs impact proxy (gap per contract). Noisy in aggregate; diagnostic per coin.",
    )
    lines.append(
        "These per-asset gaps are relevant for considering asset-specific ADL mechanisms based on liquidity."
    )
    lines.append("")

    lines.append("## Definitions used in this report")
    lines.append("")
    lines.append("### Positions and equity")
    lines.append("")
    lines.append("A position is modeled as")
    lines.append("")
    lines.append("$$")
    lines.append("p = (q, c, t, b),")
    lines.append("$$")
    lines.append("")
    lines.append("where $q$ is signed base contracts, $c$ is collateral, $t$ is entry time, and $b$ is side.")
    lines.append("Equity at time $T$ is")
    lines.append("")
    lines.append("$$")
    lines.append("e_T(p) = c + \\mathrm{PnL}_T(p).")
    lines.append("$$")
    lines.append("")

    lines.append("### Wealth-space ADL policies")
    lines.append("")
    lines.append("A wealth-space ADL policy $\\pi$ is represented by:")
    lines.append("")
    lines.append("- a severity parameter $\\theta^\\pi \\in [0,1]$ choosing a budget $B = \\theta^\\pi D_T(P_n)$, and")
    lines.append("- a haircut fraction $h^\\pi(p) \\in [0,1]$ for each winner with $e_T(p) > 0$.")
    lines.append("")
    lines.append("Post-ADL equity is")
    lines.append("")
    lines.append("$$")
    lines.append("e_T^{\\mathrm{post}}(p) = (1 - h^\\pi(p)) e_T(p) \\quad \\text{for } e_T(p) > 0.")
    lines.append("$$")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("### 1. Wave decomposition")
    lines.append("")
    lines.append("The event is treated as a sequence of ADL waves indexed by $t = 1,2,\\dots,T$.")
    lines.append("")
    lines.append("For each wave $t$, we identify the pre-wave position set $P_n^{(t)}$, and compute the shortfall:")
    lines.append("")
    lines.append("$$")
    lines.append("D_t = \\sum_{p \\in P_n^{(t)}} \\bigl(-e_t(p)\\bigr)_+.")
    lines.append("$$")
    lines.append("")

    lines.append("### 2. Induced wealth-space haircuts from production ADL")
    lines.append("")
    lines.append("To compare production ADL to wealth-space policies, we define two equity outcomes for each position on the same realized price path:")
    lines.append("")
    lines.append("- $e_t^{\\mathrm{no\\text{-}ADL}}(p)$: counterfactual where ADL is disabled in wave $t$ while holding the realized price path fixed.")
    lines.append("- $e_t^{\\mathrm{ADL}}(p)$: realized equity after production ADL in wave $t$.")
    lines.append("")
    lines.append("Define the induced haircut:")
    lines.append("")
    lines.append("$$")
    lines.append("h_t^{\\mathrm{prod}}(p) =")
    lines.append("\\begin{cases}")
    lines.append("\\dfrac{e_t^{\\mathrm{no\\text{-}ADL}}(p) - e_t^{\\mathrm{ADL}}(p)}{e_t(p)}, & e_t(p) > 0, \\\\[6pt]")
    lines.append("0, & \\text{otherwise}.")
    lines.append("\\end{cases}")
    lines.append("$$")
    lines.append("")
    lines.append("Total wealth removed from winners in wave $t$ is")
    lines.append("")
    lines.append("$$")
    lines.append("H_t^{\\mathrm{prod}} = \\sum_{p \\in P_n^{(t)}} \\bigl(e_t^{\\mathrm{no\\text{-}ADL}}(p) - e_t^{\\mathrm{ADL}}(p)\\bigr)_+.")
    lines.append("$$")
    lines.append("")

    lines.append("### 3. Comparator policies and overshoot")
    lines.append("")
    lines.append("Let $H_t^{\\star}$ be the winner-equity extraction under a comparator policy family that covers the same deficit.")
    lines.append("")
    lines.append("Define overshoot (excess haircut) in wave $t$:")
    lines.append("")
    lines.append("$$")
    lines.append("E_t = H_t^{\\mathrm{prod}} - H_t^{\\star}.")
    lines.append("$$")
    lines.append("")
    lines.append("Aggregate overshoot:")
    lines.append("")
    lines.append("$$")
    lines.append("E_{\\mathrm{total}} = \\sum_{t=1}^T E_t.")
    lines.append("$$")
    lines.append("")

    lines.append("### 4. Interpretation of the \\$653m figure")
    lines.append("")
    lines.append("When the analysis reports an excess haircuts number like \\$653m, the intended meaning is:")
    lines.append("")
    lines.append("*It is the sum of equity-dollar differences between production ADL and a benchmark policy that covers the same deficits.*")
    lines.append("")
    lines.append("Formally:")
    lines.append("")
    lines.append("$$")
    lines.append("\\$653\\text{m} \\equiv \\sum_{t=1}^T \\bigl(H_t^{\\mathrm{prod}} - H_t^{\\star}\\bigr).")
    lines.append("$$")
    lines.append("")
    lines.append("This is not the notional value of positions closed.")
    lines.append("")

    lines.append("## Results summary")
    lines.append("")
    lines.append("### Mechanism-design takeaway (separate from price-path choices)")
    lines.append("")
    lines.append("There are two different knobs in this report, and they should not be conflated:")
    lines.append("")
    lines.append(
        "1. Policy-dependent (mechanism design): who you select to bear the haircut in each wave "
        "(queue vs pro-rata vs leverage-weighted variants), holding the same realized event data and the "
        "same deficit proxy definition fixed."
    )
    lines.append(
        "2. Path-dependent (scenario choice): how you define the evaluation window for markouts / opportunity "
        "cost (the horizon sweep in Figure 03)."
    )
    lines.append("")
    lines.append(
        "The cleanest, least controversial takeaway is the policy ordering in (1): under this reconstruction "
        "objective (overshoot/residual in equity space), queue-style selection concentrates haircuts, while "
        "leverage-weighted or waterfilling-style rules spread the burden. This is a ranking under the "
        "overshoot/residual metrics, not a full welfare ranking under execution constraints."
    )
    lines.append("")
    lines.append(
        "You can read that ordering directly from `policy_per_wave_metrics.csv` by comparing total overshoot "
        "vs needed across policy families with the evaluation horizon fixed."
    )
    lines.append("")
    if policy_rows:
        lines.append("Policy ordering (total overshoot vs needed, horizon fixed):")
        lines.append("")
        lines.append("| policy | definition | total_overshoot_usd |")
        lines.append("| --- | --- | --- |")
        policy_defs = {
            "production (queue)": "Observed production ADL (queue-based selection).",
            "pro-rata": "Wealth-proportional allocation (equity pro-rata).",
            "vector (mirror descent)": "Vector mirror-descent allocation toward budget.",
            "drift pro-rata (integer)": "Pro-rata with integer rounding (drift-based).",
            "pro-rata ILP (integer)": "Integer program for pro-rata targets.",
            "fixed-point ILP (integer)": "Integer program for fixed-point targets.",
        }
        for label, value in sorted(policy_rows, key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"| {label} | {policy_defs.get(label, 'n/a')} | {usd(value)} |")
        lines.append("")
        lines.append("Negative values indicate undershoot (haircuts below the deficit proxy).")
        lines.append("")
    lines.append("Once you compute $H_t^{\\mathrm{prod}}$ using induced equity losses, the empirical questions become well-posed:")
    lines.append("")
    lines.append("1. Does production ADL concentrate haircuts on a small set of winners relative to a pro-rata benchmark?")
    lines.append("2. Does production ADL overshoot the deficit coverage implied by the comparator family, and by how much?")
    lines.append("3. How sensitive are these conclusions to the counterfactual definition of $e_t^{\\mathrm{no\\text{-}ADL}}$?")
    lines.append("")

    lines.append("## Gaps and what is still missing")
    lines.append("")
    lines.append("1. Counterfactual dependence: $e_t^{\\mathrm{no\\text{-}ADL}}$ is not directly observable.")
    lines.append("2. Matching-engine opacity: without internal logs we cannot separate selection from execution with perfect fidelity.")
    lines.append("3. Unit consistency: any substitution of notional contracts for equity dollars inflates headline numbers.")
    lines.append(
        "4. Markout metric: this cost measure is not a two-party cashflow identity, so it will not be exactly "
        "zero-sum in raw data under imperfect fill pairing, time-indexed marks, and potential bankruptcy/insurance "
        "wedges."
    )
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

    lines.append("### 4. Qualify the \"\\$653M\" headline as wealth-space overshoot")
    lines.append("")
    lines.append(
        "Add one sentence immediately after the first \"\\$653M\" claim in the abstract/intro, and the same "
        "qualifier after the \"\\$653.6M overshoot\" sentence in §9.2:"
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
        "> Worked sign check: sell 10 at USD 5 and mark $\\Delta$ later USD 6, then $Q=-10$ and "
        "$C_{\\text{tot}}=(-10)(6-5)=-10$ (a USD 10 loss under the sign convention).  \n"
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

    lines.append("### 10. Minimal theorem corrections for the theorems in the paper")
    lines.append("")
    theorem_block = r"""#### §2: Background

- **Make the haircut numeraire explicit (cash vs PnL).** Keep equity as $e_{i,t}=c_{i,t}+\mathrm{PnL}_{i,t}$, but state clearly that the empirical/mapping model in this repo treats **ADL haircuts as seizing positive PnL capacity only** (principal/collateral is returned on forced close, and it is modeled as protected and policy-independent under the public-data observation model, so it is not "haircut capacity").
  > In the empirical mapping used here, $\mathrm{PnL}_{i,t}$ denotes unrealized PnL at the venue's mark price / the repo's `position_unrealized_pnl` variable at time $t$.
- **Deficit stays loser-side negative equity.** Keep $D_t=\sum_i(-e_{i,t})_+$, and explicitly connect it to the loser-side construction (liquidated/insolvent accounts; see the repo's "loser deficit waves").
- **Introduce PnL-only winner capacity objects (replace equity-based capacity).** Anywhere the paper uses "winner equity mass" as the haircutable resource, replace it with:

  $$
  U_t = \sum_i (\mathrm{PnL}_{i,t})_+ \text{ (total positive PnL capacity)}
  $$

  $$
  \upsilon_t = \max_i (\mathrm{PnL}_{i,t})_+ \text{ (top winner positive PnL).}
  $$

  This is the minimal change needed for "haircuts in PnL space" consistency.
- **Update the ADL policy definition (haircut applies to $(\mathrm{PnL})_+$, not $e_+$).** Replace the post-policy update
  $$
  e_i \leftarrow (1-h_i)e_i
  $$
  with the PnL-only update
  $$
  \mathrm{PnL}_i \leftarrow \mathrm{PnL}_i - h_i(\mathrm{PnL}_i)_+,
  \qquad
  e_i \leftarrow c_i + \mathrm{PnL}_i.
  $$
- **Update budget balance + feasibility constraints accordingly.** Replace $e_t(\cdot)_+$ with $(\mathrm{PnL}_t(\cdot))_+$ in the policy constraints:
  $$
  \sum_i h_i(\mathrm{PnL}_{i,t})_+ \;=\; \theta D_t,
  \qquad
  \theta D_t \le U_t.
  $$
  (Feasibility should not be stated in terms of $W_t=\sum e_+$ under PnL-only capacity.)
  > Under PnL-only capacity, equity solvency $W_t-D_t>0$ does **not** imply one-round coverability; there is an intermediate regime $D_t < W_t$ but $D_t > U_t$ ("solvent-but-not-coverable" without touching principal).
- **Queue / pro-rata formulas:** In the queue and pro-rata examples, replace every instance of "haircut times winner equity" with "haircut times winner $(\mathrm{PnL})_+$" (scores/rankings can still be computed from leverage and realized/closed PnL; the key is the haircut base).

#### §3: Risk + fairness preliminaries

- Any fairness/risk metric described as "winner equity" should be interpreted as "winner profit" in the PnL-only model, i.e. based on $(\mathrm{PnL})_+$. Deficits $D_t$ remain computed from loser negative equity.
- PTSR/PMR and Schur/majorization comparisons are unchanged after the substitution "positive equity vector $\to$ positive PnL vector," since the proofs rely on budget-balance and convex/ordering arguments, not on the specific decomposition $e=c+\mathrm{PnL}$.
- If one evaluates fairness on total equity $e=c+u$ with heterogeneous $c_i$, affine shifts can change some orderings; the clean statement is that fairness/majorization claims apply to the haircutable endowment $u=(\mathrm{PnL})_+$.

#### §4: Severity optimization

- Replace the "maximum haircut mass" cap $\sum_i (e_{i,t})_+$ with $\sum_i (\mathrm{PnL}_{i,t})_+$ wherever it appears as the feasibility constraint.
- Severity/separation identities that depend only on the scalar budget $\theta D_t$ (e.g. $R_t=(1-\theta)_+D_t$) are unchanged under the PnL-only haircut base.

#### §5: Negative results

- Interpret $\omega_t$ (top winner) and the EV scale $b_n$ as extreme values of **positive PnL**, not of positive equity. All EVT-style bounds whose inputs are "(i) budget balance, (ii) maxima/means scaling" go through verbatim under this substitution.

#### §6: Fairness

- Apply the fairness axioms/optimality to haircuts on $(\mathrm{PnL})_+$ (and caps interpreted as limits on PnL haircuts), not to total equity. Capped pro-rata remains the unique convex-optimal rule under per-account caps after this substitution.

#### §7: Risk-aware policies (RAP)

- Keep insolvency modeling in equity (since insolvency is an equity concept), but treat the control variable as a haircut on $(\mathrm{PnL})_+$ that moves equity by the same dollar amount while leaving cash untouched. Replace any use of $(e_{i,t})_+$ as available haircut budget/capacity with $(\mathrm{PnL}_{i,t})_+$.

#### Do any of the theorems actually change?

In substance, **most proofs and qualitative orderings do not change**: they are statements about (i) a nonnegative winner endowment vector, (ii) a scalar budget $H=\theta D$, and (iii) how different allocation rules distribute that budget (concentration/majorization).

What *does* change is mostly **the statement of the objects**:

- **Mechanical substitution (most theorems):** replace the "winner equity" endowment $(e_t)_+$ by the "winner profit" endowment $(\mathrm{PnL}_t)_+$ everywhere it appears as haircut capacity. The same arguments go through with the substituted vector.
- **Proof sketch (cash as an affine shift).** Write each winner's equity as
  $$
  e_i \;=\; c_i + u_i,\qquad u_i = (\mathrm{PnL}_i)_+,
  $$
  where $c_i\ge 0$ is the protected cash/principal component and $u_i$ is the haircutable profit endowment.
  A PnL-only ADL action is equivalently an allocation of seized amounts $x_i\in[0,u_i]$ satisfying $\sum_i x_i = H$ (where $H=\theta D$ is the chosen budget), yielding post-ADL equities
  $$
  e_i' \;=\; c_i + (u_i - x_i).
  $$
  Since the cash vector $c$ is policy-independent in this model, comparing two policies $A,B$ cancels it:
  $$
  e^{\prime A}-e^{\prime B} \;=\; (u-x^A)-(u-x^B).
  $$
  Thus any theorem whose proof uses only (i) a budget identity in the haircutable endowment ($\sum_i x_i=H$ or $\sum_i u_i h_i = H$), and (ii) ordering/convexity/majorization properties of the survivor vector $u-x$ (or $u\odot(1-h)$), is unchanged after substituting the endowment vector from $(e_t)_+$ to $(\mathrm{PnL}_t)_+$.
  If in addition $c_i\equiv C$ is common across winners, then adding $C\mathbf 1$ preserves majorization/orderings, so equity-level fairness orderings are identical as well.
- **Real semantic change (coverability vs solvency):** under PnL-only haircuts, $\mathsf{Solv}_T=W_T-D_T>0$ does **not** imply the deficit is coverable by ADL. The correct one-round feasibility condition is $D_T \le U_T=\sum (\mathrm{PnL}_T)_+$. There is an intermediate regime $D_T<W_T$ but $D_T>U_T$ where the exchange is solvent in gross equity but cannot clear the deficit without touching principal.
- **RAP section (if kept equity-leverage-based):** RAP derivations written with multiplicative equity updates $e\mapsto (1-h)e$ should either (a) reinterpret $e$ in that section as the haircutable profit endowment, or (b) be re-derived under the additive PnL-only update $e=c+\mathrm{PnL}-h(\mathrm{PnL})_+$. The high-level "tilt haircuts toward higher-risk winners" conclusion is unchanged, but the exact algebraic form depends on which choice is made.

#### §8: Stackelberg + PoA ratios

Most theoretical results outside §8 use linear/additive haircut updates, so the PnL-only correction does not change their statements.

Section 8 is the exception: it includes a Stackelberg equilibrium (non-additive) and a PoA ratio that divides by equilibrium values. The ratio can be written as:

$$
\frac{C+\mathrm{PnL}_{\text{worst}}}{C+\mathrm{PnL}_{\text{avg}}}
\quad \text{or} \quad
\frac{\mathrm{PnL}_{\text{worst}}}{\mathrm{PnL}_{\text{avg}}}.
$$

**Remark (PoA ratio stability).** Assume the cash baseline $C$ is invariant across the equilibria being compared and $\min(|\mathrm{PnL}_{\text{worst}}|,|\mathrm{PnL}_{\text{avg}}|)\ge \varepsilon>0$ (equivalently, $C/\mathrm{PnL}_{\text{worst}}$ and $C/\mathrm{PnL}_{\text{avg}}$ are bounded). Then

$$
\frac{C+\mathrm{PnL}_{\text{worst}}}{C+\mathrm{PnL}_{\text{avg}}}=\frac{\mathrm{PnL}_{\text{worst}}}{\mathrm{PnL}_{\text{avg}}}\cdot\frac{1+C/\mathrm{PnL}_{\text{worst}}}{1+C/\mathrm{PnL}_{\text{avg}}}.
$$

Under this condition, the multiplicative factor is $O(1)$, so PoA ordering/bounds are preserved up to constants bounded by functions of $\varepsilon$ and $C$. If either equilibrium PnL is near zero, the ratio becomes ill-conditioned; treat that as a degenerate regime.
"""
    for line in theorem_block.splitlines():
        lines.append(line)
    lines.append("")

    lines.append("### 11. Optional: rename §9 to preempt “ledger invariant” critiques")
    lines.append("")
    lines.append("Rename §9 to: “Empirical Analysis under Partial Observability: The October 10 Event.”")
    lines.append("")

    lines.append("### 12. Global guardrail for headline numbers (search-and-patch)")
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
    mermaid_img = fig_dir / "ADL-analysis-how-pieces-fit-together.png"
    try:
        mermaid_rel = mermaid_img.relative_to(out_dir)
    except ValueError:
        mermaid_rel = mermaid_img
    if mermaid_img.exists():
        lines.append(f"![]({mermaid_rel.as_posix()})")
        lines.append("")
        lines.append("Figure M1. Static diagram for PDF export.")
        lines.append("")
    else:
        lines.append(f"_Missing image: {mermaid_rel.as_posix()}_")
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

    lines.append("## Appendix: mapping close in contracts to haircut in equity")
    lines.append("")
    lines.append(
        "A contract-space close is a statement about $\\Delta q_t(p)$. A wealth-space haircut is a statement "
        "about $h_t(p)$. The induced-haircut definition makes this precise: it treats a close as relevant only "
        "insofar as it changes equity under the realized path, i.e., changes $e_t^{\\mathrm{ADL}}(p)$ relative "
        "to $e_t^{\\mathrm{no\\text{-}ADL}}(p)$."
    )
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
