#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.use("Agg")


def usd_short(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}${v/1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v/1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}${v/1e3:.2f}K"
    return f"{sign}${v:.2f}"


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_coin_gap_inputs(out_dir: Path) -> Optional[pd.DataFrame]:
    queue = load_csv(out_dir / "per_shock_metrics_queue.csv")
    if queue is None or queue.empty or "overshoot_queue_usd" not in queue.columns:
        return None
    equity = (
        queue.groupby("coin", as_index=False)["overshoot_queue_usd"]
        .sum()
        .rename(columns={"overshoot_queue_usd": "equity_overshoot_usd"})
    )

    needed_path = out_dir / "eval_horizon_sweep" / "gap_ms=5000" / "horizon_ms=0" / "two_pass_wave_coin_gap_per_contract.csv"
    if not needed_path.exists():
        candidates = sorted(out_dir.glob("eval_horizon_sweep/gap_ms=*/horizon_ms=0/two_pass_wave_coin_gap_per_contract.csv"))
        needed_path = candidates[0] if candidates else None
    if needed_path and Path(needed_path).exists():
        needed = load_csv(Path(needed_path))
        if needed is None or needed.empty:
            needed = pd.DataFrame(columns=["coin", "gap_usd_x_qty"])
        needed = (
            needed.groupby("coin", as_index=False)["gap_usd_x_qty"]
            .sum()
            .rename(columns={"gap_usd_x_qty": "needed_budget_usd"})
        )
    else:
        needed = pd.DataFrame(columns=["coin", "needed_budget_usd"])

    per_event = load_csv(out_dir / "two_time" / "per_adl_event_metrics.csv")
    if per_event is not None and not per_event.empty and "abs_qty_nonadl_60000ms" in per_event.columns:
        liq = (
            per_event.groupby("coin", as_index=False)["abs_qty_nonadl_60000ms"]
            .sum()
            .rename(columns={"abs_qty_nonadl_60000ms": "nonadl_volume_60s"})
        )
    else:
        liq = pd.DataFrame(columns=["coin", "nonadl_volume_60s"])

    df = equity.merge(needed, on="coin", how="left").merge(liq, on="coin", how="left")
    df["needed_budget_usd"] = pd.to_numeric(df["needed_budget_usd"], errors="coerce").fillna(0.0)
    df["nonadl_volume_60s"] = pd.to_numeric(df["nonadl_volume_60s"], errors="coerce").fillna(0.0)
    df["gap_ratio"] = np.where(df["needed_budget_usd"] > 0, df["equity_overshoot_usd"] / df["needed_budget_usd"], np.nan)
    df["gap_abs_usd"] = df["equity_overshoot_usd"] - df["needed_budget_usd"]
    return df


def load_coin_undo_impact_inputs(out_dir: Path) -> Optional[pd.DataFrame]:
    gap_path = out_dir / "eval_horizon_sweep" / "gap_ms=5000" / "horizon_ms=0" / "two_pass_wave_coin_gap_per_contract.csv"
    if not gap_path.exists():
        candidates = sorted(out_dir.glob("eval_horizon_sweep/gap_ms=*/horizon_ms=0/two_pass_wave_coin_gap_per_contract.csv"))
        gap_path = candidates[0] if candidates else None
    if not gap_path or not Path(gap_path).exists():
        return None
    gap = load_csv(Path(gap_path))
    if gap is None or gap.empty:
        return None
    if "gap_usd_x_qty" not in gap.columns or "qty_contracts" not in gap.columns or "coin" not in gap.columns:
        return None

    gap["gap_usd_x_qty"] = pd.to_numeric(gap["gap_usd_x_qty"], errors="coerce").fillna(0.0)
    gap["qty_contracts"] = pd.to_numeric(gap["qty_contracts"], errors="coerce").fillna(0.0)
    gap_coin = gap.groupby("coin", as_index=False).agg(
        gap_usd_x_qty=("gap_usd_x_qty", "sum"),
        qty_contracts=("qty_contracts", "sum"),
    )
    gap_coin["gap_per_contract_usd"] = np.where(
        gap_coin["qty_contracts"] > 0,
        gap_coin["gap_usd_x_qty"] / gap_coin["qty_contracts"],
        np.nan,
    )

    per_event = load_csv(out_dir / "two_time" / "per_adl_event_metrics.csv")
    if per_event is None or per_event.empty:
        return None
    if "coin" not in per_event.columns:
        return None
    if "abs_qty_nonadl_60000ms" not in per_event.columns or "undo_frac_60000ms" not in per_event.columns:
        return None

    per_event["abs_qty_nonadl_60000ms"] = pd.to_numeric(per_event["abs_qty_nonadl_60000ms"], errors="coerce").fillna(0.0)
    per_event["undo_frac_60000ms"] = pd.to_numeric(per_event["undo_frac_60000ms"], errors="coerce").fillna(0.0)
    per_event["undo_weighted"] = per_event["abs_qty_nonadl_60000ms"] * per_event["undo_frac_60000ms"]

    undo_coin = per_event.groupby("coin", as_index=False).agg(
        nonadl_volume_60s=("abs_qty_nonadl_60000ms", "sum"),
        undo_weighted_sum=("undo_weighted", "sum"),
        undo_frac_60s_mean=("undo_frac_60000ms", "mean"),
    )
    undo_coin["undo_frac_60s_wtd"] = np.where(
        undo_coin["nonadl_volume_60s"] > 0,
        undo_coin["undo_weighted_sum"] / undo_coin["nonadl_volume_60s"],
        np.nan,
    )

    df = gap_coin.merge(undo_coin, on="coin", how="left")
    df["nonadl_volume_60s"] = pd.to_numeric(df["nonadl_volume_60s"], errors="coerce").fillna(0.0)
    df["undo_frac_60s_wtd"] = pd.to_numeric(df["undo_frac_60s_wtd"], errors="coerce")
    return df


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overview_tiles(headlines: dict, out_png: Path) -> None:
    h = headlines.get("headline", {})
    d = headlines.get("derived", {})

    vals = [
        ("Queue overshoot", "equity USD", h.get("queue_total_overshoot_usd", 0.0)),
        ("Production overshoot", "vs needed (PnL USD)", h.get("prod_overshoot_vs_needed_usd", 0.0)),
        ("Overcollateralization", "equity / PnL", h.get("winner_overcollateralization_equity_over_pnl", {}).get("trimmed_mean", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    ax.axis("off")
    for i, (label, subtitle, value) in enumerate(vals):
        x0 = 0.03 + i * 0.32
        w = 0.29
        rect = Rectangle((x0, 0.18), w, 0.66, facecolor="#f8f8f8", edgecolor="#222", lw=1.0)
        ax.add_patch(rect)
        ax.text(x0 + 0.02, 0.78, label, fontsize=9, va="top", ha="left")
        ax.text(x0 + 0.02, 0.68, subtitle, fontsize=8, va="top", ha="left", color="#444")
        if "Overcollateralization" in label:
            ax.text(x0 + 0.02, 0.43, f"{value:.2f}×", fontsize=18, fontweight="bold", ha="left", va="center")
        else:
            ax.text(x0 + 0.02, 0.43, usd_short(float(value)), fontsize=18, fontweight="bold", ha="left", va="center")
    ax.text(0.03, 0.06, f"Naive expected PnL closed: {usd_short(d.get('naive_expected_pnl_closed_usd', 0.0))}", fontsize=9)
    save_fig(fig, out_png)


def plot_definition_diff(out_png: Path, *, old_queue_overshoot: Optional[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")

    left = Rectangle((0.05, 0.15), 0.42, 0.7, facecolor="#ffe8e8", edgecolor="#aa0000", lw=1.0)
    right = Rectangle((0.53, 0.15), 0.42, 0.7, facecolor="#e8f5e8", edgecolor="#1f7a1f", lw=1.0)
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.07, 0.78, "Old queue capacity (incorrect)", fontsize=10, fontweight="bold", ha="left", va="top")
    ax.text(0.07, 0.64, "Haircut capacity = equity + principal", fontsize=9, ha="left", va="top")
    ax.text(0.07, 0.52, "Overstates how much can be haircutted", fontsize=9, ha="left", va="top")
    if old_queue_overshoot is not None:
        ax.text(0.07, 0.34, f"Old headline: {usd_short(old_queue_overshoot)}", fontsize=10, ha="left", va="top")
    else:
        ax.text(0.07, 0.34, "Old headline: n/a", fontsize=10, ha="left", va="top")

    ax.text(0.55, 0.78, "Corrected queue capacity (OSS)", fontsize=10, fontweight="bold", ha="left", va="top")
    ax.text(0.55, 0.64, "Haircut capacity = PnL-only", fontsize=9, ha="left", va="top")
    ax.text(0.55, 0.52, "Uses position_unrealized_pnl", fontsize=9, ha="left", va="top")
    ax.text(0.55, 0.40, "Cash/principal is protected", fontsize=9, ha="left", va="top")

    ax.text(0.05, 0.04, "This correction is the main reason the headline changes.", fontsize=9)
    save_fig(fig, out_png)


def plot_mapping_bars(headlines: dict, out_png: Path) -> None:
    h = headlines.get("headline", {})
    d = headlines.get("derived", {})
    labels = ["Queue overshoot (equity USD)", "Naive expected PnL", "Observed PnL overshoot"]
    vals = [
        float(h.get("queue_total_overshoot_usd", 0.0)),
        float(d.get("naive_expected_pnl_closed_usd", 0.0)),
        float(h.get("prod_overshoot_vs_needed_usd", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(range(len(vals)), np.array(vals) / 1e6, color=["#2b6cb0", "#718096", "#c53030"])
    ax.set_xticks(range(len(vals)), labels, rotation=15, ha="right")
    ax.set_ylabel("USD (millions)")
    ax.set_title("Why equity-$ and PnL-$ headlines diverge")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), usd_short(v), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, out_png)


def plot_horizon_robustness(sweep: pd.DataFrame, out_png: Path) -> None:
    df = sweep.sort_values("eval_horizon_ms").copy()
    x = pd.to_numeric(df["eval_horizon_ms"], errors="coerce").fillna(0).to_numpy(float) / 1000.0
    y = pd.to_numeric(df["total_overshoot_prod_minus_needed_usd"], errors="coerce").fillna(0.0).to_numpy(float) / 1e6
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, y, color="#c53030", lw=2.0)
    ax.scatter(x, y, color="#c53030", s=18)
    ax.set_xlabel("Evaluation horizon Δ (seconds)")
    ax.set_ylabel("Overshoot vs needed (USD millions)")
    ax.set_title("Opportunity-cost sensitivity (horizon sweep)")
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    ax.axhspan(ymin, ymax, color="#c53030", alpha=0.08, label="min–max band")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_png)


def plot_queue_concentration(per_shock: pd.DataFrame, out_png: Path) -> None:
    df = per_shock.groupby("coin", as_index=False)["overshoot_queue_usd"].sum()
    df = df.sort_values("overshoot_queue_usd", ascending=False).head(10)
    if df.empty:
        return
    values = df["overshoot_queue_usd"].to_numpy(float)
    labels = df["coin"].astype(str).tolist()
    cum = np.cumsum(values) / float(values.sum()) * 100.0 if values.sum() > 0 else np.zeros_like(values)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, values / 1e6, color="#2f855a", alpha=0.85)
    ax.set_ylabel("Equity overshoot (USD millions)")
    ax.set_title("Queue overshoot (equity USD, top 10 coins)")
    ax.tick_params(axis="x", rotation=20)

    ax2 = ax.twinx()
    ax2.plot(labels, cum, color="#2b6cb0", marker="o")
    ax2.set_ylabel("Cumulative share (%)")
    ax2.set_ylim(0, 110)
    save_fig(fig, out_png)


def plot_equity_vs_needed_scatter(df: pd.DataFrame, out_png: Path) -> None:
    plot_df = df[(df["equity_overshoot_usd"] > 0) & (df["needed_budget_usd"] > 0)].copy()
    if plot_df.empty:
        return
    x = plot_df["equity_overshoot_usd"].to_numpy(float)
    y = plot_df["needed_budget_usd"].to_numpy(float)
    c = np.log10(plot_df["nonadl_volume_60s"].to_numpy(float) + 1.0)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=40, alpha=0.8, edgecolor="white", linewidth=0.3)
    lo = max(min(x.min(), y.min()), 1.0)
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="#c53030", lw=1.2, ls="--", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Equity overshoot per coin (USD)")
    ax.set_ylabel("PnL-closed proxy per coin (needed budget, USD)")
    ax.set_title("Equity vs PnL-closed proxy per symbol")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(non-ADL volume + 1)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    save_fig(fig, out_png)


def plot_gap_ratio_vs_liquidity(df: pd.DataFrame, out_png: Path) -> None:
    plot_df = df[(df["needed_budget_usd"] > 0) & (df["nonadl_volume_60s"] > 0)].copy()
    if plot_df.empty:
        return
    x = plot_df["nonadl_volume_60s"].to_numpy(float)
    y = plot_df["gap_ratio"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.scatter(x, y, s=36, alpha=0.7, color="#2b6cb0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#c53030", lw=1.0, ls="--", label="equity = PnL proxy")
    ax.set_xlabel("Non-ADL volume (60s, log scale)")
    ax.set_ylabel("Equity / PnL proxy ratio (log scale)")
    ax.set_title("Does lower liquidity show larger equity/PnL gaps?")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)
    save_fig(fig, out_png)


def plot_gap_ratio_bars(df: pd.DataFrame, out_png: Path) -> None:
    plot_df = df[(df["needed_budget_usd"] >= 1e5) & (df["gap_ratio"].notna())].copy()
    plot_df = plot_df.sort_values("gap_ratio", ascending=False).head(12)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(plot_df["coin"].astype(str), plot_df["gap_ratio"].to_numpy(float), color="#2f855a", alpha=0.85)
    ax.set_ylabel("Equity / PnL proxy ratio")
    ax.set_title("Largest equity/PnL gaps (min $100k needed budget)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, out_png)


def plot_undo_vs_impact_per_coin(df: pd.DataFrame, out_png: Path) -> None:
    plot_df = df[
        (df["gap_per_contract_usd"] > 0)
        & (df["nonadl_volume_60s"] > 0)
        & (df["undo_frac_60s_wtd"].notna())
    ].copy()
    if plot_df.empty:
        return
    x = plot_df["gap_per_contract_usd"].to_numpy(float)
    y = plot_df["undo_frac_60s_wtd"].to_numpy(float)
    c = np.log10(plot_df["nonadl_volume_60s"].to_numpy(float) + 1.0)

    rho = plot_df[["gap_per_contract_usd", "undo_frac_60s_wtd"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=40, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.axhline(0.25, color="#2f855a", lw=1.0, ls="--", label="undo=0.25")
    ax.set_xlabel("Impact proxy (gap per contract, USD, log scale)")
    ax.set_ylabel("Undo fraction (60s, volume-weighted)")
    ax.set_title("Undo fraction vs impact proxy (per coin)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(non-ADL volume + 1)")
    ax.text(0.02, 0.96, f"Spearman ρ={rho:.2f}, n={len(plot_df)}", transform=ax.transAxes, fontsize=9, va="top")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    save_fig(fig, out_png)


def plot_two_time_scatter(per_user: pd.DataFrame, out_png: Path) -> None:
    x_col = "share_any_nonadl_fill_60000ms"
    y_col = "mean_undo_frac_60000ms"
    if x_col not in per_user.columns or y_col not in per_user.columns:
        return
    x = pd.to_numeric(per_user[x_col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(per_user[y_col], errors="coerce").fillna(0.0)

    total = int(len(x))
    q_strategic = (x >= 0.5) & (y >= 0.25)
    q_undo = (x < 0.5) & (y >= 0.25)
    q_passive = (x < 0.5) & (y < 0.25)
    q_active = (x >= 0.5) & (y < 0.25)

    def pct(mask: pd.Series) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(mask.mean())

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.add_patch(Rectangle((0.5, 0.25), 0.5, 0.75, facecolor="#bee3f8", alpha=0.08, zorder=0))
    ax.add_patch(Rectangle((0.0, 0.25), 0.5, 0.75, facecolor="#e9d8fd", alpha=0.08, zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), 0.5, 0.25, facecolor="#edf2f7", alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((0.5, 0.0), 0.5, 0.25, facecolor="#fbd38d", alpha=0.08, zorder=0))
    ax.scatter(x, y, s=10, alpha=0.28, color="#1f2937", edgecolor="none")
    ax.axvline(0.5, color="#111827", lw=1.4, ls="-")
    ax.axhline(0.25, color="#111827", lw=1.4, ls="-")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of non-ADL trading within 60s")
    ax.set_ylabel("Undo fraction within 60s")
    ax.set_title("Two-time behavior: strategic vs passive")
    ax.text(0.02, 0.98, "Quadrant labels show share of users", transform=ax.transAxes, fontsize=8, va="top")
    ax.text(0.62, 0.98, "share=0.5", transform=ax.transAxes, fontsize=8, va="top", ha="left")
    ax.text(0.02, 0.74, "undo=0.25", transform=ax.transAxes, fontsize=8, va="top", ha="left")
    ax.text(0.73, 0.82, f"Strategic\n{pct(q_strategic):.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.25, 0.82, f"High-undo\n{pct(q_undo):.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.25, 0.12, f"Passive\n{pct(q_passive):.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.73, 0.12, f"Active\n{pct(q_active):.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_png)


def plot_two_time_quadrant_shares(
    per_user: pd.DataFrame,
    out_png: Path,
    *,
    weights: Optional[pd.Series] = None,
    title: str = "Two-time quadrant shares (user mass)",
    ylabel: str = "Share of users (%)",
) -> None:
    x_col = "share_any_nonadl_fill_60000ms"
    y_col = "mean_undo_frac_60000ms"
    if x_col not in per_user.columns or y_col not in per_user.columns:
        return
    x = pd.to_numeric(per_user[x_col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(per_user[y_col], errors="coerce").fillna(0.0)
    if len(x) == 0:
        return
    if weights is None:
        weights = pd.Series(1.0, index=per_user.index)
    else:
        weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)

    q_strategic = (x >= 0.5) & (y >= 0.25)
    q_active = (x >= 0.5) & (y < 0.25)
    q_high_undo = (x < 0.5) & (y >= 0.25)
    q_passive = (x < 0.5) & (y < 0.25)

    labels = ["Strategic", "Active", "High-undo", "Passive"]
    counts = np.array(
        [
            weights[q_strategic].sum(),
            weights[q_active].sum(),
            weights[q_high_undo].sum(),
            weights[q_passive].sum(),
        ],
        dtype=float,
    )
    total = float(counts.sum())
    shares = counts / total * 100.0 if total > 0 else np.zeros_like(counts)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(labels, shares, color=["#2b6cb0", "#718096", "#2f855a", "#a0aec0"])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(10.0, float(np.max(shares)) * 1.2))
    ax.set_title(title)
    for b, v in zip(bars, shares):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    save_fig(fig, out_png)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create additional visuals for OSS corrections and analysis.")
    parser.add_argument("--out", default="out", help="Output directory (default: out)")
    parser.add_argument("--fig-dir", default=None, help="Figures output directory (default: <out>/figures_corrections)")
    parser.add_argument("--old-queue-overshoot", type=float, default=None, help="Optional legacy queue overshoot value")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir / "figures_corrections"
    fig_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []

    headlines = load_json(out_dir / "headlines.json")
    if headlines:
        p = fig_dir / "00_overview_tiles.png"
        plot_overview_tiles(headlines, p)
        outputs.append(p)

        p = fig_dir / "02_mapping_bars.png"
        plot_mapping_bars(headlines, p)
        outputs.append(p)

    p = fig_dir / "01_definition_diff.png"
    plot_definition_diff(p, old_queue_overshoot=args.old_queue_overshoot)
    outputs.append(p)

    sweep = load_csv(out_dir / "eval_horizon_sweep_gap_ms=5000.csv")
    if sweep is not None and not sweep.empty:
        p = fig_dir / "03_horizon_robustness.png"
        plot_horizon_robustness(sweep, p)
        outputs.append(p)

    per_shock = load_csv(out_dir / "per_shock_metrics_queue.csv")
    if per_shock is not None and not per_shock.empty:
        p = fig_dir / "04_queue_concentration.png"
        plot_queue_concentration(per_shock, p)
        outputs.append(p)

    per_user = load_csv(out_dir / "two_time" / "per_user_reaction_metrics.csv")
    per_event = load_csv(out_dir / "two_time" / "per_adl_event_metrics.csv")
    if per_user is not None and not per_user.empty:
        p = fig_dir / "05_two_time_scatter.png"
        plot_two_time_scatter(per_user, p)
        outputs.append(p)

        p = fig_dir / "06_two_time_quadrant_shares.png"
        plot_two_time_quadrant_shares(per_user, p)
        outputs.append(p)

        if per_event is not None and not per_event.empty and "abs_qty_nonadl_60000ms" in per_event.columns:
            weights = per_event.groupby("user", as_index=True)["abs_qty_nonadl_60000ms"].sum()
            weights = per_user["user"].map(weights).fillna(0.0)
            p = fig_dir / "07_two_time_quadrant_shares_volume.png"
            plot_two_time_quadrant_shares(
                per_user,
                p,
                weights=weights,
                title="Two-time quadrant shares (non-ADL volume)",
                ylabel="Share of non-ADL volume (%)",
            )
            outputs.append(p)

    coin_gap = load_coin_gap_inputs(out_dir)
    if coin_gap is not None and not coin_gap.empty:
        p = fig_dir / "08_equity_vs_needed_scatter.png"
        plot_equity_vs_needed_scatter(coin_gap, p)
        outputs.append(p)

        p = fig_dir / "09_gap_ratio_vs_liquidity.png"
        plot_gap_ratio_vs_liquidity(coin_gap, p)
        outputs.append(p)

        p = fig_dir / "10_gap_ratio_bars.png"
        plot_gap_ratio_bars(coin_gap, p)
        outputs.append(p)

    coin_undo = load_coin_undo_impact_inputs(out_dir)
    if coin_undo is not None and not coin_undo.empty:
        p = fig_dir / "12_undo_vs_impact_per_coin.png"
        plot_undo_vs_impact_per_coin(coin_undo, p)
        outputs.append(p)

    print("[info] wrote figures:")
    for p in outputs:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
