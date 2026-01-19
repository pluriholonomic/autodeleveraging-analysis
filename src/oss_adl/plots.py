from __future__ import annotations

"""
Figure generation for `OSS/methodology.md`.

Goal:
- keep plots simple, skeptics-friendly, and directly traceable to the CSV/JSON outputs
- avoid research-notebook noise
"""

import contextlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oss_adl.policy_per_wave import generate_policy_per_wave_figures


def _read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def plot_headline_bars(*, headlines_json: Path, out_png: Path) -> None:
    h = _read_json(headlines_json)
    q = float(h["headline"]["queue_total_overshoot_usd"])
    p = float(h["headline"]["prod_overshoot_vs_needed_usd"])
    n = h["derived"]["naive_expected_pnl_closed_usd"]
    n = float(n) if n is not None else float("nan")

    labels = ["queue overshoot\n(equity-Strategy)", "prod overshoot\n(PnL-Strategy)", "naive expected\n(PnL-Strategy)"]
    vals = np.array([q, p, n], dtype=float) / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, vals, color=["#2b6cb0", "#c53030", "#2f855a"])
    ax.set_ylabel("USD (millions)")
    ax.set_title("Headline numbers (from OSS/out/headlines.json)")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.1f}M", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_overshoot_vs_horizon(*, horizon_sweep_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(horizon_sweep_csv).sort_values("eval_horizon_ms")
    x = pd.to_numeric(df["eval_horizon_ms"], errors="coerce").fillna(0).to_numpy(float) / 1000.0
    y = pd.to_numeric(df["total_overshoot_prod_minus_needed_usd"], errors="coerce").fillna(0.0).to_numpy(float) / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, y, lw=2.0, color="#c53030")
    ax.set_xlabel("Evaluation horizon Δ (seconds)")
    ax.set_ylabel("Total overshoot vs needed (USD millions)")
    ax.set_title("Two-pass production overshoot vs needed vs evaluation horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_overcollateralization_hist(
    *,
    canonical_realtime_csv: Path,
    out_png: Path,
    clip_max: float = 50.0,
    bins: int = 60,
) -> None:
    cols = ["time", "user", "is_negative_equity", "total_unrealized_pnl", "total_equity"]
    canon = pd.read_csv(canonical_realtime_csv, usecols=cols)
    canon["time"] = pd.to_numeric(canon["time"], errors="coerce").fillna(0).astype(np.int64)
    if canon["is_negative_equity"].dtype != bool:
        canon["is_negative_equity"] = (
            canon["is_negative_equity"].astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})
        )
    canon["total_unrealized_pnl"] = pd.to_numeric(canon["total_unrealized_pnl"], errors="coerce").fillna(0.0)
    canon["total_equity"] = pd.to_numeric(canon["total_equity"], errors="coerce").fillna(0.0)

    winners = canon[
        (~canon["is_negative_equity"]) & (canon["total_unrealized_pnl"] > 0.0) & (canon["total_equity"] > 0.0)
    ].copy()
    winners = winners.sort_values("time").groupby("user", as_index=False).first()
    ratio = (winners["total_equity"] / winners["total_unrealized_pnl"]).to_numpy(float)
    ratio = ratio[np.isfinite(ratio)]
    ratio = ratio[(ratio > 0.0) & (ratio <= float(clip_max))]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(ratio, bins=bins, color="#2b6cb0", alpha=0.85)
    ax.set_xlabel("Equity / PnL (clipped)")
    ax.set_ylabel("Number of users")
    ax.set_title(f"Winner overcollateralization distribution (clipped at {clip_max}×)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_undo_fraction_hist(*, per_event_csv: Path, out_png: Path, window_ms: int = 60000) -> None:
    df = pd.read_csv(per_event_csv)
    col = f"undo_frac_{int(window_ms)}ms"
    if col not in df.columns:
        raise ValueError(f"{per_event_csv} missing {col}")
    u = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)
    u = u[(u >= 0.0) & (u <= 1.0)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(u, bins=50, color="#2f855a", alpha=0.85)
    ax.set_xlabel(f"Undo fraction in {int(window_ms / 1000)}s window")
    ax.set_ylabel("Number of ADL events")
    ax.set_title("Undo fraction distribution (opposite-volume definition)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_ptsr_pmr_per_wave(*, policy_per_wave_csv: Path, out_png: Path) -> None:
    """
    Plot per-wave PTSR/PMR time series for production vs benchmark allocations.

    These are fairness-aware ratios defined in the paper (max post-haircut endowment normalized by
    socialized budget, and by theta*max-loss respectively), computed in endowment space.
    """
    df = pd.read_csv(policy_per_wave_csv)
    if "t_start" not in df.columns:
        raise ValueError(f"{policy_per_wave_csv} missing t_start")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    series = [
        ("Production", "ptsr_prod", "pmr_prod", "#c53030"),
        ("Wealth pro-rata (cont)", "ptsr_pr", "pmr_pr", "#2b6cb0"),
        ("PNL pro-rata (cont)", "ptsr_pnl_pr", "pmr_pnl_pr", "#9b2c2c"),
        ("Equity pro-rata (cont)", "ptsr_eq_pr", "pmr_eq_pr", "#2c5282"),
        ("Vector-MD", "ptsr_vector", "pmr_vector", "#6b46c1"),
        ("Min-max ILP (int)", "ptsr_fixed_point_ilp_integer", "pmr_fixed_point_ilp_integer", "#2f855a"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    ax_ptsr, ax_pmr = axes
    for label, c_ptsr, c_pmr, color in series:
        if c_ptsr in df.columns:
            y = pd.to_numeric(df[c_ptsr], errors="coerce")
            ax_ptsr.plot(df["t_start"], y, label=label, linewidth=2, color=color)
        if c_pmr in df.columns:
            y = pd.to_numeric(df[c_pmr], errors="coerce")
            ax_pmr.plot(df["t_start"], y, label=label, linewidth=2, color=color)

    ax_ptsr.set_yscale("symlog")
    ax_ptsr.set_ylabel("RD-PTSR (symlog)")
    ax_pmr.set_ylabel("PMR (log)")
    ax_pmr.set_xlabel("Wave start time (UTC)")
    ax_ptsr.set_title("Per-wave fairness ratios (endowment-space)")
    ax_ptsr.grid(True, alpha=0.25)
    ax_pmr.grid(True, alpha=0.25)
    ax_ptsr.legend(loc="upper left", ncol=2, fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_rd_nse_per_wave(*, policy_per_wave_csv: Path, out_png: Path) -> None:
    """
    Plot per-wave RD-NSE (Remaining Deficit Normalized Survivor Endowment) time series.
    """
    df = pd.read_csv(policy_per_wave_csv)
    if "t_start" not in df.columns:
        raise ValueError(f"{policy_per_wave_csv} missing t_start")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    series = [
        ("Production", "rd_nse_prod", "#c53030"),
        ("Wealth pro-rata (cont)", "rd_nse_pr", "#2b6cb0"),
        ("PNL pro-rata (cont)", "rd_nse_pnl_pr", "#9b2c2c"),
        ("Equity pro-rata (cont)", "rd_nse_eq_pr", "#2c5282"),
        ("Vector-MD", "rd_nse_vector", "#6b46c1"),
        ("Min-max ILP (int)", "rd_nse_fixed_point_ilp_integer", "#2f855a"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 4.8))
    for label, c_nse, color in series:
        if c_nse in df.columns:
            y = pd.to_numeric(df[c_nse], errors="coerce")
            ax.plot(df["t_start"], y, label=label, linewidth=2, color=color)

    ax.set_ylabel("RD-NSE")
    ax.set_xlabel("Wave start time (UTC)")
    ax.set_title("Per-wave RD-NSE (Remaining Deficit Normalized Survivor Endowment)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_policy_per_wave_performance(*, policy_per_wave_csv: Path, out_png: Path) -> None:
    """Plot policy performance per wave (figure 05)."""
    df = pd.read_csv(policy_per_wave_csv)
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    ax = axes[0]
    ax.plot(df["t_start"], df["budget_needed"].astype(float) / 1e6, label="B_needed", linewidth=2)
    ax.plot(df["t_start"], df["budget_prod"].astype(float) / 1e6, label="H_prod", linestyle="--")
    ax.set_ylabel("USD (millions)")
    ax.set_title("Budget vs production haircut total per wave")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(df["t_start"], df["overshoot_prod"], label="prod - needed", linewidth=2)
    ax.plot(df["t_start"], df["overshoot_contract_pr"], label="contract pro-rata (integer) - needed", linestyle="--")
    ax.plot(
        df["t_start"], df["overshoot_fixed_point_ilp_integer"], label="min-max ILP (integer) - needed", linestyle="--"
    )
    ax.plot(df["t_start"], df["overshoot_vector"], label="vector-md - needed", linestyle="--")
    ax.set_ylabel("USD")
    ax.set_title("Overshoot per wave")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df["t_start"], df["max_pct_prod"], label="max % haircut (production)", linewidth=2)
    ax.plot(df["t_start"], df["max_pct_contract_pr"], label="max % haircut (contract pro-rata integer)")
    ax.plot(df["t_start"], df["max_pct_fixed_point_ilp_integer"], label="max % haircut (min-max ILP integer)")
    ax.plot(df["t_start"], df["max_pct_vector"], label="max % haircut (vector-md)")
    ax.set_ylabel("max(h_i / equity_i)")
    ax.set_title("Max % haircut per wave")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(df["t_start"], df["theta_prod_capped"], label="theta_prod_capped = H_prod / D", linewidth=2)
    ax.plot(df["t_start"], df["theta_needed_capped"], label="theta_needed_capped = B_needed / D", linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("theta (capped)")
    ax.set_title("Empirical severity over time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_policy_cumulative_overshoot(*, policy_per_wave_csv: Path, out_png: Path) -> None:
    """Plot cumulative overshoot vs needed (figure 06)."""
    df = pd.read_csv(policy_per_wave_csv)
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    cum = pd.DataFrame(
        {
            "t_start": df["t_start"],
            "cum_needed": df["budget_needed"].astype(float).cumsum(),
            "cum_prod": df["budget_prod"].astype(float).cumsum(),
            "cum_pro_rata_cont": df["budget_pr"].astype(float).cumsum(),
            "cum_contract_pr": df["budget_contract_pr"].astype(float).cumsum(),
            "cum_fp_ilp_int": df["budget_fixed_point_ilp_integer"].astype(float).cumsum(),
            "cum_vector": df["budget_vector"].astype(float).cumsum(),
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(cum["t_start"], cum["cum_prod"] - cum["cum_needed"], label="production - needed", linewidth=2)
    ax.plot(
        cum["t_start"],
        cum["cum_pro_rata_cont"] - cum["cum_needed"],
        label="wealth pro-rata (cont) - needed",
        linestyle="--",
    )
    ax.plot(
        cum["t_start"],
        cum["cum_contract_pr"] - cum["cum_needed"],
        label="contract pro-rata (int) - needed",
        linestyle="--",
    )
    ax.plot(
        cum["t_start"], cum["cum_fp_ilp_int"] - cum["cum_needed"], label="min-max ILP (int) - needed", linestyle="--"
    )
    ax.plot(
        cum["t_start"], cum["cum_vector"] - cum["cum_needed"], label="vector-md - needed", linestyle="-.", linewidth=2
    )
    ax.set_ylabel("cumulative USD")
    ax.set_title("Cumulative overshoot vs needed")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def generate_all_figures(*, out_root: Path, canonical_realtime_csv: Path, gap_ms: int = 5000) -> None:
    out_root = Path(out_root)
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    headlines_json = out_root / "headlines.json"
    if headlines_json.exists():
        plot_headline_bars(headlines_json=headlines_json, out_png=figs / "01_headlines.png")

    sweep_csv = out_root / f"eval_horizon_sweep_gap_ms={int(gap_ms)}.csv"
    if sweep_csv.exists():
        plot_overshoot_vs_horizon(horizon_sweep_csv=sweep_csv, out_png=figs / "02_overshoot_vs_horizon.png")

    if Path(canonical_realtime_csv).exists():
        plot_overcollateralization_hist(
            canonical_realtime_csv=canonical_realtime_csv, out_png=figs / "03_overcollateralization_hist.png"
        )

    per_event = out_root / "two_time" / "per_adl_event_metrics.csv"
    if per_event.exists():
        plot_undo_fraction_hist(per_event_csv=per_event, out_png=figs / "04_undo_fraction_hist.png", window_ms=60000)

    # Policy comparisons: production vs benchmark allocations targeting B_needed.
    # These require the two-pass per-wave outputs at horizon=0ms.
    p = out_root / "policy_per_wave_metrics.csv"

    # If CSV doesn't exist, try to regenerate it (requires canonical CSV)
    if not p.exists():
        with contextlib.suppress(FileNotFoundError):
            generate_policy_per_wave_figures(
                out_root=out_root,
                canonical_realtime_csv=Path(canonical_realtime_csv),
                gap_ms=int(gap_ms),
                horizon_ms=0,
            )

    # Generate plots from existing CSV (either pre-included or just regenerated)
    if p.exists():
        plot_policy_per_wave_performance(policy_per_wave_csv=p, out_png=figs / "05_policy_per_wave_performance.png")
        plot_policy_cumulative_overshoot(
            policy_per_wave_csv=p, out_png=figs / "06_policy_per_wave_cumulative_overshoot.png"
        )
        plot_ptsr_pmr_per_wave(policy_per_wave_csv=p, out_png=figs / "07_ptsr_pmr_per_wave.png")
        plot_rd_nse_per_wave(policy_per_wave_csv=p, out_png=figs / "08_rd_nse_per_wave.png")
        plot_cumulative_regret_historical(policy_per_wave_csv=p, out_png=figs / "09_cumulative_regret_historical.png")
        plot_regret_decomposition_historical(policy_per_wave_csv=p, out_dir=figs)


def plot_cumulative_regret_historical(*, policy_per_wave_csv: Path, out_png: Path) -> None:
    """
    Plot cumulative regret over the historical waves (figure 09).
    Shows three panels: overshoot regret, fairness regret, total regret.
    """
    df = pd.read_csv(policy_per_wave_csv)
    if "t_start" not in df.columns:
        raise ValueError(f"{policy_per_wave_csv} missing t_start")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    # Define policies with corrected names
    policies = [
        ("Production", "overshoot_prod", "max_pct_prod", "#c53030"),
        ("Wealth pro-rata (cont)", "overshoot_pr", "max_pct_pr", "#2b6cb0"),
        ("Contract pro-rata (int)", "overshoot_contract_pr", "max_pct_contract_pr", "#f6ad55"),
        ("Min-max ILP (int)", "overshoot_fixed_point_ilp_integer", "max_pct_fixed_point_ilp_integer", "#2f855a"),
        ("Vector-MD", "overshoot_vector", "max_pct_vector", "#6b46c1"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Cumulative Overshoot Regret
    ax = axes[0]
    for label, overshoot_col, _, color in policies:
        if overshoot_col in df.columns:
            cum_overshoot = df[overshoot_col].fillna(0).cumsum() / 1e6
            ax.plot(df["t_start"], cum_overshoot, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_title("Cumulative Overshoot Regret (budget - needed)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=1)

    # Panel 2: Cumulative Fairness Regret (using max_pct * budget as proxy)
    ax = axes[1]
    for label, _overshoot_col, maxpct_col, color in policies:
        if maxpct_col in df.columns and "budget_needed" in df.columns:
            # Fairness regret = |max_pct - max_pct_pr| * budget
            maxpct = df[maxpct_col].fillna(0)
            maxpct_pr = df["max_pct_pr"].fillna(0) if "max_pct_pr" in df.columns else 0
            budget = df["budget_needed"].fillna(0)
            fairness = np.abs(maxpct - maxpct_pr) * budget
            cum_fairness = fairness.cumsum() / 1e6
            ax.plot(df["t_start"], cum_fairness, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_title("Cumulative Fairness Regret: Σ |max_pct - max_pct_pr| × budget")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Total Cumulative Regret
    ax = axes[2]
    for label, overshoot_col, maxpct_col, color in policies:
        if overshoot_col in df.columns and maxpct_col in df.columns and "budget_needed" in df.columns:
            overshoot = df[overshoot_col].fillna(0)
            maxpct = df[maxpct_col].fillna(0)
            maxpct_pr = df["max_pct_pr"].fillna(0) if "max_pct_pr" in df.columns else 0
            budget = df["budget_needed"].fillna(0)
            fairness = np.abs(maxpct - maxpct_pr) * budget
            total = overshoot + fairness
            cum_total = total.cumsum() / 1e6
            ax.plot(df["t_start"], cum_total, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_xlabel("Wave start time (UTC)")
    ax.set_title("Total Cumulative Regret (Overshoot + Fairness)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_regret_decomposition_historical(*, policy_per_wave_csv: Path, out_dir: Path) -> None:
    """
    Plot regret decomposition as three separate figures (figure 10 split).
    - 10a: Overshoot regret by policy
    - 10b: Fairness regret by policy
    - 10c: Total regret by policy
    """
    df = pd.read_csv(policy_per_wave_csv)
    if "t_start" not in df.columns:
        raise ValueError(f"{policy_per_wave_csv} missing t_start")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True, errors="coerce")
    df = df.sort_values("t_start").reset_index(drop=True)

    # Define policies with corrected names
    policies = [
        ("Production (Hyperliquid)", "overshoot_prod", "max_pct_prod", "#c53030"),
        ("Vector-MD", "overshoot_vector", "max_pct_vector", "#6b46c1"),
        ("Min-max ILP", "overshoot_fixed_point_ilp_integer", "max_pct_fixed_point_ilp_integer", "#2f855a"),
        ("Contract pro-rata", "overshoot_contract_pr", "max_pct_contract_pr", "#f6ad55"),
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 10a: Overshoot Regret (absolute distance from budget target)
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, overshoot_col, _, color in policies:
        if overshoot_col in df.columns:
            cum = np.abs(df[overshoot_col].fillna(0)).cumsum() / 1e6
            ax.plot(df["t_start"], cum, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Overshoot Regret: Σ |overshoot| (October 10, 2025)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "10a_overshoot_regret.png", dpi=200)
    plt.close(fig)

    # Figure 10b: Fairness Regret (absolute distance from Min-max ILP benchmark)
    fig, ax = plt.subplots(figsize=(10, 4))
    maxpct_ilp = (
        df["max_pct_fixed_point_ilp_integer"].fillna(0) if "max_pct_fixed_point_ilp_integer" in df.columns else 0
    )
    for label, _, maxpct_col, color in policies:
        if maxpct_col in df.columns and "budget_needed" in df.columns:
            maxpct = df[maxpct_col].fillna(0)
            budget = df["budget_needed"].fillna(0)
            fairness = np.abs(maxpct - maxpct_ilp) * budget
            cum = fairness.cumsum() / 1e6
            ax.plot(df["t_start"], cum, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Fairness Regret: Σ |max_pct − max_pct_ILP| × budget (October 10, 2025)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "10b_fairness_regret.png", dpi=200)
    plt.close(fig)

    # Figure 10c: Total Regret
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, overshoot_col, maxpct_col, color in policies:
        if overshoot_col in df.columns and maxpct_col in df.columns and "budget_needed" in df.columns:
            overshoot = np.abs(df[overshoot_col].fillna(0))
            maxpct = df[maxpct_col].fillna(0)
            budget = df["budget_needed"].fillna(0)
            fairness = np.abs(maxpct - maxpct_ilp) * budget
            total = overshoot + fairness
            cum = total.cumsum() / 1e6
            ax.plot(df["t_start"], cum, label=label, linewidth=2, color=color)
    ax.set_ylabel("Cumulative USD (millions)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Total Regret: Overshoot + Fairness (October 10, 2025)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "10c_total_regret.png", dpi=200)
