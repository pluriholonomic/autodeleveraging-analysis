from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from oss_adl.bad_debt import compute_loser_waves, write_loser_waves
from oss_adl.paths import resolve_repo_paths
from oss_adl.plots import generate_all_figures
from oss_adl.queue_overshoot import compute_queue_overshoot, write_queue_outputs
from oss_adl.two_pass_replay import (
    run_eval_horizon_sweep,
    run_wave_gap_sweep,
    summarize_overshoot_robustness_from_sweep,
)
from oss_adl.two_time_behavior import classify_strategic_vs_passive, write_two_time_outputs


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _load_queue_overshoot(out_root: Path) -> float:
    p = out_root / "summary_totals.csv"
    df = pd.read_csv(p)
    row = df[df["policy"].astype(str) == "queue"]
    if row.empty:
        raise ValueError(f"Could not find policy=='queue' in {p}")
    return float(pd.to_numeric(row.iloc[0]["total_overshoot"], errors="coerce"))


def _load_prod_overshoot_vs_needed(out_root: Path, *, gap_ms: int, horizon_ms: int) -> float:
    p = out_root / f"eval_horizon_sweep_gap_ms={int(gap_ms)}.csv"
    df = pd.read_csv(p)
    df["eval_horizon_ms"] = pd.to_numeric(df["eval_horizon_ms"], errors="coerce").fillna(0).astype(int)
    row = df[df["eval_horizon_ms"] == int(horizon_ms)]
    if row.empty:
        raise ValueError(f"No eval_horizon_ms=={horizon_ms} row in {p}")
    return float(pd.to_numeric(row.iloc[0]["total_overshoot_prod_minus_needed_usd"], errors="coerce"))


def _winner_overcollateralization(out_root: Path, canonical_realtime_csv: Path, *, trim_alpha: float) -> dict:
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

    if ratio.size == 0:
        return {"n_users": 0, "mean": float("nan"), "median": float("nan"), "trimmed_mean": float("nan")}

    a = float(np.clip(trim_alpha, 0.0, 0.49))
    lo = float(np.quantile(ratio, a))
    hi = float(np.quantile(ratio, 1.0 - a))
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    trimmed_mean = float(np.mean(trimmed)) if trimmed.size else float("nan")

    return {
        "n_users": int(ratio.size),
        "mean": float(np.mean(ratio)),
        "median": float(np.median(ratio)),
        "trimmed_mean": float(trimmed_mean),
        "p10": float(np.quantile(ratio, 0.10)),
        "p90": float(np.quantile(ratio, 0.90)),
    }


def cmd_queue(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rp = resolve_repo_paths(oss_root=oss_root)

    outputs = compute_queue_overshoot(
        winners_csv=rp.hypermultiadl_winners_csv,
        liquidations_csv=rp.hypermultiadl_liquidations_csv,
        gap_seconds=float(args.gap_seconds),
        follower_decay_beta=float(args.follower_decay_beta),
        input_mode=str(getattr(args, "queue_input_mode", "auto")),
        loser_deficit_select_mode=str(getattr(args, "queue_deficit_select_mode", "total_equity")),
        loser_deficit_value_mode=str(getattr(args, "queue_deficit_value_mode", "total_equity")),
    )
    # Write to out_root directly to mirror the repo’s `results/` expectations.
    _, summary_path = write_queue_outputs(out_dir=out_root, outputs=outputs)
    print("[info] wrote", summary_path)


def cmd_replay(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rp = resolve_repo_paths(oss_root=oss_root)

    # 1) loser waves
    lw = compute_loser_waves(
        canonical_realtime_csv=rp.hyperreplay_canonical_realtime_csv, gap_ms=int(args.gap_ms), prefer_equity=True
    )
    waves_csv, _ = write_loser_waves(out_dir=out_root, loser_waves=lw)
    print("[info] wrote", waves_csv)

    # 2) eval-horizon sweep (fixed waves)
    horizons = [int(x) for x in args.horizons_ms]
    _df = run_eval_horizon_sweep(
        hyperreplay_root=rp.hyperreplay_dir,
        loser_waves_csv=waves_csv,
        out_root=out_root,
        gap_ms=int(args.gap_ms),
        horizons_ms=horizons,
    )
    print("[info] wrote", out_root / f"eval_horizon_sweep_gap_ms={int(args.gap_ms)}.csv")

    # 3) robustness summary
    rob = summarize_overshoot_robustness_from_sweep(
        sweep_csv=out_root / f"eval_horizon_sweep_gap_ms={int(args.gap_ms)}.csv",
        half_lives_ms=[int(x) for x in args.half_lives_ms],
        band_max_ms=int(args.band_max_ms),
    )
    (out_root / "overshoot_robustness.json").write_text(_json_dumps(rob))
    print("[info] wrote", out_root / "overshoot_robustness.json")


def cmd_gap_sweep(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rp = resolve_repo_paths(oss_root=oss_root)

    gaps = [int(x) for x in args.gaps_ms]
    _df = run_wave_gap_sweep(
        hyperreplay_root=rp.hyperreplay_dir,
        canonical_realtime_csv=rp.hyperreplay_canonical_realtime_csv,
        out_root=out_root,
        gaps_ms=gaps,
        eval_horizon_ms=int(args.eval_horizon_ms),
    )
    print("[info] wrote", out_root / "wave_gap_sweep.csv")


def cmd_two_time(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()
    rp = resolve_repo_paths(oss_root=oss_root)

    gap_ms = int(args.gap_ms)
    h0 = int(args.h0_ms)
    h1 = int(args.h1_ms)

    waves_csv = out_root / "bad_debt_loser_waves.csv"
    if not waves_csv.exists():
        raise FileNotFoundError(f"Missing {waves_csv}. Run `oss-adl replay` first (or `oss-adl all`).")

    base = out_root / "eval_horizon_sweep" / f"gap_ms={gap_ms}"
    h0_csv = base / f"horizon_ms={h0}" / "two_pass_wave_prod_haircuts.csv"
    h1_csv = base / f"horizon_ms={h1}" / "two_pass_wave_prod_haircuts.csv"
    if not h0_csv.exists() or not h1_csv.exists():
        raise FileNotFoundError(
            f"Missing horizon haircuts CSVs:\n- {h0_csv}\n- {h1_csv}\nRun `oss-adl replay` with horizons including {h0} and {h1}."
        )

    fills = [rp.hyperreplay_raw_dir / "20_fills.json", rp.hyperreplay_raw_dir / "21_fills.json"]
    out = classify_strategic_vs_passive(
        fills_json_paths=fills,
        waves_csv=waves_csv,
        canonical_realtime_csv=rp.hyperreplay_canonical_realtime_csv,
        two_pass_h0_prod_haircuts_csv=h0_csv,
        two_pass_h1_prod_haircuts_csv=h1_csv,
        queue_summary_csv=out_root / "summary_totals.csv",
        horizon_sweep_csv=out_root / f"eval_horizon_sweep_gap_ms={gap_ms}.csv",
        pnl_closed_horizon_ms=0,
    )
    out_dir = out_root / "two_time"
    paths = write_two_time_outputs(out_dir=out_dir, outputs=out)
    print("[info] wrote", paths["report.md"])


def cmd_headlines(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()
    rp = resolve_repo_paths(oss_root=oss_root)

    queue_overshoot = _load_queue_overshoot(out_root)
    prod_overshoot_vs_needed = _load_prod_overshoot_vs_needed(
        out_root, gap_ms=int(args.gap_ms), horizon_ms=int(args.horizon_ms)
    )
    oc = _winner_overcollateralization(
        out_root, rp.hyperreplay_canonical_realtime_csv, trim_alpha=float(args.trim_alpha)
    )

    naive_expected_pnl_closed = None
    if np.isfinite(oc.get("trimmed_mean", float("nan"))) and oc["trimmed_mean"] > 0:
        naive_expected_pnl_closed = float(queue_overshoot / oc["trimmed_mean"])

    factor = None
    if naive_expected_pnl_closed is not None and prod_overshoot_vs_needed > 0:
        factor = float(naive_expected_pnl_closed / prod_overshoot_vs_needed)

    summary = {
        "inputs": {
            "out_root": str(out_root),
            "gap_ms": int(args.gap_ms),
            "horizon_ms": int(args.horizon_ms),
            "trim_alpha": float(args.trim_alpha),
            "hyperreplay_canonical_realtime_csv": str(rp.hyperreplay_canonical_realtime_csv),
        },
        "headline": {
            "queue_total_overshoot_usd": float(queue_overshoot),
            "prod_overshoot_vs_needed_usd": float(prod_overshoot_vs_needed),
            "winner_overcollateralization_equity_over_pnl": oc,
        },
        "derived": {
            "naive_expected_pnl_closed_usd": naive_expected_pnl_closed,
            "factor_naive_expected_over_observed": factor,
        },
    }
    (out_root / "headlines.json").write_text(_json_dumps(summary))
    print(_json_dumps(summary), end="")


def cmd_all(args: argparse.Namespace) -> None:
    # Run the core steps in the order used by the methodology.
    cmd_queue(args)
    cmd_replay(args)
    cmd_headlines(args)
    cmd_two_time(args)
    cmd_plots(args)


def cmd_plots(args: argparse.Namespace) -> None:
    oss_root = Path(args.oss_root).resolve()
    out_root = Path(args.out).resolve()

    # Use explicit path if provided
    if getattr(args, "canonical_csv", None) is not None:
        canonical_realtime_csv = Path(args.canonical_csv).resolve()
    else:
        # For plots command, we only need the canonical CSV (not the full repo paths).
        # Try multiple locations to find it.
        canonical_realtime_csv = Path("/nonexistent")

        # Check environment variable first
        if "HYPERREPLAY_DIR" in os.environ:
            candidate = (
                Path(os.environ["HYPERREPLAY_DIR"]) / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"
            )
            if candidate.exists():
                canonical_realtime_csv = candidate

        # Fall back to external/HyperReplay
        if not canonical_realtime_csv.exists():
            candidate = (
                oss_root / "external" / "HyperReplay" / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"
            )
            if candidate.exists():
                canonical_realtime_csv = candidate

        # Fall back to vendored HyperReplay (monorepo layout)
        if not canonical_realtime_csv.exists():
            candidate = oss_root.parent / "HyperReplay" / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"
            if candidate.exists():
                canonical_realtime_csv = candidate
    generate_all_figures(
        out_root=out_root, canonical_realtime_csv=canonical_realtime_csv, gap_ms=int(getattr(args, "gap_ms", 5000))
    )
    print("[info] wrote figures under", out_root / "figures")


def main() -> None:
    oss_root_default = Path(__file__).resolve().parents[2]
    out_default = oss_root_default / "out"

    p = argparse.ArgumentParser(
        prog="oss-adl", description="Minimal OSS reproduction of key ADL headline numbers + figures."
    )
    p.add_argument(
        "--oss-root", type=Path, default=oss_root_default, help="Path to OSS/ folder (defaults to this checkout)."
    )
    p.add_argument("--out", type=Path, default=out_default, help="Output directory (defaults to OSS/out).")

    sub = p.add_subparsers(dest="cmd", required=True)

    # queue
    q = sub.add_parser("queue", help="Compute ~650m queue overshoot (equity-$) from HyperMultiAssetedADL exports.")
    q.add_argument("--gap-seconds", type=float, default=5.0)
    q.add_argument("--follower-decay-beta", type=float, default=5.0)
    q.add_argument("--queue-input-mode", choices=["auto", "combined", "separate"], default="auto")
    q.add_argument(
        "--queue-deficit-select-mode",
        choices=["position_unrealized_pnl", "total_unrealized_pnl", "total_equity"],
        default="total_equity",
        help="Which loser-side field determines *membership* in the deficit set (rows with this field < 0).",
    )
    q.add_argument(
        "--queue-deficit-value-mode",
        choices=["position_unrealized_pnl", "total_unrealized_pnl", "total_equity"],
        default="total_equity",
        help="Which loser-side field defines the deficit *magnitude* (sum of -value over selected rows).",
    )
    q.set_defaults(func=cmd_queue)

    # replay (includes horizon sweep)
    r = sub.add_parser("replay", help="Run two-pass replay and evaluation-horizon sweep from HyperReplay raw stream.")
    r.add_argument("--gap-ms", type=int, default=5000)
    r.add_argument(
        "--horizons-ms",
        type=lambda s: [int(x) for x in str(s).split(",") if str(x).strip()],
        default=[0, 500, 1000, 2000, 5000],
        help="Comma-separated eval horizons (ms). Must include 0 and 5000 for two-time report defaults.",
    )
    r.add_argument(
        "--half-lives-ms",
        type=lambda s: [int(x) for x in str(s).split(",") if str(x).strip()],
        default=[500, 1000, 2000, 5000],
        help="Half-lives (ms) for discounted overshoot summary.",
    )
    r.add_argument("--band-max-ms", type=int, default=10000)
    r.set_defaults(func=cmd_replay)

    # two-time
    tt = sub.add_parser("two-time", help="Run two-time (undo) classification and write a reconciliation report.")
    tt.add_argument("--gap-ms", type=int, default=5000)
    tt.add_argument("--h0-ms", type=int, default=0)
    tt.add_argument("--h1-ms", type=int, default=5000)
    tt.set_defaults(func=cmd_two_time)

    # headlines
    h = sub.add_parser("headlines", help="Compute and print headline numbers (a–c) + derived naive mapping.")
    h.add_argument("--gap-ms", type=int, default=5000)
    h.add_argument("--horizon-ms", type=int, default=0)
    h.add_argument("--trim-alpha", type=float, default=0.03)
    h.set_defaults(func=cmd_headlines)

    # all
    a = sub.add_parser("all", help="Run queue + replay sweep + headlines + two-time report.")
    a.add_argument("--gap-ms", type=int, default=5000)
    a.add_argument("--gap-seconds", type=float, default=5.0)
    a.add_argument("--follower-decay-beta", type=float, default=5.0)
    a.add_argument("--queue-input-mode", choices=["auto", "combined", "separate"], default="auto")
    a.add_argument(
        "--queue-deficit-select-mode",
        choices=["position_unrealized_pnl", "total_unrealized_pnl", "total_equity"],
        default="total_equity",
    )
    a.add_argument(
        "--queue-deficit-value-mode",
        choices=["position_unrealized_pnl", "total_unrealized_pnl", "total_equity"],
        default="total_equity",
    )
    a.add_argument(
        "--horizons-ms",
        type=lambda s: [int(x) for x in str(s).split(",") if str(x).strip()],
        default=[0, 500, 1000, 2000, 5000],
    )
    a.add_argument(
        "--half-lives-ms",
        type=lambda s: [int(x) for x in str(s).split(",") if str(x).strip()],
        default=[500, 1000, 2000, 5000],
    )
    a.add_argument("--band-max-ms", type=int, default=10000)
    a.add_argument("--h0-ms", type=int, default=0)
    a.add_argument("--h1-ms", type=int, default=5000)
    a.add_argument("--horizon-ms", type=int, default=0)
    a.add_argument("--trim-alpha", type=float, default=0.03)
    a.set_defaults(func=cmd_all)

    # plots
    pl = sub.add_parser("plots", help="Generate figures referenced by OSS/methodology.md.")
    pl.add_argument("--gap-ms", type=int, default=5000)
    pl.add_argument(
        "--canonical-csv",
        type=Path,
        default=None,
        help="Path to canonical realtime CSV (for overcollateralization histogram). Auto-detected if not specified.",
    )
    pl.set_defaults(func=cmd_plots)

    # gap sweep
    gs = sub.add_parser("gap-sweep", help="Wave partition sensitivity: sweep gap_ms (changes segmentation).")
    gs.add_argument(
        "--gaps-ms",
        type=lambda s: [int(x) for x in str(s).split(",") if str(x).strip()],
        default=[250, 500, 1000, 2000, 5000, 10000],
    )
    gs.add_argument("--eval-horizon-ms", type=int, default=0)
    gs.set_defaults(func=cmd_gap_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
