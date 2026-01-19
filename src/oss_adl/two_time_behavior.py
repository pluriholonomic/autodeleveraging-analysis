from __future__ import annotations

"""
Two-time experiments (strategic vs passive winners) + undo fraction.

This is the minimal OSS port of:
  - `two-time-experiments/classify_accounts.py`
  - `two-time-experiments/check_undo_variants.py`

Why it matters for the controversy:
----------------------------------
If you naively map an *equity-space* overshoot number (approx. $650m) into a "PnL closed" number
using a single average equity/PnL ratio, you typically overpredict the observed PnL-closed
proxy (≈ $45–50m).

The "two-time" analysis shows this gap is not a paradox: the opportunity-cost / horizon
component concentrates on accounts that do *not* actively unwind ("passive"), and those
accounts also exhibit materially different equity/PnL ratios.
"""

import bisect
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class Fill:
    time_ms: int
    delta_pos: float  # +sz if buy, -sz if sell
    is_adl: bool
    px: float
    sz: float
    side: str


def _to_int(x: object) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _to_float(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if (v != v) else float(v)


def normalize_coin(coin: str) -> str:
    c = str(coin or "")
    if c.startswith("hyperliquid:"):
        c = c.replace("hyperliquid:", "")
    return c.upper()


def iter_raw_fills(paths: Iterable[Path]) -> Iterable[tuple[str, dict]]:
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                block = json.loads(line)
                for event in block.get("events") or []:
                    if not isinstance(event, (list, tuple)) or len(event) != 2:
                        continue
                    user, details = event
                    if not isinstance(details, dict):
                        continue
                    yield str(user), details


def load_fills_df(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for user, d in iter_raw_fills(paths):
        coin = normalize_coin(d.get("coin", ""))
        if not coin or coin.startswith("@") or coin == "PURR/USDC":
            continue
        t = _to_int(d.get("time", 0))
        side = str(d.get("side", "") or "")
        sz = abs(_to_float(d.get("sz", 0.0)))
        px = _to_float(d.get("px", 0.0))
        if t <= 0 or sz <= 0.0 or side not in {"B", "A"}:
            continue
        is_adl = str(d.get("dir", "") or "") == "Auto-Deleveraging"
        delta = sz if side == "B" else -sz
        rows.append(
            {
                "time_ms": int(t),
                "user": str(user),
                "coin": str(coin),
                "side": side,
                "sz": float(sz),
                "px": float(px),
                "delta_pos": float(delta),
                "is_adl": bool(is_adl),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["time_ms", "user", "coin", "side", "sz", "px", "delta_pos", "is_adl"])
    return pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)


def build_nonadl_index(fills: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    df = fills[~fills["is_adl"]].copy()
    if df.empty:
        return {}
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (u, c), g in df.groupby(["user", "coin"], sort=False):
        g = g.sort_values("time_ms")
        out[(str(u), str(c))] = (g["time_ms"].to_numpy(np.int64), g["delta_pos"].to_numpy(float))
    return out


def compute_undo_fraction(delta_adl: float, deltas_post: np.ndarray) -> float:
    """
    Opposite-volume undo fraction:
      undone = min(|Δ_ADL|, Σ_{post: opposite sign} |Δ|)
      undo = undone / |Δ_ADL|
    """
    d = float(delta_adl)
    if abs(d) <= EPS:
        return 0.0
    s = 1.0 if d > 0 else -1.0
    remaining = abs(d)
    undone = 0.0
    for dt in deltas_post:
        dt = float(dt)
        if dt * s < 0.0:
            take = min(remaining, abs(dt))
            undone += take
            remaining -= take
            if remaining <= EPS:
                break
    return float(np.clip(undone / abs(d), 0.0, 1.0))


def undo_net_fraction(delta_adl: np.ndarray, net_delta: np.ndarray) -> np.ndarray:
    """
    Net-based undo fraction (robustness):
      undo_net = min(|Δ|, max(0, -sign(Δ)*net_delta)) / |Δ|
    """
    d = np.asarray(delta_adl, dtype=float)
    net = np.asarray(net_delta, dtype=float)
    abs_d = np.abs(d)
    s = np.sign(d)
    undone = np.maximum(0.0, -s * net)
    undone = np.minimum(undone, abs_d)
    out = np.zeros_like(abs_d, dtype=float)
    mask = abs_d > EPS
    out[mask] = undone[mask] / abs_d[mask]
    return out


def overcollateralization_equity_over_pnl(canonical_realtime_csv: Path, *, trim_alpha: float = 0.03) -> dict:
    """
    Per-user first-appearance ratio: total_equity / total_unrealized_pnl for winners with pnl>0.
    """
    cols = ["time", "user", "is_negative_equity", "total_unrealized_pnl", "total_equity"]
    canon = pd.read_csv(canonical_realtime_csv, usecols=cols)
    canon["time"] = pd.to_numeric(canon["time"], errors="coerce").fillna(0).astype(np.int64)
    # Robust bool parsing ("False" string must not become True).
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
        "trimmed_mean": trimmed_mean,
        "p10": float(np.quantile(ratio, 0.10)),
        "p90": float(np.quantile(ratio, 0.90)),
    }


def load_two_pass_user_totals(prod_haircuts_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(prod_haircuts_csv)
    if df.empty:
        return pd.DataFrame(columns=["user", "haircut_prod_usd_total"])
    if "user" not in df.columns or "haircut_prod_usd" not in df.columns:
        raise ValueError(f"{prod_haircuts_csv} missing required columns user/haircut_prod_usd")
    df["haircut_prod_usd"] = pd.to_numeric(df["haircut_prod_usd"], errors="coerce").fillna(0.0)
    out = df.groupby("user", as_index=False)["haircut_prod_usd"].sum()
    return out.rename(columns={"haircut_prod_usd": "haircut_prod_usd_total"})


@dataclass(frozen=True)
class TwoTimeOutputs:
    per_event: pd.DataFrame
    per_user: pd.DataFrame
    strategic_accounts: pd.DataFrame
    passive_accounts: pd.DataFrame
    report_md: str


def classify_strategic_vs_passive(
    *,
    fills_json_paths: list[Path],
    waves_csv: Path,
    canonical_realtime_csv: Path,
    two_pass_h0_prod_haircuts_csv: Path,
    two_pass_h1_prod_haircuts_csv: Path,
    queue_summary_csv: Path | None = None,
    horizon_sweep_csv: Path | None = None,
    pnl_closed_horizon_ms: int = 0,
    windows_ms: list[int] | None = None,
    min_adl_events: int = 5,
    strategic_min_share_traded_60s: float = 0.5,
    strategic_min_undo_frac_60s: float = 0.25,
    passive_max_share_traded_300s: float = 0.05,
    trim_alpha: float = 0.03,
) -> TwoTimeOutputs:
    """
    Core classifier + reconcile report.
    """
    fills = load_fills_df(fills_json_paths)
    nonadl_index = build_nonadl_index(fills)

    # Load wave intervals for labeling ADL events.
    waves = pd.read_csv(waves_csv).copy()
    if not {"wave", "t_start_ms", "t_end_ms"} <= set(waves.columns):
        raise ValueError(f"{waves_csv} missing required columns wave,t_start_ms,t_end_ms")
    waves["wave"] = pd.to_numeric(waves["wave"], errors="coerce").fillna(-1).astype(int)
    waves["t_start_ms"] = pd.to_numeric(waves["t_start_ms"], errors="coerce").fillna(0).astype(np.int64)
    waves["t_end_ms"] = pd.to_numeric(waves["t_end_ms"], errors="coerce").fillna(0).astype(np.int64)
    intervals: list[tuple[int, int, int]] = [
        (int(w), int(t0), int(t1))
        for w, t0, t1 in waves[["wave", "t_start_ms", "t_end_ms"]].itertuples(index=False, name=None)
    ]
    intervals.sort(key=lambda x: x[1])

    def wave_for_time(t_ms: int) -> int | None:
        for w, t0, t1 in intervals:
            if t0 <= int(t_ms) <= t1:
                return int(w)
        return None

    # ADL events = ADL fills in raw stream.
    adl = fills[fills["is_adl"]].copy()
    adl = adl.sort_values("time_ms").reset_index(drop=True)
    adl["event_id"] = np.arange(len(adl), dtype=int)
    adl["wave"] = adl["time_ms"].apply(lambda t: wave_for_time(int(t)))
    # Pandas will upcast Optional[int] -> float with NaN. Normalize to int with -1 sentinel.
    adl["wave"] = pd.to_numeric(adl["wave"], errors="coerce").fillna(-1).astype(int)

    # Per-event reaction metrics.
    if windows_ms is None:
        windows_ms = [5000, 60000, 300000]
    win_set = sorted({int(w) for w in windows_ms})
    ev_rows: list[dict] = []
    for r in adl.itertuples(index=False):
        user = str(r.user)
        coin = str(r.coin)
        t0 = int(r.time_ms)
        delta_adl = float(r.delta_pos)
        key = (user, coin)
        times, deltas = nonadl_index.get(key, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)))
        # Post-ADL deltas in each window.
        row = {
            "event_id": int(r.event_id),
            "wave": int(r.wave),
            "time_ms": int(t0),
            "user": user,
            "coin": coin,
            "delta_adl_pos": float(delta_adl),
        }

        # Latency to first non-ADL fill
        j = bisect.bisect_right(times, t0) if times.size else 0
        if j < times.size:
            row["latency_to_first_nonadl_fill_ms"] = int(times[j] - t0)
        else:
            row["latency_to_first_nonadl_fill_ms"] = None

        for W in win_set:
            t_hi = t0 + int(W)
            if times.size == 0:
                row[f"n_nonadl_fills_{W}ms"] = 0
                row[f"abs_qty_nonadl_{W}ms"] = 0.0
                row[f"net_delta_pos_nonadl_{W}ms"] = 0.0
                row[f"undo_frac_{W}ms"] = 0.0
                continue
            lo = bisect.bisect_right(times, t0)
            hi = bisect.bisect_right(times, t_hi)
            sub = deltas[lo:hi]
            row[f"n_nonadl_fills_{W}ms"] = int(sub.size)
            row[f"abs_qty_nonadl_{W}ms"] = float(np.sum(np.abs(sub))) if sub.size else 0.0
            row[f"net_delta_pos_nonadl_{W}ms"] = float(np.sum(sub)) if sub.size else 0.0
            row[f"undo_frac_{W}ms"] = float(compute_undo_fraction(delta_adl, sub))
        ev_rows.append(row)

    per_event = pd.DataFrame(ev_rows)

    # Aggregate per user.
    per_user_rows: list[dict] = []
    if not per_event.empty:
        g = per_event.groupby("user", sort=False)
        for user, sub in g:
            row = {"user": str(user), "n_adl_events": int(len(sub))}
            lat = sub["latency_to_first_nonadl_fill_ms"].dropna()
            row["median_latency_ms"] = int(np.median(lat)) if len(lat) else None
            for W in win_set:
                nf = pd.to_numeric(sub[f"n_nonadl_fills_{W}ms"], errors="coerce").fillna(0.0)
                row[f"share_any_nonadl_fill_{W}ms"] = float(np.mean(nf > 0)) if len(nf) else 0.0
                row[f"mean_undo_frac_{W}ms"] = (
                    float(np.mean(pd.to_numeric(sub[f"undo_frac_{W}ms"], errors="coerce").fillna(0.0)))
                    if len(sub)
                    else 0.0
                )
            # convenience score for 60s window
            if 60000 in win_set:
                row["strategic_score"] = float(row["share_any_nonadl_fill_60000ms"] * row["mean_undo_frac_60000ms"])
            per_user_rows.append(row)

    per_user = pd.DataFrame(per_user_rows)

    # Behavioral labels
    strategic: pd.DataFrame
    passive: pd.DataFrame
    if per_user.empty:
        strategic = per_user.copy()
        passive = per_user.copy()
    else:
        strategic = per_user[
            (per_user["n_adl_events"] >= int(min_adl_events))
            & (per_user.get("share_any_nonadl_fill_60000ms", 0.0) >= float(strategic_min_share_traded_60s))
            & (per_user.get("mean_undo_frac_60000ms", 0.0) >= float(strategic_min_undo_frac_60s))
        ].copy()
        passive = per_user[
            (per_user["n_adl_events"] >= int(min_adl_events))
            & (per_user.get("share_any_nonadl_fill_300000ms", 1.0) <= float(passive_max_share_traded_300s))
        ].copy()

    strategic = (
        strategic.sort_values(["strategic_score", "n_adl_events"], ascending=False, na_position="last")
        if not strategic.empty
        else strategic
    )
    passive = passive.sort_values(["n_adl_events"], ascending=False) if not passive.empty else passive

    # --- Group reconciliation pieces ---
    queue_overshoot_usd: float | None = None
    prod_overshoot_vs_needed_usd: float | None = None
    implied_reconcile_ratio: float | None = None
    if queue_summary_csv is not None and Path(queue_summary_csv).exists():
        q = pd.read_csv(queue_summary_csv)
        qrow = q[q["policy"].astype(str) == "queue"]
        if not qrow.empty and "total_overshoot" in qrow.columns:
            queue_overshoot_usd = float(pd.to_numeric(qrow.iloc[0]["total_overshoot"], errors="coerce"))

    if horizon_sweep_csv is not None and Path(horizon_sweep_csv).exists():
        hs = pd.read_csv(horizon_sweep_csv)
        hs["eval_horizon_ms"] = pd.to_numeric(hs["eval_horizon_ms"], errors="coerce").fillna(0).astype(int)
        row = hs[hs["eval_horizon_ms"] == int(pnl_closed_horizon_ms)]
        if not row.empty and "total_overshoot_prod_minus_needed_usd" in row.columns:
            prod_overshoot_vs_needed_usd = float(
                pd.to_numeric(row.iloc[0]["total_overshoot_prod_minus_needed_usd"], errors="coerce")
            )

    if (
        (queue_overshoot_usd is not None)
        and (prod_overshoot_vs_needed_usd is not None)
        and (prod_overshoot_vs_needed_usd > EPS)
    ):
        implied_reconcile_ratio = float(queue_overshoot_usd / prod_overshoot_vs_needed_usd)

    # Overcollateralization by group
    oc_all = overcollateralization_equity_over_pnl(canonical_realtime_csv, trim_alpha=float(trim_alpha))

    # Time-discount by group using two-pass per-user totals at two horizons.
    h0 = load_two_pass_user_totals(two_pass_h0_prod_haircuts_csv).rename(columns={"haircut_prod_usd_total": "h0"})
    h1 = load_two_pass_user_totals(two_pass_h1_prod_haircuts_csv).rename(columns={"haircut_prod_usd_total": "h1"})
    hh = h0.merge(h1, on="user", how="outer").fillna(0.0)
    hh["delta_haircut"] = hh["h1"] - hh["h0"]

    def _group_time_discount(users: pd.Series) -> dict:
        u = set(users.astype(str).tolist())
        sub = hh[hh["user"].astype(str).isin(u)].copy()
        if sub.empty:
            return {"n_users": 0, "sum_delta": 0.0, "mean_delta": 0.0, "median_delta": 0.0}
        return {
            "n_users": int(len(sub)),
            "sum_delta": float(sub["delta_haircut"].sum()),
            "mean_delta": float(sub["delta_haircut"].mean()),
            "median_delta": float(sub["delta_haircut"].median()),
        }

    disc_strat = (
        _group_time_discount(strategic["user"])
        if not strategic.empty
        else {"n_users": 0, "sum_delta": 0.0, "mean_delta": 0.0, "median_delta": 0.0}
    )
    disc_pass = (
        _group_time_discount(passive["user"])
        if not passive.empty
        else {"n_users": 0, "sum_delta": 0.0, "mean_delta": 0.0, "median_delta": 0.0}
    )

    # Equity/PnL by group (using canonical file)
    def _equity_pnl_by_group(users: pd.Series, trim_alpha_local: float) -> dict:
        cols = ["time", "user", "is_negative_equity", "total_unrealized_pnl", "total_equity"]
        canon = pd.read_csv(canonical_realtime_csv, usecols=cols)
        canon["time"] = pd.to_numeric(canon["time"], errors="coerce").fillna(0).astype(np.int64)
        if canon["is_negative_equity"].dtype != bool:
            canon["is_negative_equity"] = (
                canon["is_negative_equity"].astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})
            )
        canon["total_unrealized_pnl"] = pd.to_numeric(canon["total_unrealized_pnl"], errors="coerce").fillna(0.0)
        canon["total_equity"] = pd.to_numeric(canon["total_equity"], errors="coerce").fillna(0.0)
        u = set(users.astype(str).tolist())
        c = canon[canon["user"].astype(str).isin(u)].copy()
        c = c[(~c["is_negative_equity"]) & (c["total_unrealized_pnl"] > 0.0) & (c["total_equity"] > 0.0)].copy()
        c = c.sort_values("time").groupby("user", as_index=False).first()
        ratio = (c["total_equity"] / c["total_unrealized_pnl"]).to_numpy(float)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            return {"n_users": 0, "median": float("nan"), "trimmed_mean": float("nan")}
        a = float(np.clip(trim_alpha_local, 0.0, 0.49))
        lo = float(np.quantile(ratio, a))
        hi = float(np.quantile(ratio, 1.0 - a))
        trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
        return {
            "n_users": int(ratio.size),
            "median": float(np.median(ratio)),
            "trimmed_mean": float(np.mean(trimmed)) if trimmed.size else float("nan"),
        }

    oc_strat = (
        _equity_pnl_by_group(strategic["user"], trim_alpha)
        if not strategic.empty
        else {"n_users": 0, "median": float("nan"), "trimmed_mean": float("nan")}
    )
    oc_pass = (
        _equity_pnl_by_group(passive["user"], trim_alpha)
        if not passive.empty
        else {"n_users": 0, "median": float("nan"), "trimmed_mean": float("nan")}
    )

    report_lines = []
    report_lines.append("## Strategic vs passive winners (behavioral classification)")
    report_lines.append(f"- **Fills loaded**: {len(fills):,}")
    report_lines.append(f"- **ADL fills (winner-side)**: {int(adl.shape[0]):,}")
    report_lines.append(f"- **Per-user reaction rows**: {int(per_user.shape[0]):,}")
    report_lines.append("")
    report_lines.append("### Classification thresholds")
    report_lines.append(f"- **min ADL events**: {int(min_adl_events)}")
    report_lines.append(
        f"- **strategic**: share_any_nonadl_fill_60s >= {strategic_min_share_traded_60s:.2f} and mean_undo_frac_60s >= {strategic_min_undo_frac_60s:.2f}"
    )
    report_lines.append(f"- **passive**: share_any_nonadl_fill_300s <= {passive_max_share_traded_300s:.2f}")
    report_lines.append("")
    report_lines.append("### Headline reconciliation (sanity)")
    if queue_overshoot_usd is not None:
        report_lines.append(f"- **queue overshoot (equity-$)**: {queue_overshoot_usd / 1e6:.2f}M")
    if prod_overshoot_vs_needed_usd is not None:
        report_lines.append(
            f"- **production overshoot-vs-needed (PnL-$, horizon={int(pnl_closed_horizon_ms)}ms)**: {prod_overshoot_vs_needed_usd / 1e6:.2f}M"
        )
    if implied_reconcile_ratio is not None:
        report_lines.append(f"- **implied equity/PnL ratio to reconcile** (queue/prod): {implied_reconcile_ratio:.2f}×")
    report_lines.append("")
    report_lines.append("### Overcollateralization (equity / PnL)")
    report_lines.append(
        f"- all winners (trimmed mean {trim_alpha:.0%}): **{oc_all.get('trimmed_mean', float('nan')):.2f}×**"
    )
    report_lines.append("")
    report_lines.append("### Overcollateralization (equity / PnL) by group")
    report_lines.append("| group | n users | median | trimmed mean (3%) |")
    report_lines.append("|---|---:|---:|---:|")
    report_lines.append(
        f"| strategic | {oc_strat['n_users']} | {oc_strat['median']:.2f}× | {oc_strat['trimmed_mean']:.2f}× |"
    )
    report_lines.append(
        f"| passive | {oc_pass['n_users']} | {oc_pass['median']:.2f}× | {oc_pass['trimmed_mean']:.2f}× |"
    )
    report_lines.append("")
    report_lines.append("### Group comparison (time-discount component via two-pass)")
    report_lines.append("| group | n users | sum Δhaircut (h1-h0) | mean Δhaircut | median Δhaircut |")
    report_lines.append("|---|---:|---:|---:|---:|")
    report_lines.append(
        f"| strategic | {disc_strat['n_users']} | {disc_strat['sum_delta']:.2f} | {disc_strat['mean_delta']:.2f} | {disc_strat['median_delta']:.2f} |"
    )
    report_lines.append(
        f"| passive | {disc_pass['n_users']} | {disc_pass['sum_delta']:.2f} | {disc_pass['mean_delta']:.2f} | {disc_pass['median_delta']:.2f} |"
    )
    report_lines.append("")
    report_lines.append("### Top accounts")
    report_lines.append("- **strategic**: see `strategic_accounts.csv`")
    report_lines.append("- **passive**: see `passive_accounts.csv`")
    report_md = "\n".join(report_lines) + "\n"

    return TwoTimeOutputs(
        per_event=per_event,
        per_user=per_user,
        strategic_accounts=strategic,
        passive_accounts=passive,
        report_md=report_md,
    )


def write_two_time_outputs(*, out_dir: Path, outputs: TwoTimeOutputs) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_adl_event_metrics.csv": out_dir / "per_adl_event_metrics.csv",
        "per_user_reaction_metrics.csv": out_dir / "per_user_reaction_metrics.csv",
        "strategic_accounts.csv": out_dir / "strategic_accounts.csv",
        "passive_accounts.csv": out_dir / "passive_accounts.csv",
        "report.md": out_dir / "report.md",
    }
    outputs.per_event.to_csv(paths["per_adl_event_metrics.csv"], index=False)
    outputs.per_user.to_csv(paths["per_user_reaction_metrics.csv"], index=False)
    outputs.strategic_accounts.to_csv(paths["strategic_accounts.csv"], index=False)
    outputs.passive_accounts.to_csv(paths["passive_accounts.csv"], index=False)
    paths["report.md"].write_text(outputs.report_md)
    return paths
