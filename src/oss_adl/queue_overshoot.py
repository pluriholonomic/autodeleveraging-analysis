from __future__ import annotations

"""
Queue overshoot (equity-$) reproduction.

This module is the minimal OSS rewrite of the parts of `src/run_policies.py` that are
required to reproduce the headline "~$650m queue overshoot (equity-$)" number used in
the public debate.

Important nuance (adversary-facing):
------------------------------------
This is *not* a reconstruction of Hyperliquid’s internal matching engine. It is the
stylized **wealth-space queue abstraction** that:

- ranks winners by a score (here: per-position `closed_pnl`),
- allocates a budget greedily in USD haircut space,
- and then compares "budget used" vs a loser-side deficit proxy to compute overshoot.

This is exactly the "queue in wealth space vs contract space" separation emphasized in
the addendum draft:
  https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def assign_cluster_ids(df: pd.DataFrame, *, gap_seconds: float) -> pd.DataFrame:
    """
    Per-coin time clustering using a simple gap rule.

    We keep this identical-in-spirit to `src/run_policies.py:assign_cluster_ids` to make the
    ~650m number reproducible.
    """
    if df.empty:
        out = df.copy()
        out["cluster_id"] = pd.Series(dtype=int)
        return out
    out = df.sort_values(["coin", "block_time"]).copy()
    diffs = out.groupby("coin")["block_time"].diff().dt.total_seconds().fillna(0.0)
    out["cluster_id"] = diffs.gt(float(gap_seconds)).groupby(out["coin"]).cumsum() + 1
    return out


def allocate_priority(*, budget: float, capacities: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Greedy queue allocation in USD haircut space:
    - sort by score desc
    - fully haircut top items up to cap until budget exhausted
    """
    if capacities.size == 0 or budget <= 0.0:
        return np.zeros_like(capacities, dtype=float)
    order = np.argsort(scores)[::-1]
    caps = capacities.astype(float, copy=True)
    haircuts = np.zeros_like(caps, dtype=float)
    remaining = float(budget)
    for idx in order:
        if remaining <= EPS:
            break
        take = min(remaining, caps[idx])
        if take > 0.0:
            haircuts[idx] = take
            remaining -= take
    return haircuts


def capacity_from_row(row: pd.Series) -> float:
    """
    Per-position haircut capacity in the PnL-only model.

    We prefer `position_unrealized_pnl` (per-position unrealized PnL). This matches the
    idea "ADL haircuts live in PnL space, cash is protected".
    """
    if "position_unrealized_pnl" in row.index:
        v = float(pd.to_numeric(row["position_unrealized_pnl"], errors="coerce") or 0.0)
        return max(v, 0.0)
    v = float(pd.to_numeric(row.get("closed_pnl", 0.0), errors="coerce") or 0.0)
    return max(v, 0.0)


def aggregate_winners_for_cluster(cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-position vector used by the wealth-space queue.

    Output columns:
      - identifier: position_id (user+coin)
      - capacity: USD capacity (>=0)
      - queue_score: USD score used for ranking
    """
    if cluster.empty:
        return pd.DataFrame(columns=["identifier", "capacity", "queue_score"])

    winners = cluster[cluster["is_winner"]].copy() if "is_winner" in cluster.columns else cluster.copy()
    winners["closed_pnl"] = pd.to_numeric(winners.get("closed_pnl"), errors="coerce").fillna(0.0)
    winners = winners[winners["closed_pnl"] > 0.0].copy()
    if winners.empty:
        return pd.DataFrame(columns=["identifier", "capacity", "queue_score"])

    # Position identifier: (user, coin). This matches the original pipeline’s choice and is
    # intentionally conservative: it avoids implicitly allowing "one account's other positions"
    # to subsidize capacity here.
    winners["user"] = winners["user"].astype(str)
    winners["coin"] = winners["coin"].astype(str)
    winners["position_id"] = winners["user"] + "_" + winners["coin"]

    # Account-level PnL (for an optional cap) if present.
    account_capacity_map: dict[str, float] = {}
    if "total_unrealized_pnl" in winners.columns:
        account_capacity_map = (
            pd.to_numeric(winners["total_unrealized_pnl"], errors="coerce")
            .fillna(0.0)
            .groupby(winners["user"])
            .first()
            .to_dict()
        )

    rows: list[dict[str, float]] = []
    for _, r in winners.iterrows():
        score = float(r["closed_pnl"])
        if score <= 0.0:
            continue

        pos_cap = capacity_from_row(r)
        if pos_cap <= 0.0:
            continue

        # Optional account-level cap (if positive). This matches the "min(position, account)" logic.
        acct_cap = float(account_capacity_map.get(str(r["user"]), 0.0))
        cap = min(pos_cap, acct_cap) if acct_cap > 0.0 else pos_cap
        if cap <= 0.0:
            continue

        # Key quirk from the original pipeline:
        # enforce cap >= score so that a "score-based budget" can actually be allocated.
        cap = max(cap, score)

        rows.append(
            {
                "identifier": str(r["position_id"]),
                "capacity": float(cap),
                "queue_score": float(score),
            }
        )

    return pd.DataFrame(rows)


def deficit_from_closed_pnl(cluster: pd.DataFrame) -> float:
    """
    Deficit proxy from whatever rows in the cluster carry the loser-side negative quantity.

    For the **combined** input mode (single CSV containing both winners and losers):
    - loser rows are those with `is_negative_equity == True`
    - we map an appropriate loser-side negative quantity into `closed_pnl` (see below),
      then compute deficit from negative `closed_pnl`.

    For the **separate** input mode (winners CSV + liquidations CSV):
    - the liquidation export’s losing-side rows typically have negative `closed_pnl`,
      and we compute deficit from those.

    NOTE: This is a *proxy* object used for the queue-overshoot reproduction, not the two-pass
    production reconstruction. The OSS methodology explicitly distinguishes these.
    """
    if cluster.empty:
        return 0.0
    if "closed_pnl" not in cluster.columns:
        return 0.0
    pnl = pd.to_numeric(cluster["closed_pnl"], errors="coerce").fillna(0.0).to_numpy(float)
    return float((-pnl[pnl < 0.0]).sum())


def deficit_from_loser_fields(
    cluster: pd.DataFrame,
    *,
    select_col: str,
    value_col: str,
) -> float:
    """
    Deficit computed on loser-side rows using two separate columns:

    - selection column: determines which loser rows are "in" the deficit set (rows with select_col < 0)
    - value column: determines the deficit magnitude (sum of -value_col for selected rows where value_col < 0)

    This matches the structure of `src/run_policies.py::compute_deficit`, where selection is
    `closed_pnl < 0` but magnitude may be computed from `total_equity`.
    """
    if cluster.empty:
        return 0.0
    losers = cluster[~cluster["is_winner"]].copy() if "is_winner" in cluster.columns else cluster.copy()
    if losers.empty:
        return 0.0
    if select_col not in losers.columns:
        raise ValueError(f"Missing select_col={select_col} in loser rows")
    if value_col not in losers.columns:
        raise ValueError(f"Missing value_col={value_col} in loser rows")

    s = pd.to_numeric(losers[select_col], errors="coerce").fillna(0.0).to_numpy(float)
    v = pd.to_numeric(losers[value_col], errors="coerce").fillna(0.0).to_numpy(float)
    mask = (s < 0.0) & (v < 0.0)
    return float((-v[mask]).sum())


@dataclass(frozen=True)
class QueueOvershootOutputs:
    per_shock: pd.DataFrame
    summary_totals: pd.DataFrame


def compute_queue_overshoot(
    *,
    winners_csv: Path,
    liquidations_csv: Path | None = None,
    gap_seconds: float = 5.0,
    follower_decay_beta: float = 5.0,
    input_mode: str = "auto",  # auto | combined | separate
    loser_deficit_select_mode: str = "total_equity",  # position_unrealized_pnl | total_unrealized_pnl | total_equity
    loser_deficit_value_mode: str = "total_equity",  # position_unrealized_pnl | total_unrealized_pnl | total_equity
) -> QueueOvershootOutputs:
    """
    Compute the queue overshoot (equity-$) headline table.
    """
    winners_csv = Path(winners_csv)
    liquidations_csv = Path(liquidations_csv) if liquidations_csv is not None else None

    # Two supported input modes:
    #   (A) combined: one CSV contains BOTH winners and losers, distinguished by `is_negative_equity`
    #   (B) separate: winners CSV + separate liquidations CSV (legacy)
    winners_raw = pd.read_csv(winners_csv)

    mode = str(input_mode or "auto").strip().lower()
    if mode not in {"auto", "combined", "separate"}:
        raise ValueError("input_mode must be one of: auto, combined, separate")

    has_neg_flag = "is_negative_equity" in winners_raw.columns
    has_any_neg = False
    if has_neg_flag:
        has_any_neg = (
            winners_raw["is_negative_equity"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "t", "yes", "y"})
            .any()
        )

    use_combined = (mode == "combined") or (mode == "auto" and has_neg_flag and has_any_neg)

    if use_combined:
        # --- Mode (A): combined ---
        df = winners_raw.copy()
        df["coin"] = df["coin"].astype(str).str.upper()
        df["user"] = df["user"].astype(str)
        df["block_time"] = pd.to_datetime(
            pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(np.int64), unit="ms", utc=True
        )
        is_neg = df["is_negative_equity"].astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})

        losers = df[is_neg].copy()
        winners = df[~is_neg].copy()
        losers["is_winner"] = False
        winners["is_winner"] = True
        event_df = pd.concat([losers, winners], ignore_index=True)

    else:
        # --- Mode (B): separate ---
        if liquidations_csv is None:
            raise ValueError("separate queue-overshoot mode requires liquidations_csv")

        winners = winners_raw.copy()
        winners["coin"] = winners["coin"].astype(str).str.upper()
        winners["user"] = winners["user"].astype(str)
        winners["block_time"] = pd.to_datetime(
            pd.to_numeric(winners["time"], errors="coerce").fillna(0).astype(np.int64), unit="ms", utc=True
        )
        winners["is_winner"] = True

        liq = pd.read_csv(liquidations_csv, parse_dates=["block_time"])
        liq["coin"] = liq["coin"].astype(str).str.upper()
        liq["block_time"] = pd.to_datetime(liq["block_time"], utc=True, errors="coerce")
        liq["is_winner"] = False
        event_df = pd.concat([liq, winners], ignore_index=True)

    # Cluster ids must be assigned on the combined winner+loser stream to avoid mismatched
    # ids (see `src/run_policies.py --cluster-assignment combined`).
    event_df = assign_cluster_ids(event_df, gap_seconds=float(gap_seconds))
    event_df = event_df.sort_values(["coin", "cluster_id", "block_time"]).reset_index(drop=True)

    records: list[dict] = []
    # IMPORTANT: in the original simulation code, follower participation is tracked **per coin**.
    # We preserve that here (otherwise you incorrectly leak churn across unrelated markets).
    participation_values: list[float] = []
    # Configure deficit selection/value columns for combined mode.
    sel = str(loser_deficit_select_mode or "total_equity").strip()
    val = str(loser_deficit_value_mode or "total_equity").strip()
    for x in (sel, val):
        if x not in {"position_unrealized_pnl", "total_unrealized_pnl", "total_equity"}:
            raise ValueError(
                "loser_deficit_select_mode / loser_deficit_value_mode must be one of: "
                "position_unrealized_pnl, total_unrealized_pnl, total_equity"
            )

    for coin, df_coin in event_df.groupby("coin", sort=True):
        # Participation mass per "position identifier" (per coin).
        participation: dict[str, float] = {}

        for cluster_id, cluster in df_coin.groupby("cluster_id", sort=True):
            cluster = cluster.sort_values("block_time")
            t0 = cluster["block_time"].min()
            t1 = cluster["block_time"].max()

            if use_combined:
                deficit = deficit_from_loser_fields(cluster, select_col=sel, value_col=val)
            else:
                deficit = deficit_from_closed_pnl(cluster)

            w = aggregate_winners_for_cluster(cluster)
            if w.empty:
                records.append(
                    {
                        "coin": str(coin),
                        "cluster_id": int(cluster_id),
                        "t_start": t0,
                        "t_end": t1,
                        "deficit_usd": float(deficit),
                        "total_capacity_usd": 0.0,
                        "budget_queue_usd": 0.0,
                        "overshoot_queue_usd": 0.0,
                        "residual_queue_usd": float(max(deficit, 0.0)),
                    }
                )
                continue

            ids = w["identifier"].astype(str).to_numpy()
            caps0 = pd.to_numeric(w["capacity"], errors="coerce").fillna(0.0).to_numpy(float)
            scores0 = pd.to_numeric(w["queue_score"], errors="coerce").fillna(0.0).to_numpy(float)

            # Initialize unseen ids.
            for uid in ids:
                participation.setdefault(str(uid), 1.0)

            factors = np.array([participation[str(uid)] for uid in ids], dtype=float)
            caps = caps0 * factors
            scores = scores0 * factors

            total_capacity = float(caps.sum())
            queue_budget_target = float(min(scores.sum(), total_capacity))
            haircuts = allocate_priority(budget=queue_budget_target, capacities=caps, scores=scores)
            budget = float(haircuts.sum())

            # Update participation (churn model): mass <- mass * exp(-beta * haircut/cap)
            beta = float(follower_decay_beta)
            if beta > 0.0:
                for uid, cap_i, h_i in zip(ids, caps, haircuts, strict=True):
                    if cap_i > EPS and h_i > 0.0:
                        frac = float(h_i / cap_i)
                        participation[str(uid)] *= math.exp(-beta * frac)

            overshoot = float(max(budget - deficit, 0.0))
            residual = float(max(deficit - budget, 0.0))

            records.append(
                {
                    "coin": str(coin),
                    "cluster_id": int(cluster_id),
                    "t_start": t0,
                    "t_end": t1,
                    "deficit_usd": float(deficit),
                    "total_capacity_usd": float(total_capacity),
                    "budget_queue_usd": float(budget),
                    "overshoot_queue_usd": float(overshoot),
                    "residual_queue_usd": float(residual),
                }
            )

        participation_values.extend(list(participation.values()))

    per_shock = pd.DataFrame(records)

    summary_totals = pd.DataFrame(
        [
            {
                "policy": "queue",
                "total_residual": float(per_shock["residual_queue_usd"].sum()) if not per_shock.empty else 0.0,
                "total_overshoot": float(per_shock["overshoot_queue_usd"].sum()) if not per_shock.empty else 0.0,
                # Included for compatibility with legacy summaries; this OSS pipeline does not model revenue here.
                "total_revenue_lost": 0.0,
                "avg_participation": float(np.mean(participation_values)) if participation_values else 1.0,
            }
        ]
    )

    return QueueOvershootOutputs(per_shock=per_shock, summary_totals=summary_totals)


def write_queue_outputs(*, out_dir: Path, outputs: QueueOvershootOutputs) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_shock_path = out_dir / "per_shock_metrics_queue.csv"
    summary_path = out_dir / "summary_totals.csv"
    outputs.per_shock.to_csv(per_shock_path, index=False)
    outputs.summary_totals.to_csv(summary_path, index=False)
    return per_shock_path, summary_path
