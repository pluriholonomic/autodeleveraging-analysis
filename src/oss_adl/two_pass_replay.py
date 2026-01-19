from __future__ import annotations

"""
Two-pass replay (production haircuts) + evaluation-horizon sweep.

This is an OSS-focused port of the core logic in:
  - `scripts/reconstruct_two_pass_full_equity_delta.py`
  - `scripts/sweep_eval_horizon_ms.py`
  - `scripts/summarize_overshoot_robustness.py`

We keep the semantics intentionally aligned with the addendum draft:
  https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md

Key object (wave t, wealth space):
  H_t^prod = Σ_{i∈winners(t)} (e_{t,end}^{noADL}(i) - e_{t,end}^{ADL}(i))_+

We construct e^{noADL} by re-playing the realized event stream with ADL fills *removed from
state updates* but *kept in the realized price path*.
"""

import json
import tarfile
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

EPS = 1e-12

# HyperReplay snapshot time used in the upstream dataset (Oct 10–11 event).
SNAPSHOT_TIME_MS = 1760126694218

# Conservative end-time guard for the dataset window.
ADL_END_TIME_MS = 1760131620000


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def assemble_parts(raw_dir: Path, part_prefix: str, output_name: str) -> Path:
    """
    Some distributions ship `*.tar.xz.part-*` to avoid very large single files.
    """
    target = raw_dir / output_name
    if target.exists():
        return target
    parts = sorted(raw_dir.glob(f"{part_prefix}.part-*"))
    if not parts:
        return target
    with target.open("wb") as out_f:
        for part in parts:
            out_f.write(part.read_bytes())
    return target


def ensure_clearinghouse_inputs(raw_dir: Path) -> None:
    """
    Ensure the minimal HyperReplay raw inputs exist; if not, extract from the tar archive.
    """
    archive_prefix = "clearinghouse_snapshot_20251010.tar.xz"
    archive = assemble_parts(raw_dir, archive_prefix, archive_prefix)
    required = [
        raw_dir / "20_fills.json",
        raw_dir / "21_fills.json",
        raw_dir / "20_misc.json",
        raw_dir / "21_misc.json",
        raw_dir / "account_value_snapshot_758750000_1760126694218.json",
        raw_dir / "perp_positions_by_market_758750000_1760126694218.json",
        raw_dir / "spot_balances__758750000_1760126694218.json",
    ]
    if all(p.exists() for p in required):
        return
    if not archive.exists():
        missing = [p.name for p in required if not p.exists()]
        raise FileNotFoundError("Missing clearinghouse inputs: " + ", ".join(missing))
    with tarfile.open(archive, "r:xz") as tar:
        # Security: validate paths before extraction to prevent path traversal
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(raw_dir)


def compute_unrealized_pnl_for_state(state: dict, last_prices: dict[str, float]) -> float:
    """
    Mark-to-last-fill-price unrealized PnL, per perp position.
    """
    total_unrealized = 0.0
    positions = state.get("positions", {}) or {}
    for coin, pos in positions.items():
        size = float(pos.get("size", 0.0) or 0.0)
        if abs(size) <= EPS:
            continue
        px = float(last_prices.get(coin, 0.0) or 0.0)
        if px <= 0.0:
            continue
        entry = float(pos.get("entry_price", px) or px)
        if entry <= 0.0:
            continue
        if size > 0:
            total_unrealized += size * (px - entry)
        else:
            total_unrealized += abs(size) * (entry - px)
    return float(total_unrealized)


def load_baseline_states(raw_dir: Path) -> tuple[dict[str, dict], dict[str, float]]:
    """
    Baseline snapshot -> (account_states, last_prices).

    IMPORTANT: We convert to a "cash-only baseline" by subtracting snapshot unrealized PnL
    from account_value. This makes "PnL is what gets haircutted" experiments consistent.
    """
    with (raw_dir / "account_value_snapshot_758750000_1760126694218.json").open() as f:
        account_values = json.load(f)
    with (raw_dir / "perp_positions_by_market_758750000_1760126694218.json").open() as f:
        positions_by_market = json.load(f)

    account_states: dict[str, dict] = {}
    last_prices: dict[str, float] = {}

    for acc in account_values:
        account_states[str(acc["user"])] = {
            "account_value": float(acc["account_value"]),
            "positions": {},
            "snapshot_time": SNAPSHOT_TIME_MS,
            "initial_unrealized": 0.0,
        }

    for market in positions_by_market:
        coin = str(market["market_name"]).replace("hyperliquid:", "")
        for pos in market.get("positions", []):
            user = str(pos["user"])
            size = float(pos["size"])
            entry_price = float(pos["entry_price"])
            notional = float(pos["notional_size"])
            mark_price = abs(notional / size) if size else entry_price

            if user not in account_states:
                account_states[user] = {
                    "account_value": float(pos.get("account_value", 0.0)),
                    "positions": {},
                    "snapshot_time": SNAPSHOT_TIME_MS,
                    "initial_unrealized": 0.0,
                }

            account_states[user]["positions"][coin] = {
                "size": size,
                "entry_price": entry_price,
                "notional": notional,
                "mark_price": mark_price,
            }

            if size:
                account_states[user]["initial_unrealized"] += size * (mark_price - entry_price)
            if mark_price and mark_price > 0:
                last_prices.setdefault(coin, float(mark_price))

    # cash-only baseline: remove snapshot unrealized PnL from account value
    for st in account_states.values():
        st["initial_account_value"] = st["account_value"]
        st["account_value"] = float(st["account_value"]) - float(st.get("initial_unrealized", 0.0))

    return account_states, last_prices


def _parse_iso_utc_ms(s: str) -> int:
    """
    HyperReplay misc events use ISO8601 timestamps; normalize to ms.
    """
    time_str = str(s).replace("Z", "+00:00")
    if "." in time_str:
        parts = time_str.split(".")
        # trim to microseconds if needed
        time_str = parts[0] + "." + parts[1][:6] + parts[1][9:]
    event_time = datetime.fromisoformat(time_str)
    return int(event_time.timestamp() * 1000)


def load_events(raw_dir: Path, *, snapshot_time_ms: int, end_time_ms: int) -> list[dict]:
    """
    Load clearinghouse fills + misc events into a single time-ordered stream.
    """
    all_events: list[dict] = []

    # fills from clearinghouse json (line-delimited blocks)
    for hour_file in ["20_fills.json", "21_fills.json"]:
        with (raw_dir / hour_file).open() as f:
            for line in f:
                block = json.loads(line)
                for event in block.get("events") or []:
                    user, details = event
                    coin = str(details.get("coin") or "")
                    if coin.startswith("@") or coin == "PURR/USDC":
                        continue
                    all_events.append(
                        {
                            "type": "fill",
                            "time": int(details["time"]),
                            "user": str(user),
                            "coin": coin,
                            "price": float(details["px"]),
                            "size": float(details["sz"]),
                            "side": str(details.get("side", "")),
                            "direction": str(details.get("dir", "Unknown")),
                            "closedPnl": float(details.get("closedPnl", 0.0) or 0.0),
                            "fee": float(details.get("fee", 0.0) or 0.0),
                            "liquidation_data": details.get("liquidation", None),
                        }
                    )

    # misc events
    for hour_file in ["20_misc.json", "21_misc.json"]:
        with (raw_dir / hour_file).open() as f:
            for line in f:
                block = json.loads(line)
                for event in block.get("events") or []:
                    timestamp = _parse_iso_utc_ms(event.get("time", "1970-01-01T00:00:00Z"))
                    inner = event.get("inner", {}) or {}

                    if "Funding" in inner:
                        for delta in inner["Funding"].get("deltas", []):
                            all_events.append(
                                {
                                    "type": "funding",
                                    "time": timestamp,
                                    "user": str(delta.get("user") or ""),
                                    "funding_amount": float(
                                        delta.get("usd")
                                        if delta.get("usd") is not None
                                        else (
                                            delta.get("usdDelta")
                                            if delta.get("usdDelta") is not None
                                            else delta.get("amount", 0.0)
                                        )
                                    ),
                                }
                            )
                    elif "UserNonFundingLedgerUpdate" in inner:
                        u = inner["UserNonFundingLedgerUpdate"]
                        user = str(u.get("user") or "")
                        delta = u.get("delta", {}) or {}
                        if "Deposit" in delta:
                            all_events.append(
                                {
                                    "type": "deposit",
                                    "time": timestamp,
                                    "user": user,
                                    "amount": float(delta["Deposit"].get("usd", 0.0)),
                                }
                            )
                        elif "Withdraw" in delta:
                            all_events.append(
                                {
                                    "type": "withdrawal",
                                    "time": timestamp,
                                    "user": user,
                                    "amount": float(delta["Withdraw"].get("usd", 0.0)),
                                }
                            )
                        elif "Transfer" in delta:
                            all_events.append(
                                {
                                    "type": "transfer",
                                    "time": timestamp,
                                    "user": user,
                                    "amount": float(delta["Transfer"].get("usd", 0.0)),
                                }
                            )
                    elif "UserAccountValue" in inner:
                        u = inner["UserAccountValue"]
                        all_events.append(
                            {
                                "type": "account_value_override",
                                "time": timestamp,
                                "user": str(u.get("user") or ""),
                                "value": float(u.get("accountValue", 0.0)),
                            }
                        )

    # filter by time window
    all_events = [e for e in all_events if snapshot_time_ms <= int(e["time"]) <= end_time_ms]
    all_events.sort(key=lambda x: int(x["time"]))
    return all_events


@dataclass(frozen=True)
class TwoPassOutputs:
    by_wave: pd.DataFrame
    prod_haircuts: pd.DataFrame
    winner_start_positions: pd.DataFrame
    coin_gap_per_contract: pd.DataFrame
    summary: dict


def run_two_pass_for_waves(
    *,
    hyperreplay_root: Path,
    loser_waves_csv: Path,
    out_dir: Path,
    eval_horizon_ms: int = 0,
) -> TwoPassOutputs:
    """
    Run the two-pass replay for the given loser wave intervals.
    """
    hr_root = Path(hyperreplay_root)
    raw_dir = hr_root / "data" / "raw"
    ensure_clearinghouse_inputs(raw_dir)

    baseline_states, baseline_prices = load_baseline_states(raw_dir)
    events = load_events(raw_dir, snapshot_time_ms=SNAPSHOT_TIME_MS, end_time_ms=ADL_END_TIME_MS)

    # Canonical ADL table is used purely for winner sets (and per-wave coin sets).
    canonical_realtime = hr_root / "data" / "canonical" / "adl_detailed_analysis_REALTIME.csv"
    adl_rows = pd.read_csv(canonical_realtime, usecols=["time", "user", "coin", "liquidated_user"])
    adl_rows["time"] = pd.to_numeric(adl_rows["time"], errors="coerce").fillna(0).astype(np.int64)
    adl_rows["user"] = adl_rows["user"].astype(str)
    adl_rows["coin"] = adl_rows["coin"].astype(str)
    adl_rows["liquidated_user"] = adl_rows["liquidated_user"].fillna("").astype(str)

    intervals = pd.read_csv(loser_waves_csv).sort_values(["t_start_ms", "t_end_ms"]).reset_index(drop=True)
    if not {"wave", "t_start_ms", "t_end_ms", "deficit_usd"} <= set(intervals.columns):
        raise ValueError(f"{loser_waves_csv} must contain wave,t_start_ms,t_end_ms,deficit_usd")
    intervals["segment"] = intervals["wave"].astype(int)

    # Precompute winner sets per wave.
    winners_by_seg: dict[int, set[str]] = {}
    coins_by_seg: dict[int, set[str]] = {}
    for _, r in intervals.iterrows():
        seg = int(r["segment"])
        t0 = int(r["t_start_ms"])
        t1 = int(r["t_end_ms"])
        sub = adl_rows[(adl_rows["time"] >= t0) & (adl_rows["time"] <= t1)]
        winners_by_seg[seg] = set(sub["user"].unique().tolist())
        coins_by_seg[seg] = set(sub["coin"].unique().tolist())
    all_winners: set[str] = set().union(*winners_by_seg.values()) if winners_by_seg else set()

    # Segment lookup (waves are disjoint): time -> seg.
    seg_intervals: list[tuple[int, int, int]] = list(
        zip(
            intervals["segment"].astype(int).tolist(),
            intervals["t_start_ms"].astype(int).tolist(),
            intervals["t_end_ms"].astype(int).tolist(),
            strict=True,
        )
    )
    seg_intervals.sort(key=lambda x: x[1])

    def seg_for_time(t_ms: int) -> int | None:
        for seg, t0, t1 in seg_intervals:
            if t0 <= t_ms <= t1:
                return int(seg)
        return None

    # Needed budget per wave (bankruptcy-gap proxy):
    #   needed = Σ |markPx - px| * |sz| over ADL fills in the wave.
    budget_needed_by_seg: dict[int, float] = {int(seg): 0.0 for seg, _, _ in seg_intervals}
    gap_usd_x_qty_by_seg_coin: dict[int, dict[str, float]] = {int(seg): {} for seg, _, _ in seg_intervals}
    qty_by_seg_coin: dict[int, dict[str, float]] = {int(seg): {} for seg, _, _ in seg_intervals}
    for fn in ["20_fills.json", "21_fills.json"]:
        with (raw_dir / fn).open() as f:
            for line in f:
                block = json.loads(line)
                for _user, details in block.get("events") or []:
                    if details.get("dir") != "Auto-Deleveraging":
                        continue
                    t = int(details.get("time") or 0)
                    seg = seg_for_time(t)
                    if seg is None:
                        continue
                    liq = details.get("liquidation")
                    if not isinstance(liq, dict):
                        continue
                    mark_px = float(liq.get("markPx") or 0.0)
                    exec_px = float(details.get("px") or 0.0)
                    q = abs(float(details.get("sz") or 0.0))
                    if mark_px <= 0.0 or exec_px <= 0.0 or q <= 0.0:
                        continue
                    gap = abs(mark_px - exec_px)
                    budget_needed_by_seg[seg] = budget_needed_by_seg.get(seg, 0.0) + gap * q
                    coin = str(details.get("coin") or "")
                    if coin and (not coin.startswith("@")) and coin != "PURR/USDC":
                        gap_usd_x_qty_by_seg_coin.setdefault(seg, {}).setdefault(coin, 0.0)
                        qty_by_seg_coin.setdefault(seg, {}).setdefault(coin, 0.0)
                        gap_usd_x_qty_by_seg_coin[seg][coin] += gap * q
                        qty_by_seg_coin[seg][coin] += q

    # Replay to wave-start checkpoints to get initial states for each wave.
    checkpoints = sorted({int(x) for x in intervals["t_start_ms"].tolist()})

    def replay_up_to_checkpoints(checkpoints_ms: list[int]) -> dict[int, tuple[dict[str, dict], dict[str, float]]]:
        checkpoints_ms = sorted({int(x) for x in checkpoints_ms})
        states: dict[str, dict] = {}
        for u in set(all_winners):
            if u in baseline_states:
                states[u] = deepcopy(baseline_states[u])
            else:
                states[u] = {"account_value": 0.0, "positions": {}, "snapshot_time": SNAPSHOT_TIME_MS}
        last_prices = dict(baseline_prices)

        out: dict[int, tuple[dict[str, dict], dict[str, float]]] = {}
        idx = 0
        for e in events:
            t = int(e["time"])
            while idx < len(checkpoints_ms) and t >= checkpoints_ms[idx]:
                out[int(checkpoints_ms[idx])] = (deepcopy(states), dict(last_prices))
                idx += 1
            if idx >= len(checkpoints_ms):
                break

            et = e.get("type")
            user = str(e.get("user"))
            if et == "fill":
                coin = str(e["coin"])
                price = float(e["price"])
                last_prices[coin] = price
                if user not in states:
                    continue
                states[user]["account_value"] += float(e.get("closedPnl", 0.0))
                states[user]["account_value"] -= float(e.get("fee", 0.0))
                if coin not in states[user]["positions"]:
                    states[user]["positions"][coin] = {"size": 0.0, "entry_price": price, "notional": 0.0}
                fill_sz = float(e.get("size", 0.0))
                side = str(e.get("side", ""))
                curr_size = float(states[user]["positions"][coin].get("size", 0.0) or 0.0)
                if side == "B":
                    curr_size += fill_sz
                elif side == "A":
                    curr_size -= fill_sz
                states[user]["positions"][coin]["size"] = float(curr_size)
            elif et == "funding":
                if user in states:
                    states[user]["account_value"] += float(e.get("funding_amount", 0.0))
            elif et == "deposit":
                if user in states:
                    states[user]["account_value"] += float(e.get("amount", 0.0))
            elif et == "withdrawal":
                if user in states:
                    states[user]["account_value"] -= float(e.get("amount", 0.0))
            elif et == "transfer":
                if user in states:
                    states[user]["account_value"] += float(e.get("amount", 0.0))
            elif et == "account_value_override":
                if user in states:
                    states[user]["account_value"] = float(e.get("value", states[user]["account_value"]))

        while idx < len(checkpoints_ms):
            out[int(checkpoints_ms[idx])] = (deepcopy(states), dict(last_prices))
            idx += 1
        return out

    start_snaps = replay_up_to_checkpoints(checkpoints)

    def replay_wave(
        *,
        init_states: dict[str, dict],
        init_prices: dict[str, float],
        t_start: int,
        t_end: int,
        t_eval_end: int,
        users_of_interest: set[str],
        disable_adl: bool,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """
        Replay [t_start, t_eval_end]. For t<=t_end we update state; after t_end we only update last_prices.
        """
        states = deepcopy(init_states)
        last_prices = dict(init_prices)
        for e in events:
            t = int(e["time"])
            if t < int(t_start):
                continue
            if t > int(t_eval_end):
                break
            et = e.get("type")
            user = str(e.get("user"))

            if et == "fill":
                coin = str(e["coin"])
                price = float(e["price"])
                last_prices[coin] = price
                if t > int(t_end):
                    continue
                is_adl = e.get("direction") == "Auto-Deleveraging"
                if disable_adl and is_adl:
                    continue
                if user not in users_of_interest:
                    continue
                if user not in states:
                    states[user] = {"account_value": 0.0, "positions": {}, "snapshot_time": t}
                states[user]["account_value"] += float(e.get("closedPnl", 0.0))
                states[user]["account_value"] -= float(e.get("fee", 0.0))
                if coin not in states[user]["positions"]:
                    states[user]["positions"][coin] = {"size": 0.0, "entry_price": price, "notional": 0.0}
                fill_sz = float(e.get("size", 0.0))
                side = str(e.get("side", ""))
                curr_size = float(states[user]["positions"][coin].get("size", 0.0) or 0.0)
                if side == "B":
                    curr_size += fill_sz
                elif side == "A":
                    curr_size -= fill_sz
                states[user]["positions"][coin]["size"] = float(curr_size)
            elif et == "funding":
                if t <= int(t_end) and user in users_of_interest:
                    states[user]["account_value"] += float(e.get("funding_amount", 0.0))
            elif et == "deposit":
                if t <= int(t_end) and user in users_of_interest:
                    states[user]["account_value"] += float(e.get("amount", 0.0))
            elif et == "withdrawal":
                if t <= int(t_end) and user in users_of_interest:
                    states[user]["account_value"] -= float(e.get("amount", 0.0))
            elif et == "transfer":
                if t <= int(t_end) and user in users_of_interest:
                    states[user]["account_value"] += float(e.get("amount", 0.0))
            elif et == "account_value_override":
                if t <= int(t_end) and user in users_of_interest:
                    states[user]["account_value"] = float(e.get("value", states[user]["account_value"]))

        equity_end: dict[str, float] = {}
        upnl_end: dict[str, float] = {}
        for u in users_of_interest:
            st = states.get(u)
            if st is None:
                continue
            av = float(st.get("account_value", 0.0) or 0.0)
            upnl = compute_unrealized_pnl_for_state(st, last_prices)
            equity_end[u] = av + upnl
            upnl_end[u] = upnl
        return equity_end, upnl_end, last_prices

    rows = []
    winner_pos_rows: list[dict] = []
    gap_coin_rows: list[dict] = []
    prod_haircut_rows: list[dict] = []

    it = intervals.itertuples(index=False, name=None)
    cols = list(intervals.columns)
    # Column indices (avoid pandas name-mangling edge cases).
    i_seg = cols.index("segment")
    i_t0 = cols.index("t_start_ms")
    i_t1 = cols.index("t_end_ms")
    i_D = cols.index("deficit_usd")
    i_max_loss = cols.index("max_loss_usd") if "max_loss_usd" in cols else None
    for w in tqdm(list(it), desc=f"two-pass replay (horizon={eval_horizon_ms}ms)"):
        seg = int(w[i_seg])
        t_start = int(w[i_t0])
        t_end = int(w[i_t1])
        t_eval_end = int(t_end + int(eval_horizon_ms))
        D = float(w[i_D])
        max_loss_usd = float(w[i_max_loss]) if i_max_loss is not None else float("nan")

        seg_winners = winners_by_seg.get(seg, set())

        init = start_snaps.get(t_start)
        if init is None:
            continue
        init_states_all, init_prices = init
        users_of_interest = set(seg_winners)
        init_states = {u: deepcopy(init_states_all[u]) for u in users_of_interest if u in init_states_all}

        # winner start positions (for discrete contract rounding experiments)
        for u in seg_winners:
            st = init_states.get(u)
            if not st:
                continue
            for coin, p in (st.get("positions", {}) or {}).items():
                sz = float((p or {}).get("size", 0.0) or 0.0)
                if abs(sz) <= EPS:
                    continue
                winner_pos_rows.append(
                    {
                        "wave": int(seg),
                        "t_start_ms": int(t_start),
                        "user": str(u),
                        "coin": str(coin),
                        "size_contracts": float(sz),
                        "abs_size_contracts": float(abs(sz)),
                    }
                )

        # per-coin avg gap per contract for this wave
        for coin, gap_usd_x_qty in (gap_usd_x_qty_by_seg_coin.get(seg, {}) or {}).items():
            tot_q = float((qty_by_seg_coin.get(seg, {}) or {}).get(coin, 0.0) or 0.0)
            if tot_q <= EPS:
                continue
            gap_coin_rows.append(
                {
                    "wave": int(seg),
                    "t_start_ms": int(t_start),
                    "t_end_ms": int(t_end),
                    "coin": str(coin),
                    "gap_usd_x_qty": float(gap_usd_x_qty),
                    "qty_contracts": float(tot_q),
                    "gap_per_contract_usd": float(gap_usd_x_qty / tot_q),
                }
            )

        eq_end_adl, upnl_end_adl, _ = replay_wave(
            init_states=init_states,
            init_prices=init_prices,
            t_start=t_start,
            t_end=t_end,
            t_eval_end=t_eval_end,
            users_of_interest=users_of_interest,
            disable_adl=False,
        )
        eq_end_noadl, upnl_end_noadl, _ = replay_wave(
            init_states=init_states,
            init_prices=init_prices,
            t_start=t_start,
            t_end=t_end,
            t_eval_end=t_eval_end,
            users_of_interest=users_of_interest,
            disable_adl=True,
        )

        H_prod = 0.0
        max_prod_haircut = 0.0
        winners_equity_start = 0.0
        winners_pnl_start = 0.0

        for u in seg_winners:
            ea = float(eq_end_adl.get(u, 0.0))
            eb = float(eq_end_noadl.get(u, 0.0))
            d = eb - ea
            if d > 0.0:
                H_prod += d
                max_prod_haircut = max(max_prod_haircut, d)

            # equity/pnl at wave start (from init snapshot)
            st = init_states.get(u)
            if st is not None:
                av0 = float(st.get("account_value", 0.0) or 0.0)
                upnl0 = compute_unrealized_pnl_for_state(st, init_prices)
                eq0 = av0 + upnl0
                winners_equity_start += max(eq0, 0.0)
                winners_pnl_start += max(upnl0, 0.0)
            else:
                eq0 = 0.0
                upnl0 = 0.0

            # Per-winner export: start equity proxy + end equities/pnl + haircut.
            haircut_u = max(d, 0.0)
            prod_haircut_rows.append(
                {
                    "wave": int(seg),
                    "t_start_ms": int(t_start),
                    "t_end_ms": int(t_end),
                    "t_eval_end_ms": int(t_eval_end),
                    "user": str(u),
                    "equity_start_usd": float(eq0),
                    "pnl_start_usd": float(upnl0),
                    "equity_end_adl_usd": float(ea),
                    "equity_end_noadl_usd": float(eb),
                    "pnl_end_adl_usd": float(upnl_end_adl.get(u, 0.0)),
                    "pnl_end_noadl_usd": float(upnl_end_noadl.get(u, 0.0)),
                    "haircut_prod_usd": float(haircut_u),
                }
            )

        B_needed = float(budget_needed_by_seg.get(seg, 0.0))
        overshoot_prod_minus_needed = float(H_prod - B_needed)

        theta_prod = (H_prod / D) if D > EPS else (0.0 if H_prod <= EPS else float("inf"))
        theta_needed = (B_needed / D) if D > EPS else (0.0 if B_needed <= EPS else float("inf"))
        theta_prod_capped = float(np.clip(theta_prod, 0.0, 1.0)) if np.isfinite(theta_prod) else theta_prod
        theta_needed_capped = float(np.clip(theta_needed, 0.0, 1.0)) if np.isfinite(theta_needed) else theta_needed

        # Simple lower bound benchmarks: "raise the same solvency budget B_needed" in wealth space.
        H_star_equity = float(min(B_needed, winners_equity_start)) if B_needed > EPS else 0.0
        H_star_pnl = float(min(B_needed, winners_pnl_start)) if B_needed > EPS else 0.0

        rows.append(
            {
                "wave": int(seg),
                "t_start_ms": int(t_start),
                "t_end_ms": int(t_end),
                "t_eval_end_ms": int(t_eval_end),
                "deficit_usd": float(D),
                "max_loss_usd": float(max_loss_usd),
                "budget_needed_usd": float(B_needed),
                "budget_prod_usd": float(H_prod),
                "overshoot_prod_minus_needed_usd": float(overshoot_prod_minus_needed),
                "max_prod_haircut_usd": float(max_prod_haircut),
                "winners_equity_start_usd": float(winners_equity_start),
                "winners_pnl_start_usd": float(winners_pnl_start),
                "theta_prod": float(theta_prod),
                "theta_prod_capped": float(theta_prod_capped)
                if isinstance(theta_prod_capped, float)
                else theta_prod_capped,
                "theta_needed": float(theta_needed),
                "theta_needed_capped": float(theta_needed_capped)
                if isinstance(theta_needed_capped, float)
                else theta_needed_capped,
                "H_star_equity": float(H_star_equity),
                "H_star_pnl": float(H_star_pnl),
            }
        )

    by_wave = pd.DataFrame(rows).sort_values("t_start_ms").reset_index(drop=True)
    prod_haircuts = pd.DataFrame(prod_haircut_rows)
    winner_start_positions = pd.DataFrame(winner_pos_rows)
    coin_gap_per_contract = pd.DataFrame(gap_coin_rows)

    summary = {
        "hyperreplay_root": str(hr_root),
        "loser_waves_csv": str(Path(loser_waves_csv)),
        "eval_horizon_ms": int(eval_horizon_ms),
        "num_waves": int(len(by_wave)),
        "total_deficit_usd": float(by_wave["deficit_usd"].sum()) if len(by_wave) else 0.0,
        "total_budget_needed_usd": float(by_wave["budget_needed_usd"].sum()) if len(by_wave) else 0.0,
        "total_H_prod_usd": float(by_wave["budget_prod_usd"].sum()) if len(by_wave) else 0.0,
        "total_overshoot_prod_minus_needed_usd": float(by_wave["overshoot_prod_minus_needed_usd"].sum())
        if len(by_wave)
        else 0.0,
        "mean_theta_prod_capped": float(by_wave["theta_prod_capped"].mean()) if len(by_wave) else 0.0,
        "mean_theta_needed_capped": float(by_wave["theta_needed_capped"].mean()) if len(by_wave) else 0.0,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "two_pass_equity_delta_by_wave.csv").write_text(by_wave.to_csv(index=False))
    (out_dir / "two_pass_wave_prod_haircuts.csv").write_text(prod_haircuts.to_csv(index=False))
    (out_dir / "two_pass_wave_winner_start_positions.csv").write_text(winner_start_positions.to_csv(index=False))
    (out_dir / "two_pass_wave_coin_gap_per_contract.csv").write_text(coin_gap_per_contract.to_csv(index=False))
    (out_dir / "two_pass_equity_delta_summary.json").write_text(_json_dumps(summary))

    return TwoPassOutputs(
        by_wave=by_wave,
        prod_haircuts=prod_haircuts,
        winner_start_positions=winner_start_positions,
        coin_gap_per_contract=coin_gap_per_contract,
        summary=summary,
    )


def run_eval_horizon_sweep(
    *,
    hyperreplay_root: Path,
    loser_waves_csv: Path,
    out_root: Path,
    gap_ms: int,
    horizons_ms: Iterable[int],
) -> pd.DataFrame:
    """
    Run two-pass replay at multiple evaluation horizons Δ, keeping waves fixed.

    Output layout (mirrors the original repo for familiarity):
      out_root/
        eval_horizon_sweep/gap_ms=<gap_ms>/horizon_ms=<Δ>/two_pass_*.csv
      out_root/
        eval_horizon_sweep_gap_ms=<gap_ms>.csv
    """
    out_root = Path(out_root)
    sweep_dir = out_root / "eval_horizon_sweep" / f"gap_ms={int(gap_ms)}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for h in sorted({int(x) for x in horizons_ms}):
        h_dir = sweep_dir / f"horizon_ms={h}"
        out = run_two_pass_for_waves(
            hyperreplay_root=hyperreplay_root,
            loser_waves_csv=loser_waves_csv,
            out_dir=h_dir,
            eval_horizon_ms=int(h),
        )
        rows.append(
            {
                "gap_ms": int(gap_ms),
                "eval_horizon_ms": int(h),
                **{
                    k: out.summary[k]
                    for k in [
                        "num_waves",
                        "total_deficit_usd",
                        "total_budget_needed_usd",
                        "total_H_prod_usd",
                        "total_overshoot_prod_minus_needed_usd",
                    ]
                },
            }
        )

    df = pd.DataFrame(rows).sort_values("eval_horizon_ms").reset_index(drop=True)
    out_csv = out_root / f"eval_horizon_sweep_gap_ms={int(gap_ms)}.csv"
    out_csv.write_text(df.to_csv(index=False))
    return df


def summarize_overshoot_robustness_from_sweep(
    *,
    sweep_csv: Path,
    half_lives_ms: Iterable[int] = (500, 1000, 2000, 5000),
    band_max_ms: int = 10000,
) -> dict:
    """
    Compute the "robustness bundle" from a horizon sweep CSV:
      - O(0)
      - O^disc(tau) for selected half-lives
      - ΔO^disc(tau)
      - horizon band [min,max] over Δ∈[0,band_max_ms]
    """
    df = pd.read_csv(sweep_csv).sort_values("eval_horizon_ms")
    horizon = pd.to_numeric(df["eval_horizon_ms"], errors="coerce").fillna(0).to_numpy(float)
    overshoot = pd.to_numeric(df["total_overshoot_prod_minus_needed_usd"], errors="coerce").fillna(0.0).to_numpy(float)

    def overshoot_discounted(half_life_ms: float) -> float:
        lam = np.log(2.0) / float(half_life_ms)
        weights = lam * np.exp(-lam * horizon)
        integral = _trapz(overshoot * weights, horizon)
        norm = _trapz(weights, horizon)
        return float(integral / norm) if norm > EPS else float("nan")

    # O(0)
    overshoot_0 = float(overshoot[horizon == 0][0]) if np.any(horizon == 0) else float("nan")

    # band
    mask = (horizon >= 0.0) & (horizon <= float(band_max_ms))
    band_min = float(np.min(overshoot[mask])) if np.any(mask) else float("nan")
    band_max = float(np.max(overshoot[mask])) if np.any(mask) else float("nan")

    disc = {}
    for hl in half_lives_ms:
        od = overshoot_discounted(float(hl))
        disc[int(hl)] = {
            "O_disc": float(od),
            "DeltaO_disc": float(od - overshoot_0) if np.isfinite(overshoot_0) else float("nan"),
        }

    return {
        "sweep_csv": str(Path(sweep_csv)),
        "O0": float(overshoot_0),
        "band_max_ms": int(band_max_ms),
        "O_min_band": float(band_min),
        "O_max_band": float(band_max),
        "discounted": disc,
    }


def run_wave_gap_sweep(
    *,
    hyperreplay_root: Path,
    canonical_realtime_csv: Path,
    out_root: Path,
    gaps_ms: Iterable[int],
    eval_horizon_ms: int = 0,
) -> pd.DataFrame:
    """
    Sensitivity experiment: vary the clustering gap_ms (changes the wave partition).

    This corresponds to "Experiment A" in the repo's sweep note: changing gap_ms merges/splits waves,
    so the curve is not smooth; we report it as a robustness diagnostic rather than a single headline.
    """
    from oss_adl.bad_debt import compute_loser_waves, write_loser_waves

    out_root = Path(out_root)
    rows = []
    for g in [int(x) for x in gaps_ms]:
        subdir = out_root / "gap_sweep" / f"gap_ms={g}"
        subdir.mkdir(parents=True, exist_ok=True)
        lw = compute_loser_waves(canonical_realtime_csv=canonical_realtime_csv, gap_ms=g, prefer_equity=True)
        waves_csv, _ = write_loser_waves(out_dir=subdir, loser_waves=lw)
        out = run_two_pass_for_waves(
            hyperreplay_root=hyperreplay_root,
            loser_waves_csv=waves_csv,
            out_dir=subdir,
            eval_horizon_ms=int(eval_horizon_ms),
        )
        rows.append(
            {
                "gap_ms": int(g),
                "eval_horizon_ms": int(eval_horizon_ms),
                **{
                    k: out.summary[k]
                    for k in [
                        "num_waves",
                        "total_deficit_usd",
                        "total_budget_needed_usd",
                        "total_H_prod_usd",
                        "total_overshoot_prod_minus_needed_usd",
                    ]
                },
            }
        )

    df = pd.DataFrame(rows).sort_values("gap_ms").reset_index(drop=True)
    out_csv = out_root / "wave_gap_sweep.csv"
    out_csv.write_text(df.to_csv(index=False))
    return df
