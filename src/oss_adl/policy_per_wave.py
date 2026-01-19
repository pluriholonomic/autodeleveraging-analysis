from __future__ import annotations

"""
Per-wave benchmark comparisons: production vs counterfactual haircutting policies.

This module is a minimal OSS port of `scripts/plot_policies_per_wave.py` from the main
repo, adapted to read inputs from OSS's two-pass reconstruction outputs.

Key idea:
- production (H_prod) is a realized wealth-space haircut reconstructed via two-pass replay
- benchmarks are "best-effort" allocations that target the needed budget B_needed
  using the same winner sets/capacities derived from the canonical REALTIME table.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from oss_adl.adl_contract_pro_rata import (
    ContractPosition,
    contracts_pro_rata_close_integer,
    losses_to_equity_haircuts,
    rounded_wealth_pro_rata_to_contracts,
)
from oss_adl.vector_mirror_descent import VectorMirrorDescent

EPS = 1e-12


def allocate_pro_rata(budget: float, capacities: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Capped pro-rata allocation in USD haircut space.

    - capacities[i] is the max USD haircut available on i
    - weights[i] determines shares among active accounts (defaults to capacities)
    """
    caps = np.asarray(capacities, dtype=float).copy()
    if caps.size == 0 or float(budget) <= 0.0:
        return np.zeros_like(caps, dtype=float)
    if weights is None:
        w = caps.copy()
    else:
        w = np.asarray(weights, dtype=float).copy()
        if w.shape != caps.shape:
            raise ValueError("weights must have same shape as capacities")
    w = np.maximum(w, EPS)

    haircuts = np.zeros_like(caps, dtype=float)
    remaining = float(budget)
    active = caps > EPS
    while remaining > EPS and np.any(active):
        idx = np.nonzero(active)[0]
        share = w[idx]
        share = share / float(np.sum(share))
        allocation = remaining * share
        progress = 0.0
        for local, j in enumerate(idx):
            cap_left = float(caps[j] - haircuts[j])
            if cap_left <= EPS:
                continue
            take = min(cap_left, float(allocation[local]))
            if take <= 0.0:
                continue
            haircuts[j] += take
            progress += take
        remaining -= progress
        active = (caps - haircuts) > EPS
        w[~active] = 0.0
    return haircuts


def _max_survivor_endowment(cap_eff: np.ndarray, haircuts_usd: np.ndarray) -> float:
    """
    Max post-haircut endowment: max_i (cap_eff_i - haircut_i)_+.

    In the OSS empirical mapping, cap_eff is the effective PNL-haircut capacity (min(PNL, equity)),
    and haircuts_usd are USD seized amounts.
    """
    caps = np.asarray(cap_eff, dtype=float)
    h = np.asarray(haircuts_usd, dtype=float)
    if caps.size == 0:
        return 0.0
    rem = np.maximum(caps - h, 0.0)
    return float(rem.max()) if rem.size else 0.0


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if (denom is not None and float(denom) > EPS) else float("nan")


def _theta_capped(budget_usd: float, deficit_usd: float) -> float:
    if float(deficit_usd) <= EPS:
        return 0.0
    th = float(budget_usd) / float(deficit_usd)
    return float(np.clip(th, 0.0, 1.0)) if np.isfinite(th) else th


def _fixed_point_ilp_integer_contracts(
    *,
    users: list[str],
    budget_target: float,
    cap_eff: np.ndarray,
    equity: np.ndarray,
    pair_users: list[str],
    pair_maxk_arr: np.ndarray,
    pair_dpc_arr: np.ndarray,
    dar: np.ndarray,
    max_scale: float = 64.0,
    n_bisect: int = 18,
    tol_abs_usd: float = 1e-6,
    use_mip: bool = True,
    mip_time_limit: float = 30.0,
) -> tuple[np.ndarray, float, float]:
    """
    Min-max ILP: minimize the maximum haircut percentage across users.

    Solves the mixed-integer program:
        minimize M
        subject to:
            Σ_j k_j * dpc_j >= budget_target         (meet budget)
            0 <= k_j <= maxk_j, k_j integer          (position limits)
            Σ_{j in user_i} k_j * dpc_j <= M * eq_i  (fairness per user)

    Falls back to greedy rounding if PuLP is unavailable or solver fails.

    Returns:
      haircuts_usd_by_user (len(users))
      total_budget_usd
      M_opt (optimal max haircut fraction)
    """
    B = float(budget_target)
    if B <= EPS or not users:
        return np.zeros(len(users), dtype=float), 0.0, 0.0

    cap_eff = np.asarray(cap_eff, dtype=float)
    equity = np.asarray(equity, dtype=float)
    if cap_eff.shape != equity.shape or cap_eff.shape != (len(users),):
        raise ValueError("cap_eff and equity must be shape (n_users,)")

    if len(pair_users) == 0:
        return np.zeros(len(users), dtype=float), 0.0, 0.0

    # Map user -> index in users array.
    user_to_i = {u: i for i, u in enumerate(users)}

    # Build per-user lists of pair indices
    pairs_by_user: list[list[int]] = [[] for _ in range(len(users))]
    dar_sum = np.zeros(len(users), dtype=float)
    for j, u in enumerate(pair_users):
        i = user_to_i.get(u)
        if i is not None:
            pairs_by_user[i].append(int(j))
            dar_sum[i] += float(dar[j])

    # Try MIP solver (PuLP)
    if use_mip:
        try:
            import pulp

            n_pairs = len(pair_users)
            n_users = len(users)

            # Create the problem
            prob = pulp.LpProblem("MinMaxHaircut", pulp.LpMinimize)

            # Variables: k_j = contracts to close for pair j (integer)
            k_vars = [
                pulp.LpVariable(f"k_{j}", lowBound=0, upBound=int(pair_maxk_arr[j]), cat=pulp.LpInteger)
                for j in range(n_pairs)
            ]

            # Variable: M = max haircut percentage (continuous)
            M = pulp.LpVariable("M", lowBound=0, upBound=1.0, cat=pulp.LpContinuous)

            # Objective: minimize M
            prob += M, "Minimize_Max_Haircut_Pct"

            # Constraint 1: Meet budget
            prob += pulp.lpSum([k_vars[j] * pair_dpc_arr[j] for j in range(n_pairs)]) >= B, "Budget"

            # Constraint 2: Fairness - for each user with equity > 0
            for i in range(n_users):
                if equity[i] > EPS and pairs_by_user[i]:
                    user_haircut = pulp.lpSum([k_vars[j] * pair_dpc_arr[j] for j in pairs_by_user[i]])
                    prob += user_haircut <= M * equity[i], f"Fairness_user_{i}"

            # Solve with time limit
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit)
            status = prob.solve(solver)

            if status == pulp.LpStatusOptimal:
                # Extract solution
                k_opt = np.array([pulp.value(k_vars[j]) or 0.0 for j in range(n_pairs)])
                M_opt = pulp.value(M) or 0.0

                # Compute per-user haircuts
                losses_user = np.zeros(n_users, dtype=float)
                for i in range(n_users):
                    for j in pairs_by_user[i]:
                        losses_user[i] += k_opt[j] * pair_dpc_arr[j]

                haircuts_user, _uncovered = losses_to_equity_haircuts(
                    losses_usd=losses_user, equity_usd=equity, cap_usd=cap_eff
                )
                total = float(np.sum(haircuts_user))
                return haircuts_user, total, float(M_opt)

        except Exception as e:
            logger.debug("MIP solver failed, falling back to greedy: %s", e)

    # Fallback: greedy rounding
    cap_contract_user = np.minimum(np.maximum(dar_sum, 0.0), np.maximum(cap_eff, 0.0))
    cap_total = float(np.sum(cap_contract_user))
    if cap_total <= EPS:
        return np.zeros(len(users), dtype=float), 0.0, 0.0

    def _realize_for_scale(scale: float) -> tuple[np.ndarray, float]:
        target_user = allocate_pro_rata(B * float(scale), cap_contract_user, weights=cap_contract_user)
        losses_user = np.zeros(len(users), dtype=float)
        for i, idxs in enumerate(pairs_by_user):
            if not idxs:
                continue
            t_u = float(target_user[i])
            if t_u <= EPS:
                continue
            denom = float(dar_sum[i])
            if denom <= EPS:
                continue
            idxs_arr = np.asarray(idxs, dtype=int)
            dar_u = dar[idxs_arr]
            dpc_u = pair_dpc_arr[idxs_arr]
            maxk_u = pair_maxk_arr[idxs_arr]
            pair_targets_u = t_u * (dar_u / denom)
            _k, realized_pair_u, _T, _R = rounded_wealth_pro_rata_to_contracts(
                target_haircuts_usd=pair_targets_u,
                loss_per_contract_usd=dpc_u,
                max_contracts=maxk_u,
            )
            losses_user[i] = float(np.sum(realized_pair_u))
        haircuts_user, _uncovered = losses_to_equity_haircuts(
            losses_usd=losses_user, equity_usd=equity, cap_usd=cap_eff
        )
        return haircuts_user, float(np.sum(haircuts_user))

    lo, hi = 0.0, 1.0
    best_h, best_total, best_scale = np.zeros(len(users), dtype=float), 0.0, 0.0
    h_hi, tot_hi = _realize_for_scale(hi)
    best_h, best_total, best_scale = h_hi, tot_hi, hi

    if tot_hi >= B - tol_abs_usd:
        return best_h, best_total, best_scale

    while hi < max_scale and tot_hi < B - tol_abs_usd:
        lo, hi = hi, min(max_scale, hi * 2.0)
        h_hi, tot_hi = _realize_for_scale(hi)
        best_h, best_total, best_scale = h_hi, tot_hi, hi
        if tot_hi >= B - tol_abs_usd:
            break

    if tot_hi < B - tol_abs_usd:
        return best_h, best_total, best_scale

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        h_mid, tot_mid = _realize_for_scale(mid)
        if tot_mid >= B - tol_abs_usd:
            hi, best_h, best_total, best_scale = mid, h_mid, tot_mid, mid
        else:
            lo = mid

    return best_h, best_total, best_scale


@dataclass(frozen=True)
class PolicyPerWavePaths:
    two_pass_by_wave_csv: Path
    winner_start_positions_csv: Path
    coin_gap_csv: Path
    prod_haircuts_csv: Path


def _resolve_inputs(*, out_root: Path, gap_ms: int, horizon_ms: int) -> PolicyPerWavePaths:
    base = Path(out_root) / "eval_horizon_sweep" / f"gap_ms={int(gap_ms)}" / f"horizon_ms={int(horizon_ms)}"
    return PolicyPerWavePaths(
        two_pass_by_wave_csv=base / "two_pass_equity_delta_by_wave.csv",
        winner_start_positions_csv=base / "two_pass_wave_winner_start_positions.csv",
        coin_gap_csv=base / "two_pass_wave_coin_gap_per_contract.csv",
        prod_haircuts_csv=base / "two_pass_wave_prod_haircuts.csv",
    )


def generate_policy_per_wave_figures(
    *,
    out_root: Path,
    canonical_realtime_csv: Path,
    gap_ms: int = 5000,
    horizon_ms: int = 0,
    cap_mode: str = "account_total_unrealized_max",
) -> None:
    """
    Generate policy-comparison figures under out_root/figures/.

    Defaults mirror the main repo's script:
    - budget target per wave is oracle-needed (budget_target = B_needed)
    - cap_mode uses per-user max(total_unrealized_pnl) within the wave, capped by max(total_equity)
    """
    out_root = Path(out_root)
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    paths = _resolve_inputs(out_root=out_root, gap_ms=int(gap_ms), horizon_ms=int(horizon_ms))
    if not paths.two_pass_by_wave_csv.exists():
        raise FileNotFoundError(f"Missing {paths.two_pass_by_wave_csv} (run `oss-adl replay` first)")

    waves = pd.read_csv(paths.two_pass_by_wave_csv)
    waves["t_start"] = pd.to_datetime(waves["t_start_ms"], unit="ms", utc=True)
    waves["t_end"] = pd.to_datetime(waves["t_end_ms"], unit="ms", utc=True)
    waves = waves.sort_values("t_start").reset_index(drop=True)

    rt = pd.read_csv(
        canonical_realtime_csv,
        usecols=[
            "time",
            "user",
            "coin",
            "closed_pnl",
            "position_unrealized_pnl",
            "total_unrealized_pnl",
            "total_equity",
        ],
    )
    rt["time"] = pd.to_numeric(rt["time"], errors="coerce").fillna(0).astype(np.int64)
    rt["closed_pnl"] = pd.to_numeric(rt["closed_pnl"], errors="coerce").fillna(0.0)
    rt["position_unrealized_pnl"] = pd.to_numeric(rt["position_unrealized_pnl"], errors="coerce").fillna(0.0)
    rt["total_unrealized_pnl"] = pd.to_numeric(rt["total_unrealized_pnl"], errors="coerce").fillna(0.0)
    rt["total_equity"] = pd.to_numeric(rt["total_equity"], errors="coerce").fillna(0.0)
    rt["user"] = rt["user"].astype(str)
    rt["coin"] = rt["coin"].astype(str)

    winner_pos = (
        pd.read_csv(paths.winner_start_positions_csv) if paths.winner_start_positions_csv.exists() else pd.DataFrame()
    )
    coin_gap = pd.read_csv(paths.coin_gap_csv) if paths.coin_gap_csv.exists() else pd.DataFrame()
    prod_h = pd.read_csv(paths.prod_haircuts_csv) if paths.prod_haircuts_csv.exists() else pd.DataFrame()

    if not winner_pos.empty:
        winner_pos["wave"] = pd.to_numeric(winner_pos["wave"], errors="coerce").fillna(-1).astype(int)
        winner_pos["user"] = winner_pos["user"].astype(str)
        winner_pos["coin"] = winner_pos["coin"].astype(str)
        winner_pos["abs_size_contracts"] = pd.to_numeric(winner_pos["abs_size_contracts"], errors="coerce").fillna(0.0)
    if not coin_gap.empty:
        coin_gap["wave"] = pd.to_numeric(coin_gap["wave"], errors="coerce").fillna(-1).astype(int)
        coin_gap["coin"] = coin_gap["coin"].astype(str)
        coin_gap["gap_usd_x_qty"] = pd.to_numeric(coin_gap["gap_usd_x_qty"], errors="coerce").fillna(0.0)
        coin_gap["gap_per_contract_usd"] = pd.to_numeric(coin_gap["gap_per_contract_usd"], errors="coerce").fillna(0.0)
    if not prod_h.empty:
        prod_h["wave"] = pd.to_numeric(prod_h["wave"], errors="coerce").fillna(-1).astype(int)
        prod_h["user"] = prod_h["user"].astype(str)
        prod_h["equity_end_noadl_usd"] = pd.to_numeric(prod_h.get("equity_end_noadl_usd"), errors="coerce").fillna(0.0)
        prod_h["haircut_prod_usd"] = pd.to_numeric(prod_h.get("haircut_prod_usd"), errors="coerce").fillna(0.0)

    if cap_mode not in {"account_total_unrealized_max", "sum_coin_position_unrealized_max"}:
        raise ValueError("cap_mode must be one of: account_total_unrealized_max, sum_coin_position_unrealized_max")

    records: list[dict] = []
    user_records: list[dict] = []

    for row in waves.itertuples(index=False):
        wave = int(row.wave)
        t0 = int(row.t_start_ms)
        t1 = int(row.t_end_ms)
        budget_needed = float(getattr(row, "budget_needed_usd", 0.0) or 0.0)
        budget_prod = float(getattr(row, "budget_prod_usd", 0.0) or 0.0)
        deficit_usd = float(getattr(row, "deficit_usd", 0.0) or 0.0)
        max_loss_usd = float(getattr(row, "max_loss_usd", np.nan))
        theta_prod_capped = float(getattr(row, "theta_prod_capped", np.nan))
        theta_needed_capped = float(getattr(row, "theta_needed_capped", np.nan))

        # Oracle-needed benchmark budget.
        budget_target = float(budget_needed)

        sub = rt[(rt["time"] >= t0) & (rt["time"] <= t1)]
        if sub.empty:
            users: list[str] = []
            caps = np.zeros(0, dtype=float)
            _scores = np.zeros(0, dtype=float)
            equity = np.zeros(0, dtype=float)
        else:
            if cap_mode == "account_total_unrealized_max":
                by_user = sub.groupby("user").agg(
                    cap=("total_unrealized_pnl", "max"),
                    score=("closed_pnl", "sum"),
                    equity=("total_equity", "max"),
                )
                users = by_user.index.astype(str).tolist()
                caps = np.maximum(by_user["cap"].to_numpy(dtype=float), 0.0)
                _scores = by_user["score"].to_numpy(dtype=float)
                equity = np.maximum(by_user["equity"].to_numpy(dtype=float), 0.0)
            else:
                tmp = sub.copy()
                tmp["pos_upnl_pos"] = tmp["position_unrealized_pnl"].clip(lower=0.0)
                by_user_coin = tmp.groupby(["user", "coin"], sort=False)["pos_upnl_pos"].max().reset_index()
                by_user_cap = by_user_coin.groupby("user", sort=False)["pos_upnl_pos"].sum()
                by_user_score = tmp.groupby("user", sort=False)["closed_pnl"].sum()
                by_user_equity = tmp.groupby("user", sort=False)["total_equity"].max()
                users = by_user_score.index.astype(str).tolist()
                caps = np.array([float(by_user_cap.get(u, 0.0)) for u in users], dtype=float)
                _scores = np.array([float(by_user_score.get(u, 0.0)) for u in users], dtype=float)
                equity = np.array([float(by_user_equity.get(u, 0.0)) for u in users], dtype=float)

        cap_eff = np.minimum(caps, equity)
        total_cap = float(cap_eff.sum())

        # (A) Wealth pro-rata (continuous), capped by cap_eff.
        h_pr = allocate_pro_rata(budget_target, cap_eff, weights=cap_eff)
        budget_pr = float(h_pr.sum())

        # (A.1) PNL pro-rata
        h_pnl_pr = allocate_pro_rata(budget_target, cap_eff, weights=caps)
        budget_pnl_pr = float(h_pnl_pr.sum())

        # (A.2) Equity pro-rata
        h_eq_pr = allocate_pro_rata(budget_target, cap_eff, weights=equity)
        budget_eq_pr = float(h_eq_pr.sum())

        # (B) Vector mirror descent: per-market allocation, then aggregate.
        # ADL happens per-coin, so Vector-MD must be constrained to users with
        # positions in each coin being ADL'd.
        h_vm = np.zeros_like(cap_eff)
        wpos_df = winner_pos[winner_pos["wave"] == wave] if (not winner_pos.empty) else pd.DataFrame()
        gpos_df = coin_gap[coin_gap["wave"] == wave] if (not coin_gap.empty) else pd.DataFrame()
        if users and (not wpos_df.empty) and (not gpos_df.empty) and budget_target > 0:
            gap_by_coin = {str(r.coin): float(r.gap_usd_x_qty) for r in gpos_df.itertuples(index=False)}
            gpc_by_coin = {str(r.coin): float(r.gap_per_contract_usd) for r in gpos_df.itertuples(index=False)}
            total_gap_wave = float(sum(v for v in gap_by_coin.values() if v > 0))
            user_to_i = {u: i for i, u in enumerate(users)}

            if total_gap_wave > EPS:
                for coin, gap_coin in gap_by_coin.items():
                    if gap_coin <= EPS:
                        continue
                    gpc = float(gpc_by_coin.get(coin, 0.0) or 0.0)
                    if gpc <= EPS:
                        continue
                    B_coin_target = float(budget_target) * float(gap_coin / total_gap_wave)
                    if B_coin_target <= EPS:
                        continue

                    # Get users with positions in this coin
                    subpos = wpos_df[wpos_df["coin"] == coin]
                    if subpos.empty:
                        continue

                    # Compute per-user capacity for this coin: dar = position_size * gap_per_contract
                    coin_users = []
                    coin_caps = []
                    for r in subpos.itertuples(index=False):
                        u = str(r.user)
                        if u not in user_to_i:
                            continue
                        qty = float(r.abs_size_contracts)
                        if qty > EPS:
                            dar_u = qty * gpc  # Dollar-at-risk for this user in this coin
                            coin_users.append(u)
                            coin_caps.append(dar_u)

                    if not coin_users:
                        continue

                    coin_caps_arr = np.array(coin_caps, dtype=float)
                    total_coin_cap = float(coin_caps_arr.sum())
                    if total_coin_cap <= EPS:
                        continue

                    # Run Vector-MD for this coin
                    vm = VectorMirrorDescent(weights=coin_caps_arr, eta0=0.15)
                    vm.reset(np.zeros_like(coin_caps_arr))
                    vm.step(np.zeros_like(coin_caps_arr), budget=min(B_coin_target, total_coin_cap))

                    # Aggregate haircuts: h_i = fraction_i * cap_i
                    for j, u in enumerate(coin_users):
                        i = user_to_i[u]
                        h_vm[i] += vm.h[j] * coin_caps_arr[j]

            # Clip to cap_eff (can't haircut more than user's total capacity)
            h_vm = np.minimum(h_vm, cap_eff)
        budget_vm = float(h_vm.sum())

        # (C) Discrete contract-space pro-rata (standard exchange style) using winner start positions + gap per contract.
        losses_by_user = dict.fromkeys(users, 0.0)
        h_cpr = np.zeros_like(cap_eff)
        budget_cpr = 0.0
        wpos = winner_pos[winner_pos["wave"] == wave] if (not winner_pos.empty) else pd.DataFrame()
        gpos = coin_gap[coin_gap["wave"] == wave] if (not coin_gap.empty) else pd.DataFrame()
        if (not wpos.empty) and (not gpos.empty) and users and budget_target > 0:
            gap_by_coin = {str(r.coin): float(r.gap_usd_x_qty) for r in gpos.itertuples(index=False)}
            gpc_by_coin = {str(r.coin): float(r.gap_per_contract_usd) for r in gpos.itertuples(index=False)}
            total_gap_wave = float(sum(v for v in gap_by_coin.values() if v > 0))
            if total_gap_wave > EPS:
                for coin, gap_coin in gap_by_coin.items():
                    if gap_coin <= EPS:
                        continue
                    gpc = float(gpc_by_coin.get(coin, 0.0) or 0.0)
                    if gpc <= EPS:
                        continue
                    B_coin_target = float(budget_target) * float(gap_coin / total_gap_wave)
                    req_contracts = float(B_coin_target / gpc) if B_coin_target > 0 else 0.0
                    if req_contracts <= EPS:
                        continue
                    subpos = wpos[wpos["coin"] == coin]
                    if subpos.empty:
                        continue
                    pos_list: list[ContractPosition] = []
                    for r in subpos.itertuples(index=False):
                        u = str(r.user)
                        if u not in losses_by_user:
                            continue
                        qty = float(r.abs_size_contracts)
                        if qty > EPS:
                            pos_list.append(ContractPosition(user_id=u, qty=qty))
                    if not pos_list:
                        continue
                    _closed, losses, _Q, _dpc = contracts_pro_rata_close_integer(
                        pos_list,
                        required_contracts=req_contracts,
                        loss_per_contract_usd=gpc,
                        eligible_side="both",
                        rounding="ceil",
                    )
                    for i, p in enumerate(pos_list):
                        losses_by_user[p.user_id] = float(losses_by_user.get(p.user_id, 0.0)) + float(losses[i])

                losses_vec = np.array([float(losses_by_user.get(u, 0.0)) for u in users], dtype=float)
                h_cpr, _uncovered = losses_to_equity_haircuts(losses_usd=losses_vec, equity_usd=equity, cap_usd=cap_eff)
                budget_cpr = float(h_cpr.sum())

        # (D) Build (user, coin) pairs for MIP solver.
        pair_users: list[str] = []
        pair_maxk: list[float] = []
        pair_dpc: list[float] = []
        pair_maxk_arr = np.zeros(0, dtype=float)
        pair_dpc_arr = np.zeros(0, dtype=float)
        dar = np.zeros(0, dtype=float)
        if users and (not wpos.empty) and (not gpos.empty) and budget_target > 0:
            gpc_by_coin = {str(r.coin): float(r.gap_per_contract_usd) for r in gpos.itertuples(index=False)}

            for r in wpos.itertuples(index=False):
                coin = str(r.coin)
                gpc = float(gpc_by_coin.get(coin, 0.0) or 0.0)
                if gpc <= EPS:
                    continue
                u = str(r.user)
                if u not in losses_by_user:
                    continue
                maxk = float(r.abs_size_contracts)
                if maxk <= EPS:
                    continue
                pair_users.append(u)
                pair_maxk.append(maxk)
                pair_dpc.append(gpc)

            if pair_users:
                pair_maxk_arr = np.asarray(pair_maxk, dtype=float)
                pair_dpc_arr = np.asarray(pair_dpc, dtype=float)
                dar = pair_maxk_arr * pair_dpc_arr

        # (E) Min-max ILP (integer contracts): minimize max haircut percentage using MIP solver.
        h_fp_ilp = np.zeros_like(cap_eff)
        budget_fp_ilp = 0.0
        fp_scale = float("nan")
        if users and pair_users and budget_target > 0 and budget_target <= float(total_cap + 1e-6):
            h_fp_ilp, budget_fp_ilp, fp_scale = _fixed_point_ilp_integer_contracts(
                users=users,
                budget_target=budget_target,
                cap_eff=cap_eff,
                equity=equity,
                pair_users=pair_users,
                pair_maxk_arr=pair_maxk_arr,
                pair_dpc_arr=pair_dpc_arr,
                dar=dar,
            )

        def max_pct_haircut(h: np.ndarray, eq: np.ndarray) -> float:
            if h.size == 0:
                return 0.0
            pct = np.zeros_like(h)
            m = eq > EPS
            pct[m] = h[m] / eq[m]
            pct = pct[np.isfinite(pct)]
            return float(np.clip(float(pct.max()) if pct.size else 0.0, 0.0, 1.0))

        h_prod_vec = np.zeros_like(cap_eff)
        rec = {
            "wave": wave,
            "t_start": row.t_start,
            "t_end": row.t_end,
            "budget_needed": budget_needed,
            "budget_prod": budget_prod,
            "budget_pr": budget_pr,
            "budget_pnl_pr": budget_pnl_pr,
            "budget_eq_pr": budget_eq_pr,
            "budget_vector": budget_vm,
            "budget_contract_pr": budget_cpr,
            "budget_fixed_point_ilp_integer": float(budget_fp_ilp),
            "overshoot_prod": budget_prod - budget_needed,
            "overshoot_pr": budget_pr - budget_needed,
            "overshoot_pnl_pr": budget_pnl_pr - budget_needed,
            "overshoot_eq_pr": budget_eq_pr - budget_needed,
            "overshoot_vector": budget_vm - budget_needed,
            "overshoot_contract_pr": budget_cpr - budget_needed,
            "overshoot_fixed_point_ilp_integer": float(budget_fp_ilp) - budget_needed,
            "max_pct_prod": 0.0,
            "max_pct_pr": max_pct_haircut(h_pr, equity),
            "max_pct_pnl_pr": max_pct_haircut(h_pnl_pr, equity),
            "max_pct_eq_pr": max_pct_haircut(h_eq_pr, equity),
            "max_pct_vector": max_pct_haircut(h_vm, equity),
            "max_pct_contract_pr": max_pct_haircut(h_cpr, equity),
            "max_pct_fixed_point_ilp_integer": max_pct_haircut(h_fp_ilp, equity),
            # Fairness-aware ratios (paper definitions), computed in endowment space:
            #   PTSR = upsilon_post / (D - H) where H is the policy haircut budget
            #   PMR  = upsilon_post / Delta^pi where Delta^pi = theta * max_loss
            #   RD-NSE = upsilon_post / (upsilon_post + D - H)
            "ptsr_prod": float("nan"),
            "ptsr_pr": _safe_ratio(_max_survivor_endowment(cap_eff, h_pr), deficit_usd - budget_pr),
            "ptsr_pnl_pr": _safe_ratio(_max_survivor_endowment(cap_eff, h_pnl_pr), deficit_usd - budget_pnl_pr),
            "ptsr_eq_pr": _safe_ratio(_max_survivor_endowment(cap_eff, h_eq_pr), deficit_usd - budget_eq_pr),
            "ptsr_vector": _safe_ratio(_max_survivor_endowment(cap_eff, h_vm), deficit_usd - budget_vm),
            "ptsr_contract_pr": _safe_ratio(_max_survivor_endowment(cap_eff, h_cpr), deficit_usd - budget_cpr),
            "ptsr_fixed_point_ilp_integer": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_fp_ilp), deficit_usd - float(budget_fp_ilp)
            ),
            "rd_nse_prod": float("nan"),
            "rd_nse_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_pr),
                _max_survivor_endowment(cap_eff, h_pr) + deficit_usd - budget_pr,
            ),
            "rd_nse_pnl_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_pnl_pr),
                _max_survivor_endowment(cap_eff, h_pnl_pr) + deficit_usd - budget_pnl_pr,
            ),
            "rd_nse_eq_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_eq_pr),
                _max_survivor_endowment(cap_eff, h_eq_pr) + deficit_usd - budget_eq_pr,
            ),
            "rd_nse_vector": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_vm),
                _max_survivor_endowment(cap_eff, h_vm) + deficit_usd - budget_vm,
            ),
            "rd_nse_contract_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_cpr),
                _max_survivor_endowment(cap_eff, h_cpr) + deficit_usd - budget_cpr,
            ),
            "rd_nse_fixed_point_ilp_integer": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_fp_ilp),
                _max_survivor_endowment(cap_eff, h_fp_ilp) + deficit_usd - float(budget_fp_ilp),
            ),
            "pmr_prod": float("nan"),
            "pmr_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_pr),
                _theta_capped(deficit_usd - budget_pr, deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "pmr_pnl_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_pnl_pr),
                _theta_capped(deficit_usd - budget_pnl_pr, deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "pmr_eq_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_eq_pr),
                _theta_capped(deficit_usd - budget_eq_pr, deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "pmr_vector": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_vm),
                _theta_capped(deficit_usd - budget_vm, deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "pmr_contract_pr": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_cpr),
                _theta_capped(deficit_usd - budget_cpr, deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "pmr_fixed_point_ilp_integer": _safe_ratio(
                _max_survivor_endowment(cap_eff, h_fp_ilp),
                _theta_capped(deficit_usd - float(budget_fp_ilp), deficit_usd) * float(max_loss_usd)
                if np.isfinite(max_loss_usd)
                else float("nan"),
            ),
            "total_cap": float(total_cap),
            "budget_target": float(budget_target),
            "fixed_point_ilp_scale": float(fp_scale) if np.isfinite(fp_scale) else None,
            "deficit_usd": float(deficit_usd),
            "max_loss_usd": float(max_loss_usd) if np.isfinite(max_loss_usd) else None,
            "theta_prod_capped": float(theta_prod_capped),
            "theta_needed_capped": float(theta_needed_capped),
        }

        # Production concentration from the two-pass export: max_i haircut_i / equity_i(noADL,end).
        if not prod_h.empty:
            ph = prod_h[prod_h["wave"] == wave]
            if not ph.empty:
                eq0 = ph["equity_end_noadl_usd"].to_numpy(dtype=float)
                hc = ph["haircut_prod_usd"].to_numpy(dtype=float)
                denom = np.where(eq0 > EPS, eq0, np.nan)
                pct = hc / denom
                pct = pct[np.isfinite(pct)]
                rec["max_pct_prod"] = float(np.clip(float(pct.max()) if pct.size else 0.0, 0.0, 1.0))

                # Production PTSR/PMR: use the same cap_eff endowment proxy and map per-user haircuts.
                # NOTE: prod_h is indexed by user and contains USD haircuts (equity deltas under the
                # cash-only baseline), so we can compare directly to cap_eff in USD.
                h_prod = {str(u): float(x) for u, x in zip(ph["user"].astype(str).tolist(), hc, strict=True)}
                h_prod_vec = (
                    np.array([float(h_prod.get(u, 0.0)) for u in users], dtype=float)
                    if users
                    else np.zeros(0, dtype=float)
                )
                ups_prod = _max_survivor_endowment(cap_eff, h_prod_vec)
                remaining_deficit_prod = float(deficit_usd) - float(budget_prod)
                rec["ptsr_prod"] = _safe_ratio(ups_prod, remaining_deficit_prod)
                rec["rd_nse_prod"] = _safe_ratio(ups_prod, ups_prod + remaining_deficit_prod)
                rec["pmr_prod"] = _safe_ratio(
                    ups_prod,
                    _theta_capped(remaining_deficit_prod, float(deficit_usd)) * float(max_loss_usd)
                    if np.isfinite(max_loss_usd)
                    else float("nan"),
                )

        records.append(rec)

        if users:
            for i, u in enumerate(users):
                user_records.append(
                    {
                        "wave": wave,
                        "t_start": row.t_start,
                        "t_end": row.t_end,
                        "user": u,
                        "cap_eff_usd": float(cap_eff[i]) if cap_eff.size else 0.0,
                        "equity_usd": float(equity[i]) if equity.size else 0.0,
                        "haircut_prod_usd": float(h_prod_vec[i]) if h_prod_vec.size else 0.0,
                        "haircut_pr_usd": float(h_pr[i]) if h_pr.size else 0.0,
                        "haircut_vector_usd": float(h_vm[i]) if h_vm.size else 0.0,
                        "haircut_contract_pr_usd": float(h_cpr[i]) if h_cpr.size else 0.0,
                        "haircut_fixed_point_ilp_integer_usd": float(h_fp_ilp[i]) if h_fp_ilp.size else 0.0,
                    }
                )

    df = pd.DataFrame(records).sort_values("t_start").reset_index(drop=True)

    # Write the per-wave table for inspection.
    (out_root / "policy_per_wave_metrics.csv").write_text(df.to_csv(index=False))
    if user_records:
        (out_root / "policy_per_wave_user_haircuts.csv").write_text(pd.DataFrame(user_records).to_csv(index=False))

    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------
    # Figure 1: 4-panel per-wave performance
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    ax = axes[0]
    ax.plot(df["t_start"], df["budget_needed"], label="needed (|mark-exec|*q)", linewidth=2)
    ax.plot(df["t_start"], df["budget_prod"], label="production (H_prod)", linewidth=2)
    ax.plot(df["t_start"], df["budget_pr"], label="wealth pro-rata (continuous)", linestyle="--")
    ax.plot(df["t_start"], df["budget_contract_pr"], label="contract pro-rata (integer contracts)", linestyle="--")
    ax.plot(
        df["t_start"], df["budget_fixed_point_ilp_integer"], label="min-max ILP (integer contracts)", linestyle="--"
    )
    ax.plot(df["t_start"], df["budget_vector"], label="vector-md (budget=needed)", linestyle="--")
    ax.set_ylabel("USD")
    ax.set_title("Budgets per wave")
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
    fig.savefig(figs / "05_policy_per_wave_performance.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 2: cumulative overshoot vs needed
    # ---------------------------------------------------------------------
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
    fig2, ax2 = plt.subplots(1, 1, figsize=(13, 4.5), sharex=True)
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.plot(cum["t_start"], cum["cum_prod"] - cum["cum_needed"], label="production - needed", linewidth=2)
    ax2.plot(
        cum["t_start"],
        cum["cum_pro_rata_cont"] - cum["cum_needed"],
        label="wealth pro-rata (cont) - needed",
        linestyle="--",
    )
    ax2.plot(
        cum["t_start"],
        cum["cum_contract_pr"] - cum["cum_needed"],
        label="contract pro-rata (int) - needed",
        linestyle="--",
    )
    ax2.plot(
        cum["t_start"], cum["cum_fp_ilp_int"] - cum["cum_needed"], label="min-max ILP (int) - needed", linestyle="--"
    )
    ax2.plot(
        cum["t_start"], cum["cum_vector"] - cum["cum_needed"], label="vector-md - needed", linestyle="-.", linewidth=2
    )
    ax2.set_ylabel("cumulative USD")
    ax2.set_title("Cumulative overshoot vs needed")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(figs / "06_policy_per_wave_cumulative_overshoot.png", dpi=200)
    plt.close(fig2)
