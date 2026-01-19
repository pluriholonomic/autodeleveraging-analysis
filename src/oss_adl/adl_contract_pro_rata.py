from __future__ import annotations

"""
Contracts/base-space pro-rata primitives (position-closure numeraire).

This module exists only to reproduce the "policy comparison" figures:
- A venue can only close whole contracts, so a wealth-space pro-rata target may be
  infeasible without rounding.
- Standard exchange implementations allocate closures pro-rata in contract units.

We keep this module intentionally parametric:
- it does not attempt to model venue-specific bankruptcy price rules
- it treats "loss per contract" (USD) as an input (or derived from a target budget)
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class ContractPosition:
    """
    user_id: identifier
    qty: signed position size in contracts (e.g., +1 ETH long, -2 ETH short)
    """

    user_id: str
    qty: float


def _max_close(qty: np.ndarray, eligible_side: str) -> np.ndarray:
    if eligible_side == "long":
        return np.maximum(qty, 0.0)
    if eligible_side == "short":
        return np.maximum(-qty, 0.0)
    if eligible_side == "both":
        return np.abs(qty)
    raise ValueError("eligible_side must be one of: 'long', 'short', 'both'")


def contracts_pro_rata_close_integer(
    positions: Iterable[ContractPosition],
    *,
    required_contracts: float,
    loss_per_contract_usd: float,
    eligible_side: str = "long",
    weights: np.ndarray | None = None,
    rounding: str = "ceil",
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Integer (whole-contract) contracts-space pro-rata.

    1) Compute ideal integer closes k_i ≈ Q * w_i / Σ w
    2) Take floor, then distribute remaining contracts by largest fractional remainder,
       respecting per-position max closable contracts.
    """
    pos_list: list[ContractPosition] = list(positions)
    n = len(pos_list)
    if n == 0:
        return np.zeros(0), np.zeros(0), 0, 0.0

    q = np.array([p.qty for p in pos_list], dtype=float)
    max_close = _max_close(q, eligible_side)
    max_int = np.floor(np.maximum(max_close, 0.0) + 1e-12).astype(int)

    if rounding not in {"ceil", "floor", "round"}:
        raise ValueError("rounding must be one of: 'ceil', 'floor', 'round'")
    if rounding == "ceil":
        Q_int = int(np.ceil(float(required_contracts)))
    elif rounding == "floor":
        Q_int = int(np.floor(float(required_contracts)))
    else:
        Q_int = int(np.round(float(required_contracts)))
    Q_int = max(Q_int, 0)

    loss_per_contract = float(max(loss_per_contract_usd, 0.0))
    if Q_int == 0 or loss_per_contract <= EPS:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), int(Q_int), float(loss_per_contract)

    eligible = max_int > 0
    if not np.any(eligible):
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), int(Q_int), float(loss_per_contract)

    if weights is None:
        w = max_close.copy()
    else:
        w = np.asarray(weights, dtype=float).copy()
        if w.shape != q.shape:
            raise ValueError("weights must have the same shape as positions")
        w = np.maximum(w, 0.0)

    w = np.where(eligible, w, 0.0)
    total_w = float(np.sum(w))
    if total_w <= EPS:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), int(Q_int), float(loss_per_contract)

    ideal = float(Q_int) * (w / total_w)
    base = np.floor(ideal).astype(int)
    base = np.minimum(base, max_int)
    closed_int = base.copy()

    remaining = int(Q_int - int(np.sum(closed_int)))
    if remaining > 0:
        frac = ideal - np.floor(ideal)
        order = np.argsort(frac)[::-1]
        for idx in order:
            if remaining <= 0:
                break
            if not eligible[idx]:
                continue
            if closed_int[idx] >= max_int[idx]:
                continue
            closed_int[idx] += 1
            remaining -= 1

    closed = closed_int.astype(float)
    losses = closed * loss_per_contract
    return closed, losses, int(Q_int), float(loss_per_contract)


def rounded_wealth_pro_rata_to_contracts(
    *,
    target_haircuts_usd: np.ndarray,
    loss_per_contract_usd: np.ndarray,
    max_contracts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Round wealth-space pro-rata targets into whole-contract closures.

    We pick integer k_i in [0, max_contracts[i]] to approximate target_haircuts_usd and
    match aggregate budget as closely as possible:
      realized_i = k_i * loss_per_contract_i

    Greedy largest-residual rounding:
    - start with k_i = floor(target / dpc)
    - add 1 contract to the position that most efficiently reduces residual
    """
    target = np.maximum(np.asarray(target_haircuts_usd, dtype=float), 0.0)
    dpc = np.maximum(np.asarray(loss_per_contract_usd, dtype=float), 0.0)
    max_k = np.maximum(np.asarray(max_contracts, dtype=float), 0.0)
    if target.shape != dpc.shape or target.shape != max_k.shape:
        raise ValueError("target_haircuts_usd, loss_per_contract_usd, max_contracts must have same shape")

    max_int = np.floor(max_k + 1e-12).astype(int)
    feasible = dpc > EPS

    k0 = np.floor(np.where(feasible, target / dpc, 0.0)).astype(int)
    k = np.minimum(k0, max_int)

    realized = k.astype(float) * dpc
    total_target = float(np.sum(target))
    total_realized = float(np.sum(realized))
    if total_target - total_realized <= 1e-6:
        return k.astype(float), realized.astype(float), total_target, total_realized

    residual = target - realized
    while True:
        cand = np.where((residual > 1e-9) & (k < max_int) & feasible)[0]
        if cand.size == 0:
            break
        score = residual[cand] / np.maximum(dpc[cand], EPS)
        j = int(cand[int(np.argmax(score))])
        k[j] += 1
        realized[j] += dpc[j]
        total_realized += float(dpc[j])
        residual[j] = target[j] - realized[j]
        if total_target - total_realized <= 1e-6:
            break

    return k.astype(float), realized.astype(float), total_target, float(total_realized)


def losses_to_equity_haircuts(
    *,
    losses_usd: np.ndarray,
    equity_usd: np.ndarray,
    cap_usd: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Bridge: convert contract-space losses into wealth-space haircuts (USD).

    Enforce: 0 <= haircut_i <= cap_i, where default cap is equity_i^+.
    Returns:
      haircuts_usd
      uncovered_usd (loss that cannot be covered given caps)
    """
    losses = np.maximum(np.asarray(losses_usd, dtype=float), 0.0)
    equity = np.asarray(equity_usd, dtype=float)
    if losses.shape != equity.shape:
        raise ValueError("losses_usd and equity_usd must have the same shape")

    if cap_usd is None:
        cap = np.maximum(equity, 0.0)
    else:
        cap = np.maximum(np.asarray(cap_usd, dtype=float), 0.0)
        if cap.shape != equity.shape:
            raise ValueError("cap_usd must have the same shape as equity_usd")

    haircuts = np.minimum(losses, cap)
    uncovered = float(np.sum(losses - haircuts))
    return haircuts.astype(float), uncovered


__all__ = [
    "ContractPosition",
    "contracts_pro_rata_close_integer",
    "rounded_wealth_pro_rata_to_contracts",
    "losses_to_equity_haircuts",
]
