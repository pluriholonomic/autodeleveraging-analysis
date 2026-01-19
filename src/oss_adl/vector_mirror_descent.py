from __future__ import annotations

"""
Vector mirror descent utilities (projected gradient) for haircut vectors.

We use this in the "benchmark policy" figures to construct a simple allocation that:
  - chooses haircut fractions h in [0,1]^n
  - matches an aggregate budget exactly: weights^T h = B

In this repo, `weights` are per-winner USD capacities (capped by equity), and the resulting
per-winner USD haircuts are: haircut_i = h_i * weight_i.

This is intentionally minimal and self-contained (numpy only).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _ensure_1d(array: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def project_weighted_box_simplex(
    y: np.ndarray,
    weights: np.ndarray,
    budget: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Euclidean projection of y onto: {x in [0,1]^n : weights^T x = budget}.

    Assumes:
      - weights > 0
      - 0 <= budget <= sum(weights)

    We solve for the Lagrange multiplier via bisection on:
      phi(lam) = weights^T clip(y - lam * weights, 0, 1) - budget.
    """
    w = _ensure_1d(weights, "weights")
    if np.any(w <= 0.0):
        raise ValueError("weights must be strictly positive")

    y = np.asarray(y, dtype=float)
    if y.shape != w.shape:
        raise ValueError("y and weights must have the same shape")

    budget = float(np.clip(float(budget), 0.0, float(w.sum())))

    def phi(lam: float) -> float:
        x = np.clip(y - lam * w, 0.0, 1.0)
        return float(np.dot(w, x) - budget)

    # Bracket the root. phi is monotone decreasing in lam.
    lam_low = float(np.min((y - 1.0) / w) - 1.0)
    lam_high = float(np.max(y / w) + 1.0)
    phi_low = phi(lam_low)
    phi_high = phi(lam_high)

    widen = 0
    while phi_low < 0.0 and widen < max_iter:
        lam_low -= max(1.0, abs(lam_low) * 0.5 + 1.0)
        phi_low = phi(lam_low)
        widen += 1
    widen = 0
    while phi_high > 0.0 and widen < max_iter:
        lam_high += max(1.0, abs(lam_high) * 0.5 + 1.0)
        phi_high = phi(lam_high)
        widen += 1

    if phi_low < 0.0 or phi_high > 0.0:
        # Fallback: boundary cases.
        if np.isclose(budget, 0.0, atol=tol):
            return np.zeros_like(y)
        if np.isclose(budget, float(w.sum()), atol=tol):
            return np.ones_like(y)
        raise RuntimeError("Failed to bracket the projection root; check inputs.")

    for _ in range(int(max_iter)):
        lam_mid = 0.5 * (lam_low + lam_high)
        phi_mid = phi(lam_mid)
        if abs(phi_mid) <= tol:
            lam_low = lam_high = lam_mid
            break
        if phi_mid > 0.0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = 0.5 * (lam_low + lam_high)
    return np.clip(y - lam_star * w, 0.0, 1.0)


@dataclass
class VectorMirrorDescent:
    """
    Euclidean mirror descent (projected gradient) on:
      {h in [0,1]^n : weights^T h = budget}.

    The step size uses eta_t = eta0 / sqrt(t).
    """

    weights: Sequence[float]
    eta0: float = 0.1
    init: Sequence[float] | None = None

    def __post_init__(self) -> None:
        self.weights = _ensure_1d(self.weights, "weights")
        if np.any(self.weights <= 0.0):
            raise ValueError("weights must be strictly positive")
        self.n = int(self.weights.size)
        if self.init is None:
            self.h = np.zeros(self.n, dtype=float)
        else:
            init = _ensure_1d(self.init, "init")
            if init.size != self.n:
                raise ValueError("init must match length of weights")
            self.h = np.clip(init, 0.0, 1.0)
        self.round = 0

    def reset(self, init: Sequence[float] | None = None) -> None:
        if init is None:
            self.h = np.zeros(self.n, dtype=float)
        else:
            init = _ensure_1d(init, "init")
            if init.size != self.n:
                raise ValueError("init must match length of weights")
            self.h = np.clip(init, 0.0, 1.0)
        self.round = 0

    def step(self, grad: Sequence[float], *, budget: float | None = None) -> np.ndarray:
        g = _ensure_1d(grad, "grad")
        if g.size != self.n:
            raise ValueError("gradient length must match weights")

        self.round += 1
        eta = float(self.eta0) / np.sqrt(float(self.round))

        y = self.h - eta * g
        if budget is None:
            self.h = np.clip(y, 0.0, 1.0)
        else:
            self.h = project_weighted_box_simplex(y, self.weights, float(budget))
        return self.h.copy()
