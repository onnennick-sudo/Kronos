"""Performance metrics. Pure pandas/numpy, no I/O."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple period-over-period returns; infinities and leading NaN dropped."""
    r = equity.astype(float).pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe: mean / sample std (ddof=1) * sqrt(252). Zero risk-free rate.

    Returns 0.0 with fewer than 2 points or zero/non-finite std.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return float(r.mean()) / sd * math.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.25 == -25%)."""
    e = pd.Series(equity, dtype=float).dropna()
    if e.empty:
        return 0.0
    peak = e.cummax()
    valid = peak > 0
    if not valid.any():
        return 0.0
    dd = (peak[valid] - e[valid]) / peak[valid]
    return float(max(dd.max(), 0.0))


def hit_rate(preds_vs_actual: list[tuple[float, float]]) -> tuple[float, float, int]:
    """Directional hit rate over (pred_r, actual_r) pairs.

    hit       = share of pairs where (pred > 0) == (actual > 0)
    base_rate = share of actual > 0 (hit rate of always predicting "up")
    n         = number of pairs
    Returns (0.0, 0.0, 0) for empty input.
    """
    n = len(preds_vs_actual)
    if n == 0:
        return 0.0, 0.0, 0
    preds = np.array([p for p, _ in preds_vs_actual], dtype=float)
    actuals = np.array([a for _, a in preds_vs_actual], dtype=float)
    hit = float(np.mean((preds > 0) == (actuals > 0)))
    base = float(np.mean(actuals > 0))
    return hit, base, n
