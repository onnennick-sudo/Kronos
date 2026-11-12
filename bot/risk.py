"""Risk helpers: daily halt check and position sizing. Pure logic, no I/O.

Thresholds are read from ``bot.config`` at call time so tests can monkeypatch them.
"""
from __future__ import annotations

import math

from bot import config


def is_halted(equity: float, last_equity: float) -> bool:
    """True when equity has fallen to or below ``last_equity * (1 - DAILY_HALT_FRAC)``."""
    return equity <= last_equity * (1 - config.DAILY_HALT_FRAC)


def size_qty(equity: float, price: float, existing_value: float = 0.0) -> int:
    """Whole shares that fit in the remaining per-position budget.

    budget = MAX_POSITION_FRAC * equity - existing_value; qty = floor(budget / price), floored at 0.
    A non-positive price yields 0 (nothing can be sized).
    """
    if price <= 0:
        return 0
    return max(0, math.floor((config.MAX_POSITION_FRAC * equity - existing_value) / price))
