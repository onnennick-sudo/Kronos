"""Pure decision logic for the paper-trading bot. No I/O, no network.

Every threshold is read from ``bot.config`` at call time so tests can monkeypatch them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bot import config, risk


@dataclass(frozen=True)
class Entry:
    symbol: str
    qty: int
    ref_price: float
    stop: float
    target: float
    rank: int
    reason: str


@dataclass(frozen=True)
class Exit:
    symbol: str
    qty: int
    reason: str


@dataclass(frozen=True)
class Trail:
    symbol: str
    order_id: str
    old_stop: float
    new_stop: float


@dataclass
class Plan:
    exits: list[Exit] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    trails: list[Trail] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- Classification -----------------------------------------------------------

def classify_daily(r_med: float, agree: float) -> str:
    """BULLISH: r_med >= DAILY_ELIGIBLE_R and agree >= DAILY_AGREE.
    BEARISH: r_med <= DAILY_BEARISH_R and agree >= DAILY_AGREE (symmetric with BULLISH).
    Otherwise NEUTRAL.
    """
    if r_med >= config.DAILY_ELIGIBLE_R and agree >= config.DAILY_AGREE:
        return "BULLISH"
    if r_med <= config.DAILY_BEARISH_R and agree >= config.DAILY_AGREE:
        return "BEARISH"
    return "NEUTRAL"


def classify_hourly(r_med: float, agree: float) -> str:
    """CONFIRM_LONG: r_med >= HOURLY_CONFIRM_R and agree >= HOURLY_CONFIRM_AGREE.
    STRONG_NEG: r_med <= HOURLY_STRONG_NEG_R and agree >= HOURLY_STRONG_NEG_AGREE.
    Otherwise NONE.
    """
    if r_med >= config.HOURLY_CONFIRM_R and agree >= config.HOURLY_CONFIRM_AGREE:
        return "CONFIRM_LONG"
    if r_med <= config.HOURLY_STRONG_NEG_R and agree >= config.HOURLY_STRONG_NEG_AGREE:
        return "STRONG_NEG"
    return "NONE"


# --- ATR ----------------------------------------------------------------------

def atr(daily_df: Any, period: int | None = None) -> float:
    """Wilder's ATR over ``period`` using ``high``/``low``/``close`` columns.

    Rows must be chronological. True range needs the previous close, so the first
    TR is taken from row 1. The seed ATR is the simple mean of the first ``period``
    TRs; each later value is ((prev * (period - 1)) + TR) / period.
    Raises ValueError when there are fewer than ``period + 1`` rows.
    """
    if period is None:
        period = config.ATR_PERIOD
    if len(daily_df) < period + 1:
        raise ValueError(f"atr needs at least {period + 1} rows, got {len(daily_df)}")
    high = np.asarray(daily_df["high"], dtype=float)
    low = np.asarray(daily_df["low"], dtype=float)
    close = np.asarray(daily_df["close"], dtype=float)
    prev_close = close[:-1]
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])
    value = float(tr[:period].mean())
    for t in tr[period:]:
        value = (value * (period - 1) + float(t)) / period
    return float(value)


# --- Ranking ------------------------------------------------------------------

def build_ranking(stats: dict[str, dict], atrs: dict[str, float]) -> list[dict]:
    """Rank every symbol whose forecast has no error.

    Sorted by r_med descending (ties broken by symbol). rank 1 is best.
    eligible = r_med >= DAILY_ELIGIBLE_R and agree >= DAILY_AGREE.
    """
    rows = []
    for sym, s in stats.items():
        if "error" in s:
            continue
        rows.append((sym, float(s["r_med"]), float(s["agree"])))
    rows.sort(key=lambda t: (-t[1], t[0]))
    out = []
    for rank, (sym, r_med, agree) in enumerate(rows, start=1):
        out.append({
            "symbol": sym,
            "rank": rank,
            "r_med": r_med,
            "agree": agree,
            "eligible": r_med >= config.DAILY_ELIGIBLE_R and agree >= config.DAILY_AGREE,
            "regime": classify_daily(r_med, agree),
            "atr": atrs.get(sym),
        })
    return out


def candidates(ranking: list[dict], n: int | None = None) -> list[str]:
    """The top ``n`` eligible symbols in rank order (eligibility is applied before the cut)."""
    if n is None:
        n = config.N_CANDIDATES
    eligible = sorted((r for r in ranking if r["eligible"]), key=lambda r: r["rank"])
    return [r["symbol"] for r in eligible[:n]]


# --- Trailing stop ------------------------------------------------------------

def trail_stop(entry_price: float, current_stop: float, last_price: float, atr: float) -> float | None:
    """New stop price, or None. Only ever moves the stop up (by more than one cent)."""
    unrealized = last_price - entry_price
    if unrealized >= config.TRAIL_START_ATR * atr:
        candidate = last_price - config.TRAIL_DISTANCE_ATR * atr
    elif unrealized >= config.TRAIL_BREAKEVEN_ATR * atr:
        candidate = entry_price
    else:
        return None
    new_stop = round(candidate, 2)
    if new_stop > current_stop + 0.01:
        return new_stop
    return None


# --- Decision -----------------------------------------------------------------

def _exit_reason(sym: str, row: dict | None, hourly_row: dict | None, skipped: list) -> str | None:
    """Return the exit reason for a held symbol, or None. Records skips as a side effect."""
    reason = None
    if hourly_row is None or "error" in hourly_row:
        skipped.append((sym, "hourly_error"))
    elif classify_hourly(hourly_row["r_med"], hourly_row["agree"]) == "STRONG_NEG":
        reason = "hourly_strong_neg"

    if row is None:
        # Forecast error for this symbol: no daily-based exit.
        skipped.append((sym, "missing_forecast"))
    elif reason is None:
        if row["rank"] > config.ROTATION_RANK_EXIT:
            reason = "rank_rotation"
        elif row["regime"] == "BEARISH":
            # Checked before r_med_nonpositive: BEARISH is the more specific reason
            # (BEARISH implies r_med <= 0, so the reverse order would make it unreachable).
            reason = "daily_bearish"
        elif row["r_med"] <= 0:
            reason = "r_med_nonpositive"
    return reason


def decide(
    ranking: list[dict],
    spy_regime: str,
    hourly: dict[str, dict],
    positions: list[dict],
    stop_orders: dict[str, dict],
    equity: float,
    halted: bool,
    stopped_out_today: set[str],
    last_prices: dict[str, float],
    sectors: dict[str, str] | None = None,
) -> Plan:
    """Build the order plan for one run.

    Order priority is exits, then entries, then trails. Total orders are capped at
    MAX_ORDERS_PER_RUN; anything beyond the cap is skipped with reason "order_cap".
    """
    if sectors is None:
        sectors = config.SECTORS
    cap = config.MAX_ORDERS_PER_RUN
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    rank_by = {row["symbol"]: row for row in ranking}
    held = {p["symbol"]: p for p in positions}

    # 1. Exits
    raw_exits: list[Exit] = []
    for pos in positions:
        sym = pos["symbol"]
        reason = _exit_reason(sym, rank_by.get(sym), hourly.get(sym), skipped)
        if reason is not None:
            raw_exits.append(Exit(sym, int(pos["qty"]), reason))
    exits = raw_exits[:cap]
    for e in raw_exits[cap:]:
        skipped.append((e.symbol, "order_cap"))
    budget = cap - len(exits)
    exit_syms = {e.symbol for e in raw_exits}
    kept_exit_syms = {e.symbol for e in exits}
    # Positions still held after exits. A dropped exit stays held.
    remaining = [p for p in positions if p["symbol"] not in kept_exit_syms]

    # 2. Entries
    entries: list[Entry] = []
    block = None
    if halted:
        block = "halted"
    elif spy_regime == "BEARISH":
        block = "spy_bearish"

    if block is not None:
        notes.append(f"entries blocked: {block}")
        for sym in candidates(ranking):
            skipped.append((sym, block))
    else:
        exposure_now = sum(float(p["market_value"]) for p in remaining)
        for sym in candidates(ranking):
            row = rank_by[sym]
            reason = None
            h = hourly.get(sym)
            if sym in held:
                reason = "already_held"
            elif sym in stopped_out_today:
                reason = "stopped_out_today"
            elif h is None or "error" in h:
                reason = "hourly_error"
            elif classify_hourly(h["r_med"], h["agree"]) != "CONFIRM_LONG":
                reason = "no_hourly_confirm"
            elif len(remaining) + len(entries) >= config.MAX_POSITIONS:
                reason = "max_positions"

            sector = sectors.get(sym)
            if reason is None and sector is not None:
                in_sector = sum(1 for p in remaining if sectors.get(p["symbol"]) == sector)
                in_sector += sum(1 for e in entries if sectors.get(e.symbol) == sector)
                if in_sector >= config.MAX_PER_SECTOR:
                    reason = "sector_cap"

            atr_val = row.get("atr")
            if reason is None and (atr_val is None or atr_val <= 0):
                reason = "no_atr"

            price = last_prices.get(sym)
            if reason is None and (price is None or price <= 0):
                reason = "no_price"

            qty = 0
            if reason is None:
                qty = risk.size_qty(equity, price)
                if qty < 1:
                    reason = "qty_lt_1"

            if reason is None:
                planned = exposure_now + sum(e.qty * e.ref_price for e in entries) + qty * price
                if planned > config.MAX_EXPOSURE_FRAC * equity:
                    reason = "exposure_cap"

            if reason is None and budget <= 0:
                reason = "order_cap"

            if reason is not None:
                skipped.append((sym, reason))
                continue

            entries.append(Entry(
                symbol=sym,
                qty=qty,
                ref_price=price,
                stop=round(price - config.STOP_ATR_MULT * atr_val, 2),
                target=round(price + config.TARGET_ATR_MULT * atr_val, 2),
                rank=row["rank"],
                reason="confirmed_candidate",
            ))
            budget -= 1

    # 3. Trails (held, not exiting, with a live stop leg)
    trails: list[Trail] = []
    for pos in positions:
        sym = pos["symbol"]
        if sym in exit_syms:
            continue
        so = stop_orders.get(sym)
        if not so:
            skipped.append((sym, "no_stop_order"))
            continue
        row = rank_by.get(sym)
        atr_val = row.get("atr") if row is not None else None
        if atr_val is None:
            skipped.append((sym, "no_atr"))
            continue
        last = last_prices.get(sym, pos["current_price"])
        new_stop = trail_stop(float(pos["avg_entry_price"]), float(so["stop_price"]), float(last), float(atr_val))
        if new_stop is None:
            continue
        if budget <= 0:
            skipped.append((sym, "order_cap"))
            continue
        trails.append(Trail(sym, so["order_id"], float(so["stop_price"]), new_stop))
        budget -= 1

    return Plan(exits=exits, entries=entries, trails=trails, skipped=skipped, notes=notes)
