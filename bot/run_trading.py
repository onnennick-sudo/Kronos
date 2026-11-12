"""Intraday runner: one decision per clock-hour slot. Places orders only in paper mode.

Usage: python -m bot.run_trading [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, time as dtime, timedelta, timezone

from bot import config, runtime, strategy, timeutil
from bot.alpaca_client import AlpacaClient, AlpacaError
from bot.forecast import KronosForecaster, seed_for
from bot.risk import is_halted
from bot.state import StateStore

log = logging.getLogger(__name__)

PAPER = runtime.PAPER
POSITIONS_FILE = "positions.json"
STOPPED_OUT_FILE = "stopped_out.jsonl"
HOURLY_HISTORY_DAYS = 140
CONFIRM_READS = 3            # re-reads of open orders after cancelling, before giving up
CONFIRM_SLEEP_S = 1.0
EXIT_REASONS = frozenset({"hourly_strong_neg", "rank_rotation", "daily_bearish", "r_med_nonpositive"})


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Kronos intraday trading slot")
    parser.add_argument("--dry-run", action="store_true", help="log order intents only; place nothing")
    return parser.parse_args(argv)


def main(argv=None, client=None, store=None, forecaster=None, now=None) -> int:
    args = _parse_args(argv)
    runtime.setup_logging()
    mode = runtime.bot_mode(args.dry_run)
    client = client if client is not None else AlpacaClient.from_env()
    store = store if store is not None else StateStore()
    forecaster = forecaster if forecaster is not None else KronosForecaster()
    now = now if now is not None else timeutil.now_et()
    slot = timeutil.slot_key(now)
    try:
        return _run(mode, client, store, forecaster, now, slot)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, then exit 1
        log.exception("trading slot %s failed", slot)
        runtime.fail(store, "trading", slot, exc)
        return 1


def _run(mode, client, store, forecaster, now, slot) -> int:
    today = now.date()
    if store.has_ok("trading", slot):
        log.info("slot %s already completed; nothing to do", slot)
        return 0
    account = client.get_account()
    if not client.get_clock()["is_open"]:
        log.info("market closed; no decision for slot %s", slot)
        return 0

    ranking, spy_regime, notes = _ranking_context(store, today)
    positions = client.get_positions()
    flat = runtime.flatten_orders(client.get_open_orders())
    halted = _halt_check(client, store, mode, slot, today, account, flat)
    stop_orders = runtime.stop_orders_by_symbol(flat)
    stopped_out = _stopped_out_today(store, today, positions)

    held = [p["symbol"] for p in positions]
    symbols = sorted(set(strategy.candidates(ranking)) | set(held))
    hourly, labels = _hourly_decisions(client, forecaster, store, symbols, now, slot)
    last_prices = _last_prices(client, symbols, hourly, positions)
    plan = strategy.decide(
        ranking, spy_regime, hourly, positions, stop_orders,
        float(account["equity"]), halted, stopped_out, last_prices,
    )

    closed, opened = _apply_plan(client, store, mode, today, slot, plan, flat, stop_orders)
    _record_hourly_predictions(store, today, slot, hourly, labels)
    held_after = (set(held) - closed) | opened
    store.write_json(POSITIONS_FILE, {"date": today.isoformat(), "symbols": sorted(held_after)})
    store.record_run(
        "trading", slot, "ok", mode=mode,
        n_exits=len(plan.exits), n_entries=len(plan.entries), n_trails=len(plan.trails),
        closed=sorted(closed), opened=sorted(opened),
        skipped=plan.skipped, notes=notes + plan.notes,
    )
    store.commit_and_push(f"trading {slot} ({mode})")
    return 0


# ----- inputs -----------------------------------------------------------------

def _ranking_context(store: StateStore, today) -> tuple[list[dict], str, list[str]]:
    """Ranking rows and the SPY regime to decide with. Missing or degraded data blocks entries."""
    notes: list[str] = []
    payload = store.load_ranking(today)
    no_ranking_today = payload is None
    if no_ranking_today:
        _, payload = store.latest_ranking()
        notes.append("no_ranking_for_today")
        runtime.alert_once(
            store, f"noranking-{today.isoformat()}", "Kronos: no ranking for today",
            "No daily ranking for today; new entries are blocked. Exits still run.", "high",
        )
    ranking = payload["ranking"] if payload else []
    spy_regime = payload["spy"]["regime"] if payload else "UNKNOWN"
    degraded = bool(payload and payload.get("degraded"))
    if degraded:
        notes.append("degraded_ranking")
    if spy_regime == "UNKNOWN" or no_ranking_today or degraded:
        spy_regime = "BEARISH"  # decide() treats BEARISH as "no new entries"
        notes.append("entries_blocked_by_ranking_state")
    return ranking, spy_regime, notes


def _halt_check(client, store, mode, slot, today, account, flat) -> bool:
    """True when entries are blocked today. A new breach records the halt and cancels open buys."""
    prior = store.halt_today(today)
    equity, last_equity = float(account["equity"]), float(account["last_equity"])
    breached = is_halted(equity, last_equity)
    if breached and prior is None:
        reason = f"equity {equity:.2f} is at or below last_equity {last_equity:.2f} less {config.DAILY_HALT_FRAC:.0%}"
        store.set_halt(today, reason)
        _cancel_open_buys(client, store, mode, slot, today, flat)
        runtime.alert_once(store, f"halt-{today.isoformat()}", "Kronos HALTED",
                           f"{reason}. No new entries today.", "urgent")
    return prior is not None or breached


def _cancel_open_buys(client, store, mode, slot, today, flat) -> None:
    for order in flat:
        if order.get("side") != "buy":
            continue
        fields = {"side": "buy", "order_id": order.get("id")}
        if mode != PAPER:
            _log_order(store, today, slot, mode, "cancel_buy", order.get("symbol"), "intent", **fields)
            continue
        try:
            client.cancel_order(order["id"])
            _log_order(store, today, slot, mode, "cancel_buy", order.get("symbol"), "cancelled", **fields)
        except AlpacaError as exc:
            _log_order(store, today, slot, mode, "cancel_buy", order.get("symbol"), "error",
                       error=str(exc), **fields)


def _stopped_out_today(store, today, positions) -> set[str]:
    """Symbols that left the book without a bot exit since the previous trading run today."""
    day = today.isoformat()
    prev = store.read_json(POSITIONS_FILE, None) or {}
    held = {p["symbol"] for p in positions}
    if prev.get("date") == day:
        was_held = set(prev.get("symbols") or [])
        exited = {r.get("symbol") for r in store.read(runtime.ORDERS_FILE)
                  if r.get("date") == day and r.get("side") == "sell" and r.get("reason") in EXIT_REASONS}
        recorded = {r.get("symbol") for r in store.read(STOPPED_OUT_FILE) if r.get("date") == day}
        for sym in sorted(was_held - held - exited - recorded):
            log.info("%s left the book without a bot exit: treating as stopped out", sym)
            store.append(STOPPED_OUT_FILE, {"date": day, "symbol": sym})
    return {r["symbol"] for r in store.read(STOPPED_OUT_FILE) if r.get("date") == day}


def _hourly_decisions(client, forecaster, store, symbols, now, slot) -> tuple[dict, dict]:
    """Hourly stats per symbol (or {"error"}) and the decision-bar label per forecast symbol."""
    if not symbols:
        return {}, {}
    series, future, labels, errors = _hourly_inputs(client, symbols, now)
    if not series:
        return dict(errors), labels
    try:
        stats = forecaster.forecast(series, future, config.HOURLY_PRED_LEN, config.HOURLY_PATHS,
                                    config.HOURLY_LOOKBACK, seed_for(now, "hourly"))
    except Exception as exc:  # noqa: BLE001 - a forecaster outage blocks entries for the slot
        log.exception("hourly forecast failed system-wide")
        runtime.alert_once(store, f"hourly-system-{slot}", "Kronos hourly forecast FAILED",
                           f"{type(exc).__name__}: {exc}. Entries blocked for this slot.", "urgent")
        return {s: {"error": "forecaster unavailable"} for s in symbols}, {}
    hourly = {sym: stats[sym] for sym in series if sym in stats}
    hourly.update(errors)
    return hourly, labels


def _hourly_inputs(client, symbols, now):
    """Regular-session hourly series per symbol. Stale or empty tickers become errors, not data."""
    today = now.date()
    calendar = runtime.calendar_window(client, today, back_days=HOURLY_HISTORY_DAYS + 10)
    closes = {c["date"]: c["close"] for c in calendar}
    sessions = set(closes)
    end = now.astimezone(timezone.utc) - timedelta(minutes=config.DATA_DELAY_MINUTES)
    raw = client.get_bars(symbols, "1Hour", now - timedelta(days=HOURLY_HISTORY_DAYS), end)
    expected = datetime.combine(today, dtime(now.hour - 1))  # the decision bar for this slot

    series, future, labels, errors = {}, {}, {}, {}
    for sym in symbols:
        df = raw.get(sym)
        if df is not None and not df.empty:
            df = timeutil.filter_regular_hourly(df, sessions, closes)
        if df is None or df.empty:
            errors[sym] = {"error": "no bars"}
            continue
        last = df["timestamps"].iloc[-1].to_pydatetime()
        if last != expected:
            errors[sym] = {"error": "stale"}
            continue
        try:
            future[sym] = timeutil.future_hourly_stamps(calendar, last, config.HOURLY_PRED_LEN)
        except ValueError as exc:
            errors[sym] = {"error": str(exc)}
            continue
        series[sym] = df
        labels[sym] = last
    return series, future, labels, errors


def _last_prices(client, symbols, hourly, positions) -> dict[str, float]:
    """Latest trade prices, falling back to the last hourly close, then the position mark."""
    try:
        prices = dict(client.get_latest_prices(symbols)) if symbols else {}
    except AlpacaError as exc:
        log.warning("latest prices unavailable, using fallbacks: %s", exc)
        prices = {}
    marks = {p["symbol"]: p["current_price"] for p in positions}
    for sym in symbols:
        if sym in prices:
            continue
        stats = hourly.get(sym)
        if stats and "error" not in stats:
            prices[sym] = float(stats["last_close"])
        elif sym in marks:
            prices[sym] = float(marks[sym])
    return prices


# ----- execution --------------------------------------------------------------

def _apply_plan(client, store, mode, today, slot, plan, flat, stop_orders) -> tuple[set[str], set[str]]:
    """Execute the plan in paper mode, or log intents otherwise. Returns (closed, opened) symbols."""
    if mode != PAPER:
        _log_intents(store, today, slot, mode, plan)
        return {e.symbol for e in plan.exits}, set()
    closed = {e.symbol for e in plan.exits if _exit_position(client, store, today, slot, e, flat, stop_orders)}
    opened = {en.symbol for en in plan.entries if _open_entry(client, store, today, slot, en)}
    for trail in plan.trails:
        _trail(client, store, today, slot, trail)
    return closed, opened


def _exit_position(client, store, today, slot, exit_, flat, stop_orders) -> bool:
    """Cancel every open order for the symbol, confirm they are gone, then sell at market."""
    fields = {"side": "sell", "qty": exit_.qty, "reason": exit_.reason}
    if not _cancel_and_confirm(client, exit_.symbol, flat):
        _log_order(store, today, slot, PAPER, "exit", exit_.symbol, "cancel_unconfirmed",
                   error="open orders remain after cancel", **fields)
        runtime.alert_once(store, f"exit-blocked-{exit_.symbol}-{slot}", "Kronos exit BLOCKED",
                           f"{exit_.symbol}: could not confirm cancellation of open orders; sell skipped.",
                           "urgent")
        return False
    try:
        client.submit_market_sell(exit_.symbol, exit_.qty, f"kx-{exit_.symbol}-{slot}")
    except AlpacaError as exc:
        _log_order(store, today, slot, PAPER, "exit", exit_.symbol, "error", error=str(exc), **fields)
        _restore_stop(client, store, today, slot, exit_, stop_orders)
        return False
    _log_order(store, today, slot, PAPER, "exit", exit_.symbol, "submitted", **fields)
    return True


def _cancel_and_confirm(client, symbol: str, flat) -> bool:
    try:
        for order in runtime.orders_for_symbol(flat, symbol):
            client.cancel_order(order["id"])
        remaining = _open_orders_for(client, symbol)
        reads = 0
        while remaining and reads < CONFIRM_READS:
            time.sleep(CONFIRM_SLEEP_S)
            reads += 1
            remaining = _open_orders_for(client, symbol)
    except AlpacaError as exc:
        log.error("cancel/confirm failed for %s: %s", symbol, exc)
        return False
    return not remaining


def _open_orders_for(client, symbol: str) -> list[dict]:
    return runtime.orders_for_symbol(runtime.flatten_orders(client.get_open_orders()), symbol)


def _restore_stop(client, store, today, slot, exit_, stop_orders) -> None:
    """After a failed sell, put the previous protective stop back and alert."""
    prev = stop_orders.get(exit_.symbol)
    if prev is None:
        detail = "sell failed and no prior stop is known; position is UNPROTECTED"
    else:
        try:
            client.submit_stop_sell(exit_.symbol, exit_.qty, prev["stop_price"], f"kxs-{exit_.symbol}-{slot}")
            detail = f"sell failed; protective stop re-placed at {prev['stop_price']:.2f}"
            _log_order(store, today, slot, PAPER, "stop_restore", exit_.symbol, "submitted",
                       side="sell", qty=exit_.qty, stop=prev["stop_price"])
        except AlpacaError as exc:
            detail = f"sell failed and stop re-place failed ({exc}); position is UNPROTECTED"
    runtime.alert_once(store, f"exit-failed-{exit_.symbol}-{slot}", "Kronos exit FAILED",
                       f"{exit_.symbol}: {detail}", "urgent")


def _open_entry(client, store, today, slot, entry) -> bool:
    fields = {"side": "buy", "qty": entry.qty, "reason": entry.reason, "stop": entry.stop, "target": entry.target}
    try:
        client.submit_bracket_buy(entry.symbol, entry.qty, entry.stop, entry.target, f"ke-{entry.symbol}-{slot}")
    except AlpacaError as exc:
        _log_order(store, today, slot, PAPER, "entry", entry.symbol, "error", error=str(exc), **fields)
        runtime.alert_once(store, f"entry-failed-{entry.symbol}-{slot}", "Kronos entry FAILED",
                           f"{entry.symbol}: {exc}", "high")
        return False
    _log_order(store, today, slot, PAPER, "entry", entry.symbol, "submitted", **fields)
    return True


def _trail(client, store, today, slot, trail) -> None:
    fields = {"side": "sell", "reason": "trail", "order_id": trail.order_id,
              "old_stop": trail.old_stop, "stop": trail.new_stop}
    try:
        client.replace_order(trail.order_id, trail.new_stop)
    except AlpacaError as exc:
        _log_order(store, today, slot, PAPER, "trail", trail.symbol, "error", error=str(exc), **fields)
        runtime.alert_once(store, f"trail-failed-{trail.symbol}-{slot}", "Kronos trail FAILED",
                           f"{trail.symbol}: could not raise stop to {trail.new_stop:.2f}; kept {trail.old_stop:.2f}.",
                           "high")
        return
    _log_order(store, today, slot, PAPER, "trail", trail.symbol, "submitted", **fields)


def _log_intents(store, today, slot, mode, plan) -> None:
    for e in plan.exits:
        _log_order(store, today, slot, mode, "exit", e.symbol, "intent",
                   side="sell", qty=e.qty, reason=e.reason)
    for en in plan.entries:
        _log_order(store, today, slot, mode, "entry", en.symbol, "intent",
                   side="buy", qty=en.qty, reason=en.reason, stop=en.stop, target=en.target)
    for t in plan.trails:
        _log_order(store, today, slot, mode, "trail", t.symbol, "intent",
                   side="sell", reason="trail", order_id=t.order_id, old_stop=t.old_stop, stop=t.new_stop)


def _log_order(store, today, slot, mode, action, symbol, status, **fields) -> None:
    store.append(runtime.ORDERS_FILE, {
        "date": today.isoformat(), "slot": slot, "mode": mode,
        "action": action, "symbol": symbol, "status": status, **fields,
    })


def _record_hourly_predictions(store, today, slot, hourly, labels) -> None:
    for sym, stats in hourly.items():
        if "error" in stats or sym not in labels:
            continue
        store.append(runtime.PREDICTIONS_FILE, {
            "kind": "hourly", "date": today.isoformat(), "slot": slot,
            "bar_label": labels[sym].isoformat(), "symbol": sym,
            "r_med": float(stats["r_med"]), "agree": float(stats["agree"]),
            "last_close": float(stats["last_close"]), "horizon": config.HOURLY_PRED_LEN,
        })


if __name__ == "__main__":
    sys.exit(main())
