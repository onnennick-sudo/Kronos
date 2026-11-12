"""Watchdog: alerts when trading slots or the daily run stop producing results.

Usage: python -m bot.run_watchdog
A crash here is not pushed through runtime.fail: GitHub emails the owner when this workflow fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, time as dtime, timezone

from bot import config, runtime, timeutil
from bot.alpaca_client import AlpacaClient
from bot.state import RUNS_FILE, StateStore

log = logging.getLogger(__name__)

TRADING_OPEN_ET = dtime(9, 30)


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Kronos watchdog")
    return parser.parse_args(argv)


def main(argv=None, client=None, store=None, now=None) -> int:
    _parse_args(argv)
    runtime.setup_logging()
    client = client if client is not None else AlpacaClient.from_env()
    store = store if store is not None else StateStore()
    now = now if now is not None else timeutil.now_et()
    try:
        sent = _run(client, store, now)
    except Exception:  # noqa: BLE001 - reported through the workflow failure email
        log.exception("watchdog check failed")
        return 1
    if sent:
        store.commit_and_push(f"watchdog {now.date().isoformat()}")
    return 0


def _run(client, store, now) -> bool:
    """Run every check; True when at least one new alert was sent."""
    sent_gap = _check_trading_gap(client, store, now) if client.get_clock()["is_open"] else False
    sent_daily = _check_daily(client, store, now)
    return sent_gap or sent_daily


def _as_et(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timeutil.ET)


def _last_ok_trading(store, today):
    """Latest successful trading record stamped on today's ET date, or None."""
    stamps = []
    for rec in store.read(RUNS_FILE):
        if rec.get("kind") != "trading" or rec.get("status") != "ok" or not rec.get("ts"):
            continue
        ts = _as_et(rec["ts"])
        if ts.date() == today:
            stamps.append(ts)
    return max(stamps, default=None)


def _check_trading_gap(client, store, now) -> bool:
    today = now.date()
    open_ts = datetime.combine(today, TRADING_OPEN_ET, tzinfo=timeutil.ET)
    last_ok = _last_ok_trading(store, today)
    start = max(t for t in (open_ts, last_ok) if t is not None)
    gap_min = (now - start).total_seconds() / 60
    if gap_min <= config.WATCHDOG_TRADING_GAP_MIN:
        return False
    last_text = last_ok.strftime("%H:%M ET") if last_ok else "none today"
    msg = (f"No successful trading slot for {gap_min:.0f} min (limit {config.WATCHDOG_TRADING_GAP_MIN}). "
           f"Last ok: {last_text}.")
    return runtime.alert_once(store, f"wd-trading-{today.isoformat()}-{now.hour}",
                              "Kronos: trading gap", msg, "urgent")


def _daily_recorded(store, today) -> bool:
    return any(
        r.get("kind") == "daily" and r.get("as_of") == today.isoformat()
        and r.get("status") in ("ok", "degraded")
        for r in store.read(RUNS_FILE)
    )


def _check_daily(client, store, now) -> bool:
    today = now.date()
    if now.time() < dtime.fromisoformat(config.DAILY_DEADLINE_ET):
        return False
    if today not in {c["date"] for c in runtime.calendar_window(client, today)}:
        return False
    if _daily_recorded(store, today):
        return False
    msg = f"No successful daily run for {today.isoformat()} by {config.DAILY_DEADLINE_ET} ET."
    return runtime.alert_once(store, f"wd-daily-{today.isoformat()}", "Kronos: daily run missing",
                              msg, "high")


if __name__ == "__main__":
    sys.exit(main())
