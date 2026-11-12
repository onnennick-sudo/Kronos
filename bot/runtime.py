"""Helpers shared by the runners (run_daily, run_trading, run_watchdog).

Mode selection, bookkeeping over raw Alpaca order dicts, alert de-duplication and the
failure path. Nothing in this module places, cancels or replaces orders.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, Iterable

from bot import alerts
from bot.state import StateStore

log = logging.getLogger(__name__)

PAPER = "paper"
DRY_RUN = "dry_run"
ALERTS_FILE = "alerts_sent.jsonl"
ORDERS_FILE = "orders.jsonl"
PREDICTIONS_FILE = "predictions.jsonl"
STOP_ORDER_TYPES = frozenset({"stop", "stop_limit"})  # trailing_stop is deliberately not a stop leg
TERMINAL_ORDER_STATUSES = frozenset({"filled", "canceled", "expired", "rejected"})
CALENDAR_BACK_DAYS = 40
CALENDAR_FWD_DAYS = 40


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def bot_mode(cli_dry_run: bool) -> str:
    """'dry_run' when the CLI flag is set; otherwise 'paper' only when BOT_MODE is exactly 'paper'."""
    if cli_dry_run:
        return DRY_RUN
    return PAPER if os.environ.get("BOT_MODE") == PAPER else DRY_RUN


def flatten_orders(open_orders: Iterable[dict]) -> list[dict]:
    """Top-level orders plus every nested bracket leg, deduplicated by id (first occurrence wins)."""
    seen: set[str] = set()
    flat: list[dict] = []
    for order in open_orders:
        for item in [order, *(order.get("legs") or [])]:
            oid = item.get("id")
            if oid is not None:
                if oid in seen:
                    continue
                seen.add(oid)
            flat.append(item)
    return flat


def stop_orders_by_symbol(flat: Iterable[dict]) -> dict[str, dict]:
    """Live protective stop per symbol: {"order_id", "stop_price"}. The first stop seen wins."""
    out: dict[str, dict] = {}
    for order in flat:
        if order.get("side") != "sell" or order.get("type") not in STOP_ORDER_TYPES:
            continue
        if order.get("status") in TERMINAL_ORDER_STATUSES or order.get("stop_price") is None:
            continue
        out.setdefault(order["symbol"], {"order_id": order["id"], "stop_price": float(order["stop_price"])})
    return out


def orders_for_symbol(flat: Iterable[dict], symbol: str) -> list[dict]:
    return [o for o in flat if o.get("symbol") == symbol]


def calendar_window(client: Any, today: date,
                    back_days: int = CALENDAR_BACK_DAYS, fwd_days: int = CALENDAR_FWD_DAYS) -> list[dict]:
    return client.get_calendar(today - timedelta(days=back_days), today + timedelta(days=fwd_days))


def alert_once(store: StateStore, key: str, title: str, msg: str, priority: str) -> bool:
    """Send a push unless ``key`` was already sent. Returns True only when a new push succeeded.

    A failed push is not recorded, so the next run may retry it.
    """
    if any(r.get("key") == key for r in store.read(ALERTS_FILE)):
        return False
    sent = alerts.send_push(title, msg, priority=priority)
    if sent:
        store.append(ALERTS_FILE, {"key": key, "title": title, "priority": priority})
    return bool(sent)


def fail(store: StateStore, kind: str, key: str, exc: BaseException) -> None:
    """Report a failed run: urgent push, error record in runs.jsonl, then sync state. Never raises."""
    message = f"{type(exc).__name__}: {exc}"[:500]
    log.error("%s run %s failed: %s", kind, key, message)
    alerts.send_push(f"Kronos {kind} FAILED", message, priority="urgent")
    try:
        store.record_run(kind, key, "error", error=message)
        store.commit_and_push(f"{kind} {key}: error")
    except Exception:  # noqa: BLE001 - the failure path must not mask the original error
        log.exception("could not record failure for %s %s", kind, key)
