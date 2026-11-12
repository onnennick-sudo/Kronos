"""Run gate for GitHub Actions. STDLIB ONLY (plus bot.config).

Usage: python -m bot.gate <trading|daily|watchdog|manual>

Prints the reason, writes ``proceed=true|false`` to $GITHUB_OUTPUT when set.
Exits 1 only on a paper-guard violation; otherwise exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from bot import config

ET = ZoneInfo("America/New_York")
PAPER_URL = "https://paper-api.alpaca.markets"  # literal on purpose: guards config drift
HTTP_TIMEOUT_S = 15
TRADING_WINDOW_START = time(10, 5)   # first trading run (10:20) is inside
TRADING_WINDOW_END = time(16, 0)     # exclusive; last trading run is 15:50


def now() -> datetime:
    """Current time in ET. Monkeypatch in tests."""
    return datetime.now(ET)


def http_get_json(url: str, headers: dict[str, str]) -> Any:
    """GET and decode JSON. Monkeypatch in tests."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _auth_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_API_SECRET_KEY", ""),
    }


def get_clock() -> dict[str, Any]:
    return http_get_json(f"{config.PAPER_BASE_URL}/v2/clock", _auth_headers())


def get_calendar_dates(start: date, end: date) -> set[date]:
    url = (f"{config.PAPER_BASE_URL}/v2/calendar"
           f"?start={start.isoformat()}&end={end.isoformat()}")
    rows = http_get_json(url, _auth_headers())
    return {date.fromisoformat(r["date"]) for r in rows}


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def _trading(t: datetime) -> tuple[bool, str]:
    in_window = TRADING_WINDOW_START <= t.time() < TRADING_WINDOW_END
    try:
        is_open = bool(get_clock()["is_open"])
    except Exception as exc:  # noqa: BLE001
        _warn(f"Alpaca clock check failed ({exc}); proceeding, runner will alert")
        if not in_window:
            return False, "clock unavailable and outside trading window"
        return True, "clock unavailable; inside trading window, proceeding"
    if not is_open:
        return False, "market closed"
    if not in_window:
        return False, f"outside trading window ({t.strftime('%H:%M')} ET)"
    return True, "market open, inside trading window"


def _daily(t: datetime) -> tuple[bool, str]:
    ready = _parse_hhmm(config.DAILY_READY_ET)
    today = t.date()
    try:
        is_session = today in get_calendar_dates(today, today)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Alpaca calendar check failed ({exc}); proceeding, runner will alert")
        if t.time() < ready:
            return False, f"calendar unavailable and before {config.DAILY_READY_ET} ET"
        return True, "calendar unavailable; after ready time, proceeding"
    if not is_session:
        return False, f"{today.isoformat()} is not a trading session"
    if t.time() < ready:
        return False, f"before daily ready time {config.DAILY_READY_ET} ET"
    return True, f"session day, after {config.DAILY_READY_ET} ET"


def evaluate(mode: str, when: datetime | None = None) -> tuple[bool, str]:
    """Return (proceed, reason) for a run mode. Paper guard is checked in main()."""
    if os.environ.get("BOT_ENABLED") != "true":
        return False, "BOT_ENABLED is not 'true'"
    t = (when or now()).astimezone(ET)
    if mode in ("manual", "watchdog"):
        return True, f"{mode}: bot enabled"
    if mode == "trading":
        return _trading(t)
    if mode == "daily":
        return _daily(t)
    return False, f"unknown mode {mode!r}"


def _write_github_output(proceed: bool) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"proceed={'true' if proceed else 'false'}\n")
    except OSError as exc:
        _warn(f"could not write GITHUB_OUTPUT: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    mode = args[0] if args else ""

    if config.PAPER_BASE_URL != PAPER_URL:
        print(f"ERROR: PAPER_BASE_URL is {config.PAPER_BASE_URL!r}; "
              f"expected {PAPER_URL!r}. Refusing to run.", file=sys.stderr)
        return 1

    proceed, reason = evaluate(mode)
    print(f"gate[{mode or '?'}]: proceed={'true' if proceed else 'false'} - {reason}")
    _write_github_output(proceed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
