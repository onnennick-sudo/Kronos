"""Thin Alpaca REST client (paper trading + market data). Offline-testable via an injected session."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from datetime import date, datetime, time as dtime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from bot import config

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
REQUEST_TIMEOUT_S = 20.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
BAR_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume"]


class AlpacaError(Exception):
    """Raised on any non-retryable failure or after retries are exhausted."""


class PaperGuardError(AlpacaError):
    """Raised when a non-paper base_url is configured."""


def _cents(price: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _parse_ts(value: str) -> datetime:
    """Parse an RFC3339 timestamp into an aware datetime (normalises fractional seconds)."""
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    m = re.match(r"^(.*?\.\d+)(.*)$", s)
    if m:
        frac = m.group(1).split(".", 1)[1][:6].ljust(6, "0")
        s = f"{m.group(1).split('.', 1)[0]}.{frac}{m.group(2)}"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)  # naive inputs follow the project's ET convention
    return dt.isoformat()


class AlpacaClient:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str = config.PAPER_BASE_URL,
        data_url: str = config.DATA_BASE_URL,
        session: Any = None,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        if base_url != config.PAPER_BASE_URL:
            raise PaperGuardError(f"refusing non-paper base_url: {base_url!r}")
        self._key_id = key_id
        self._secret_key = secret_key
        self._base_url = base_url.rstrip("/")
        self._data_url = data_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._max_retries = max_retries
        self._backoff = backoff

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AlpacaClient":
        key_id = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key_id or not secret:
            raise AlpacaError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY must be set")
        return cls(key_id, secret, **kwargs)

    # ----- HTTP core -----

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        tolerate: tuple[int, ...] = (),
    ) -> requests.Response | Any:
        attempt = 0
        while True:
            try:
                resp = self._session.request(
                    method, url, headers=self._headers(), params=params,
                    json=json_body, timeout=REQUEST_TIMEOUT_S,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                problem = f"{method} {url}: {type(exc).__name__}: {exc}"
            else:
                if resp.status_code in tolerate or 200 <= resp.status_code < 300:
                    return resp
                if resp.status_code not in RETRY_STATUSES:
                    raise AlpacaError(f"{method} {url}: HTTP {resp.status_code}: {str(resp.text)[:300]}")
                problem = f"{method} {url}: HTTP {resp.status_code}"
            if attempt >= self._max_retries:
                raise AlpacaError(f"gave up after {attempt + 1} attempts: {problem}")
            delay = self._backoff * (2 ** attempt)
            log.warning("retrying in %.2fs (%s)", delay, problem)
            time.sleep(delay)
            attempt += 1

    def _trading(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, f"{self._base_url}{path}", **kwargs)

    # ----- account / market state -----

    def get_account(self) -> dict:
        data = dict(self._trading("GET", "/v2/account").json())
        for key in ("equity", "last_equity", "cash", "buying_power"):
            if key in data:
                data[key] = float(data[key])
        return data

    def get_clock(self) -> dict:
        data = self._trading("GET", "/v2/clock").json()
        return {
            "is_open": bool(data["is_open"]),
            "timestamp": _parse_ts(data["timestamp"]),
            "next_open": _parse_ts(data["next_open"]),
            "next_close": _parse_ts(data["next_close"]),
        }

    def get_calendar(self, start: date, end: date) -> list[dict]:
        rows = self._trading(
            "GET", "/v2/calendar",
            params={"start": start.isoformat(), "end": end.isoformat()},
        ).json()
        return [
            {
                "date": date.fromisoformat(r["date"]),
                "open": dtime.fromisoformat(r["open"]),
                "close": dtime.fromisoformat(r["close"]),
            }
            for r in rows
        ]

    # ----- positions / orders -----

    def get_positions(self) -> list[dict]:
        out = []
        for p in self._trading("GET", "/v2/positions").json():
            out.append({
                "symbol": p["symbol"],
                "qty": int(math.floor(float(p["qty"]))),
                "avg_entry_price": float(p["avg_entry_price"]),
                "current_price": float(p["current_price"]),
                "market_value": float(p["market_value"]),
            })
        return out

    def get_open_orders(self) -> list[dict]:
        return self._trading(
            "GET", "/v2/orders", params={"status": "open", "nested": "true"},
        ).json()

    def submit_bracket_buy(
        self, symbol: str, qty: int, stop_price: float,
        take_profit_price: float, client_order_id: str,
    ) -> dict:
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "take_profit": {"limit_price": _cents(take_profit_price)},
            "stop_loss": {"stop_price": _cents(stop_price)},
            "client_order_id": client_order_id,
        }
        return self._trading("POST", "/v2/orders", json_body=body).json()

    def submit_market_sell(self, symbol: str, qty: int, client_order_id: str) -> dict:
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        return self._trading("POST", "/v2/orders", json_body=body).json()

    def submit_stop_sell(self, symbol: str, qty: int, stop_price: float, client_order_id: str) -> dict:
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "sell",
            "type": "stop",
            "time_in_force": "gtc",
            "stop_price": _cents(stop_price),
            "client_order_id": client_order_id,
        }
        return self._trading("POST", "/v2/orders", json_body=body).json()

    def cancel_order(self, order_id: str) -> None:
        # 404 (not found) and 422 (not cancelable, e.g. already filled) mean the order is gone.
        self._trading("DELETE", f"/v2/orders/{order_id}", tolerate=(404, 422))

    def replace_order(self, order_id: str, stop_price: float) -> dict:
        body = {"stop_price": _cents(stop_price)}
        return self._trading("PATCH", f"/v2/orders/{order_id}", json_body=body).json()

    # ----- market data -----

    def get_bars(
        self, symbols: list[str], timeframe: str, start: datetime, end: datetime,
        feed: str = config.DATA_FEED,
    ) -> dict[str, pd.DataFrame]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": _rfc3339(start),
            "end": _rfc3339(end),
            "adjustment": "all",
            "feed": feed,
            "limit": 10000,
        }
        return self._paged_bars(f"{self._data_url}/v2/stocks/bars", params)

    def get_crypto_bars(
        self, symbols: list[str], timeframe: str, start: datetime, end: datetime,
    ) -> dict[str, pd.DataFrame]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": _rfc3339(start),
            "end": _rfc3339(end),
            "limit": 10000,
        }
        return self._paged_bars(f"{self._data_url}/v1beta3/crypto/us/bars", params)

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Latest trade price per symbol from the free IEX feed. Symbols without a trade are omitted."""
        if not symbols:
            return {}
        params = {"symbols": ",".join(symbols), "feed": "iex"}
        data = self._request("GET", f"{self._data_url}/v2/stocks/trades/latest", params=params).json()
        trades = data.get("trades") or {}
        return {sym: float(t["p"]) for sym, t in trades.items() if t and t.get("p") is not None}

    def _paged_bars(self, url: str, params: dict) -> dict[str, pd.DataFrame]:
        merged: dict[str, list[dict]] = {}
        page_token = None
        while True:
            q = dict(params)
            if page_token:
                q["page_token"] = page_token
            data = self._request("GET", url, params=q).json()
            for sym, bars in (data.get("bars") or {}).items():
                merged.setdefault(sym, []).extend(bars or [])
            page_token = data.get("next_page_token")
            if not page_token:
                break

        out: dict[str, pd.DataFrame] = {}
        for sym, bars in merged.items():
            if not bars:
                continue
            df = pd.DataFrame({
                "timestamps": pd.to_datetime([b["t"] for b in bars], utc=True),
                "open": [float(b["o"]) for b in bars],
                "high": [float(b["h"]) for b in bars],
                "low": [float(b["l"]) for b in bars],
                "close": [float(b["c"]) for b in bars],
                "volume": [float(b["v"]) for b in bars],
            })
            df = (df.sort_values("timestamps")
                    .drop_duplicates(subset="timestamps", keep="last")
                    .reset_index(drop=True))
            out[sym] = df[BAR_COLUMNS]
        return out
