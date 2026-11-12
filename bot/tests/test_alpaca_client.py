"""Offline tests for bot.alpaca_client. HTTP is mocked with a FakeSession."""

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import requests

from bot import alpaca_client
from bot import config
from bot.alpaca_client import AlpacaClient, AlpacaError, PaperGuardError

ET = ZoneInfo("America/New_York")


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class FakeSession:
    """Records every request; returns queued items in order, then `default`."""

    def __init__(self, queue=(), default=None):
        self.queue = list(queue)
        self.default = default if default is not None else FakeResponse(200, {})
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.queue.pop(0) if self.queue else self.default
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(alpaca_client.time, "sleep", lambda s: slept.append(s))
    return slept


def make(session, **kwargs):
    return AlpacaClient("key", "secret", session=session, **kwargs)


# ----- paper guard and auth -----

def test_paper_guard_raises_before_any_request():
    fake = FakeSession()
    with pytest.raises(PaperGuardError):
        AlpacaClient("k", "s", base_url="https://api.alpaca.markets", session=fake)
    assert fake.calls == []
    assert issubclass(PaperGuardError, AlpacaError)


def test_auth_headers_sent():
    fake = FakeSession(queue=[FakeResponse(200, {"equity": "1", "last_equity": "1", "cash": "1", "buying_power": "1"})])
    make(fake).get_account()
    headers = fake.calls[0]["headers"]
    assert headers["APCA-API-KEY-ID"] == "key"
    assert headers["APCA-API-SECRET-KEY"] == "secret"
    assert fake.calls[0]["url"] == config.PAPER_BASE_URL + "/v2/account"


def test_from_env_reads_variables(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "envkey")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "envsecret")
    client = AlpacaClient.from_env(session=FakeSession())
    assert client._key_id == "envkey"
    assert client._secret_key == "envsecret"


# ----- retries -----

def test_retries_on_503_then_succeeds(no_sleep):
    fake = FakeSession(queue=[
        FakeResponse(503),
        FakeResponse(200, {"equity": "100.5", "last_equity": "99", "cash": "50", "buying_power": "200"}),
    ])
    acct = make(fake, backoff=1.0).get_account()
    assert acct["equity"] == 100.5
    assert len(fake.calls) == 2
    assert no_sleep == [1.0]


def test_retries_on_connection_error_then_succeeds(no_sleep):
    fake = FakeSession(queue=[
        requests.ConnectionError("boom"),
        FakeResponse(200, {"equity": "1", "last_equity": "1", "cash": "1", "buying_power": "1"}),
    ])
    make(fake).get_account()
    assert len(fake.calls) == 2


def test_raises_after_max_retries(no_sleep):
    fake = FakeSession(default=FakeResponse(503))
    with pytest.raises(AlpacaError):
        make(fake, max_retries=2, backoff=1.0).get_account()
    assert len(fake.calls) == 3  # first attempt + 2 retries
    assert no_sleep == [1.0, 2.0]  # exponential backoff


def test_non_retryable_error_raises_immediately(no_sleep):
    fake = FakeSession(default=FakeResponse(403, {"message": "forbidden"}))
    with pytest.raises(AlpacaError):
        make(fake).get_account()
    assert len(fake.calls) == 1
    assert no_sleep == []


# ----- parsing -----

def test_account_numbers_become_floats():
    fake = FakeSession(queue=[FakeResponse(200, {
        "equity": "10000.25", "last_equity": "9900", "cash": "5000.5",
        "buying_power": "20000", "status": "ACTIVE",
    })])
    acct = make(fake).get_account()
    assert acct["equity"] == 10000.25
    assert acct["last_equity"] == 9900.0
    assert acct["cash"] == 5000.5
    assert acct["buying_power"] == 20000.0
    assert acct["status"] == "ACTIVE"


def test_positions_parse_qty_and_prices():
    fake = FakeSession(queue=[FakeResponse(200, [
        {"symbol": "AAPL", "qty": "12", "avg_entry_price": "190.5",
         "current_price": "195.25", "market_value": "2343"},
        {"symbol": "MSFT", "qty": "3.7", "avg_entry_price": "400",
         "current_price": "390", "market_value": "1443"},
    ])])
    pos = make(fake).get_positions()
    assert pos[0] == {"symbol": "AAPL", "qty": 12, "avg_entry_price": 190.5,
                      "current_price": 195.25, "market_value": 2343.0}
    assert pos[1]["qty"] == 3 and isinstance(pos[1]["qty"], int)


def test_clock_parses_aware_datetimes():
    fake = FakeSession(queue=[FakeResponse(200, {
        "timestamp": "2024-03-01T10:15:30.12-05:00",
        "is_open": True,
        "next_open": "2024-03-04T09:30:00-05:00",
        "next_close": "2024-03-01T16:00:00Z",
    })])
    clock = make(fake).get_clock()
    assert clock["is_open"] is True
    for key in ("timestamp", "next_open", "next_close"):
        assert clock[key].tzinfo is not None
    assert clock["timestamp"].microsecond == 120000
    assert clock["next_close"] == datetime(2024, 3, 1, 16, 0, tzinfo=timezone.utc)


def test_calendar_parses_dates_and_times():
    fake = FakeSession(queue=[FakeResponse(200, [
        {"date": "2024-03-01", "open": "09:30", "close": "16:00"},
        {"date": "2024-07-03", "open": "09:30", "close": "13:00"},
    ])])
    cal = make(fake).get_calendar(date(2024, 3, 1), date(2024, 7, 3))
    assert cal[0] == {"date": date(2024, 3, 1), "open": time(9, 30), "close": time(16, 0)}
    assert cal[1]["close"] == time(13, 0)
    assert fake.calls[0]["params"] == {"start": "2024-03-01", "end": "2024-07-03"}


def test_open_orders_request_nested_open_status():
    fake = FakeSession(queue=[FakeResponse(200, [{"id": "o1", "legs": []}])])
    orders = make(fake).get_open_orders()
    assert orders == [{"id": "o1", "legs": []}]
    assert fake.calls[0]["params"] == {"status": "open", "nested": "true"}


# ----- bars -----

def _bar(t, o=1.0):
    return {"t": t, "o": o, "h": o + 1, "l": o - 1, "c": o + 0.5, "v": 1000, "n": 5, "vw": 1.0}


def test_bars_pagination_merges_pages_and_shapes_dataframe():
    fake = FakeSession(queue=[
        FakeResponse(200, {
            "bars": {"AAPL": [_bar("2024-01-03T14:30:00Z", 3.0), _bar("2024-01-02T14:30:00Z", 2.0)]},
            "next_page_token": "tok2",
        }),
        FakeResponse(200, {
            "bars": {
                "AAPL": [_bar("2024-01-03T14:30:00Z", 3.5), _bar("2024-01-04T14:30:00Z", 4.0)],
                "MSFT": [_bar("2024-01-02T14:30:00Z", 9.0)],
                "NOPE": [],
            },
            "next_page_token": None,
        }),
    ])
    start = datetime(2024, 1, 1, tzinfo=ET)
    end = datetime(2024, 1, 5, tzinfo=ET)
    out = make(fake).get_bars(["AAPL", "MSFT", "NOPE"], "1Day", start, end)

    assert set(out) == {"AAPL", "MSFT"}  # symbols without bars omitted
    aapl = out["AAPL"]
    assert list(aapl.columns) == ["timestamps", "open", "high", "low", "close", "volume"]
    assert isinstance(aapl["timestamps"].dtype, pd.DatetimeTZDtype)
    assert str(aapl["timestamps"].dt.tz) == "UTC"
    for col in ("open", "high", "low", "close", "volume"):
        assert aapl[col].dtype == "float64"
    assert len(aapl) == 3  # 3 unique timestamps after dedupe
    assert aapl["timestamps"].is_monotonic_increasing
    assert aapl["timestamps"].is_unique
    # Duplicate kept from the later page (keep="last")
    assert aapl.loc[aapl["timestamps"] == pd.Timestamp("2024-01-03T14:30:00Z"), "open"].item() == 3.5
    assert out["MSFT"]["close"].iloc[0] == 9.5

    first, second = fake.calls
    assert first["url"] == config.DATA_BASE_URL + "/v2/stocks/bars"
    assert "page_token" not in first["params"]
    assert second["params"]["page_token"] == "tok2"
    assert first["params"]["adjustment"] == "all"
    assert first["params"]["limit"] == 10000
    assert first["params"]["feed"] == config.DATA_FEED
    assert first["params"]["start"] == start.isoformat()
    assert first["params"]["end"] == end.isoformat()
    assert first["params"]["symbols"] == "AAPL,MSFT,NOPE"


def test_bars_empty_response_returns_empty_dict():
    fake = FakeSession(queue=[FakeResponse(200, {"bars": None, "next_page_token": None})])
    out = make(fake).get_bars(["AAPL"], "1Hour", datetime(2024, 1, 1, tzinfo=timezone.utc),
                              datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert out == {}


def test_crypto_bars_use_v1beta3_endpoint():
    fake = FakeSession(queue=[FakeResponse(200, {
        "bars": {"BTC/USD": [_bar("2024-01-01T00:00:00Z", 40000.0)]}, "next_page_token": None,
    })])
    out = make(fake).get_crypto_bars(["BTC/USD"], "1Day", datetime(2024, 1, 1, tzinfo=timezone.utc),
                                     datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert fake.calls[0]["url"] == config.DATA_BASE_URL + "/v1beta3/crypto/us/bars"
    assert "BTC/USD" in out
    assert fake.calls[0]["headers"]["APCA-API-KEY-ID"] == "key"


# ----- orders -----

def test_bracket_buy_payload_exact_and_cent_rounding():
    fake = FakeSession(queue=[FakeResponse(200, {"id": "abc"})])
    result = make(fake).submit_bracket_buy(
        "AAPL", 7, stop_price=95.004, take_profit_price=105.995, client_order_id="cid-1",
    )
    assert result == {"id": "abc"}
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == config.PAPER_BASE_URL + "/v2/orders"
    assert call["json"] == {
        "symbol": "AAPL",
        "qty": "7",
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": 106.0},
        "stop_loss": {"stop_price": 95.0},
        "client_order_id": "cid-1",
    }


def test_bracket_buy_rounds_half_up_to_cent():
    fake = FakeSession(queue=[FakeResponse(200, {}), FakeResponse(200, {})])
    client = make(fake)
    client.submit_bracket_buy("X", 1, stop_price=10.005, take_profit_price=12.345, client_order_id="a")
    client.submit_bracket_buy("X", 1, stop_price=10.0049, take_profit_price=12.344, client_order_id="b")
    first, second = fake.calls
    assert first["json"]["stop_loss"]["stop_price"] == 10.01
    assert first["json"]["take_profit"]["limit_price"] == 12.35
    assert second["json"]["stop_loss"]["stop_price"] == 10.0
    assert second["json"]["take_profit"]["limit_price"] == 12.34


def test_market_sell_payload():
    fake = FakeSession(queue=[FakeResponse(200, {"id": "s1"})])
    make(fake).submit_market_sell("MSFT", 4, "cid-sell")
    assert fake.calls[0]["json"] == {
        "symbol": "MSFT", "qty": "4", "side": "sell", "type": "market",
        "time_in_force": "day", "client_order_id": "cid-sell",
    }


def test_stop_sell_payload():
    fake = FakeSession(queue=[FakeResponse(200, {"id": "s2"})])
    make(fake).submit_stop_sell("MSFT", 4, 388.666, "cid-stop")
    assert fake.calls[0]["json"] == {
        "symbol": "MSFT", "qty": "4", "side": "sell", "type": "stop",
        "time_in_force": "gtc", "stop_price": 388.67, "client_order_id": "cid-stop",
    }


def test_replace_order_patches_stop_price():
    fake = FakeSession(queue=[FakeResponse(200, {"id": "o9", "stop_price": "101.1"})])
    out = make(fake).replace_order("o9", 101.104)
    assert out["id"] == "o9"
    assert fake.calls[0]["method"] == "PATCH"
    assert fake.calls[0]["url"] == config.PAPER_BASE_URL + "/v2/orders/o9"
    assert fake.calls[0]["json"] == {"stop_price": 101.1}


@pytest.mark.parametrize("status", [200, 204, 404, 422])
def test_cancel_order_tolerates_gone_orders(status):
    payload = None if status == 204 else {}
    fake = FakeSession(queue=[FakeResponse(status, payload)])
    assert make(fake).cancel_order("o1") is None
    assert fake.calls[0]["method"] == "DELETE"
    assert fake.calls[0]["url"] == config.PAPER_BASE_URL + "/v2/orders/o1"


def test_cancel_order_other_errors_raise(no_sleep):
    fake = FakeSession(queue=[FakeResponse(403, {"message": "nope"})])
    with pytest.raises(AlpacaError):
        make(fake).cancel_order("o1")
