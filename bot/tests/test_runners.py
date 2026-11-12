"""Offline tests for the runners (run_daily, run_trading, run_watchdog) and bot.runtime.

Fakes stand in for Alpaca (FakeClient), the forecaster (FakeForecaster), alert delivery and
git sync. FakeClient refuses every order-placing, cancelling or replacing call unless
allow_orders=True, and records each attempt, so dry-run tests can assert no order was tried.
"""

import copy
import csv
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from bot import alerts, config, run_daily, run_trading, run_watchdog, runtime, strategy
from bot.alpaca_client import AlpacaError
from bot.state import StateStore

ET = ZoneInfo("America/New_York")
SPY = config.MARKET_SYMBOL
UNIVERSE = list(config.UNIVERSE)
TODAY = date(2026, 11, 12)          # a Thursday
SLOT = "2026-11-12-10"
TRADE_NOW = datetime(2026, 11, 12, 11, 20, tzinfo=ET)
HOURLY_LONG = {"r_med": 0.005, "agree": 0.7}     # CONFIRM_LONG
HOURLY_NEG = {"r_med": -0.01, "agree": 0.7}      # STRONG_NEG
MSFT_POS = {"symbol": "MSFT", "qty": 10, "avg_entry_price": 100.0, "current_price": 101.0, "market_value": 1010.0}
MSFT_STOP = {"id": "stop-msft", "symbol": "MSFT", "side": "sell", "type": "stop",
             "status": "new", "stop_price": "95.0", "qty": "10"}


# ----- builders ---------------------------------------------------------------

def make_calendar(start: date, end: date, close=dtime(16, 0)) -> list[dict]:
    rows, d = [], start
    while d <= end:
        if d.weekday() < 5:
            rows.append({"date": d, "open": dtime(9, 30), "close": close})
        d += timedelta(days=1)
    return rows


def business_days_ending(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def closes_for(through: date, n: int = 60, base: float = 100.0) -> dict[date, float]:
    return {d: base + i * 0.01 for i, d in enumerate(business_days_ending(through, n))}


def daily_frame(closes: dict[date, float]) -> pd.DataFrame:
    days = sorted(closes)
    ts = [datetime.combine(d, dtime(0), tzinfo=ET).astimezone(timezone.utc) for d in days]
    close = [closes[d] for d in days]
    return pd.DataFrame({
        "timestamps": pd.to_datetime(ts, utc=True), "open": close,
        "high": [c * 1.01 for c in close], "low": [c * 0.99 for c in close],
        "close": close, "volume": [1_000_000.0] * len(days),
    })


def daily_table(through: date, overrides: dict[str, date] | None = None) -> dict[str, pd.DataFrame]:
    overrides = overrides or {}
    return {sym: daily_frame(closes_for(overrides.get(sym, through))) for sym in UNIVERSE + [SPY]}


def hourly_closes(through: datetime, days: int = 3, base: float = 100.0) -> dict[datetime, float]:
    labels = []
    for d in business_days_ending(through.date(), days):
        for h in config.REGULAR_HOUR_LABELS:
            label = datetime.combine(d, dtime(h))
            if label <= through:
                labels.append(label)
    return {label: base + i * 0.05 for i, label in enumerate(labels)}


def hourly_frame(through: datetime) -> pd.DataFrame:
    closes = hourly_closes(through)
    labels = sorted(closes)
    ts = [label.replace(tzinfo=ET).astimezone(timezone.utc) for label in labels]
    close = [closes[label] for label in labels]
    return pd.DataFrame({
        "timestamps": pd.to_datetime(ts, utc=True), "open": close, "high": close,
        "low": close, "close": close, "volume": [1_000.0] * len(labels),
    })


def row(sym, rank, r_med, agree=0.7, atr=2.0) -> dict:
    return {"symbol": sym, "rank": rank, "r_med": r_med, "agree": agree,
            "eligible": r_med >= config.DAILY_ELIGIBLE_R and agree >= config.DAILY_AGREE,
            "regime": strategy.classify_daily(r_med, agree), "atr": atr}


def save_ranking(store, rows, day=TODAY, spy="BULLISH", degraded=False):
    store.save_ranking(day, {
        "session": day.isoformat(), "as_of": day.isoformat(),
        "spy": {"regime": spy, "r_med": 0.02, "agree": 0.7},
        "ranking": rows, "candidates": strategy.candidates(rows), "errors": {}, "degraded": degraded,
    })


class FakeClient:
    """In-memory Alpaca stand-in. Order methods raise unless allow_orders=True."""

    def __init__(self, *, equity=100_000.0, last_equity=100_000.0, is_open=True, calendar=None,
                 bars=None, positions=(), open_orders=(), prices=None, allow_orders=False,
                 sell_error=None, replace_error=None, stubborn_cancel=False):
        self.account = {"equity": equity, "last_equity": last_equity, "cash": equity, "buying_power": equity}
        self.is_open = is_open
        self.calendar = calendar if calendar is not None else make_calendar(date(2026, 9, 1), date(2026, 12, 31))
        self.bars = bars or {}
        self.positions = [dict(p) for p in positions]
        self.open_orders = copy.deepcopy(list(open_orders))
        self.prices = dict(prices or {})
        self.allow_orders = allow_orders
        self.sell_error = sell_error
        self.replace_error = replace_error
        self.stubborn_cancel = stubborn_cancel
        self.order_attempts: list[str] = []   # every order-method call, including refused ones
        self.trace: list[tuple] = []          # order calls that were actually made, in order
        self.bar_requests: list[tuple] = []

    def get_account(self):
        return dict(self.account)

    def get_clock(self):
        return {"is_open": self.is_open}

    def get_calendar(self, start, end):
        return [dict(c) for c in self.calendar if start <= c["date"] <= end]

    def get_positions(self):
        return [dict(p) for p in self.positions]

    def get_open_orders(self):
        return copy.deepcopy(self.open_orders)

    def get_bars(self, symbols, timeframe, start, end, feed=None):
        self.bar_requests.append((tuple(symbols), timeframe))
        table = self.bars.get(timeframe, {})
        return {s: table[s].copy() for s in symbols if s in table}

    def get_latest_prices(self, symbols):
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def _order(self, name, *args):
        self.order_attempts.append(name)
        if not self.allow_orders:
            raise AssertionError(f"order method {name} called in a dry run")
        self.trace.append((name, *args))

    def submit_bracket_buy(self, symbol, qty, stop_price, take_profit_price, client_order_id):
        self._order("submit_bracket_buy", symbol, qty, stop_price, take_profit_price)
        return {"id": f"b-{symbol}"}

    def submit_market_sell(self, symbol, qty, client_order_id):
        self._order("submit_market_sell", symbol, qty)
        if self.sell_error is not None:
            raise self.sell_error
        return {"id": f"s-{symbol}"}

    def submit_stop_sell(self, symbol, qty, stop_price, client_order_id):
        self._order("submit_stop_sell", symbol, qty, stop_price)
        return {"id": f"r-{symbol}"}

    def cancel_order(self, order_id):
        self._order("cancel_order", order_id)
        if self.stubborn_cancel:
            return
        self.open_orders = [o for o in self.open_orders if o["id"] != order_id]
        for o in self.open_orders:
            o["legs"] = [leg for leg in (o.get("legs") or []) if leg["id"] != order_id]

    def replace_order(self, order_id, stop_price):
        self._order("replace_order", order_id, stop_price)
        if self.replace_error is not None:
            raise self.replace_error
        return {"id": order_id}


class FakeForecaster:
    """Returns a fixed (r_med, agree) per symbol; records every call."""

    def __init__(self, table=None, error=None):
        self.table = table or {}
        self.error = error
        self.calls: list[dict] = []

    def forecast(self, series, future, pred_len, n_paths, lookback, seed):
        self.calls.append({"symbols": sorted(series), "pred_len": pred_len, "seed": seed})
        if self.error is not None:
            raise self.error
        out = {}
        for sym, df in series.items():
            base = self.table.get(sym, {"r_med": 0.0, "agree": 0.5})
            if "error" in base:
                out[sym] = {"error": base["error"]}
                continue
            out[sym] = {"r_med": base["r_med"], "agree": base["agree"], "n": 10,
                        "last_close": float(df["close"].iloc[-1]), "final_closes": []}
        return out


@pytest.fixture(autouse=True)
def _no_bot_mode(monkeypatch):
    monkeypatch.delenv("BOT_MODE", raising=False)


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path)


@pytest.fixture
def pushes(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "send_push",
                        lambda title, message, priority="default", tags=None: sent.append((title, message, priority)) or True)
    return sent


@pytest.fixture
def emails(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "send_email", lambda subject, text, html=None: sent.append((subject, text)) or True)
    return sent


@pytest.fixture
def commits(monkeypatch):
    calls = []
    monkeypatch.setattr(StateStore, "commit_and_push", lambda self, message, retries=3: calls.append(message) or True)
    return calls


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(run_trading.time, "sleep", lambda s: slept.append(s))
    return slept


def trading_client(bar_syms=("AAPL",), stale=(), **kwargs) -> FakeClient:
    """Hourly bars through the decision bar (10:00 on TODAY) for bar_syms; stale ones end at 09:00."""
    hourly = {}
    for sym in bar_syms:
        through = datetime(2026, 11, 12, 9) if sym in stale else datetime(2026, 11, 12, 10)
        hourly[sym] = hourly_frame(through)
    return FakeClient(bars={"1Hour": hourly}, **kwargs)


def orders(store):
    return store.read(runtime.ORDERS_FILE)


# ----- runtime helpers --------------------------------------------------------

def test_bot_mode_is_paper_only_when_env_is_exact(monkeypatch):
    monkeypatch.delenv("BOT_MODE", raising=False)
    assert runtime.bot_mode(False) == "dry_run"
    monkeypatch.setenv("BOT_MODE", "paper")
    assert runtime.bot_mode(False) == "paper"
    assert runtime.bot_mode(True) == "dry_run"       # CLI flag forces dry run
    for value in ("Paper", " paper", "live", ""):
        monkeypatch.setenv("BOT_MODE", value)
        assert runtime.bot_mode(False) == "dry_run"


def test_flatten_orders_includes_legs_and_dedupes_by_id():
    bracket = {"id": "p1", "symbol": "AAPL", "side": "buy", "legs": [
        {"id": "tp1", "symbol": "AAPL", "side": "sell", "type": "limit"},
        {"id": "sl1", "symbol": "AAPL", "side": "sell", "type": "stop", "stop_price": "96"},
    ]}
    dup = {"id": "tp1", "symbol": "AAPL", "side": "sell", "type": "limit"}
    flat = runtime.flatten_orders([bracket, dup])
    assert [o["id"] for o in flat] == ["p1", "tp1", "sl1"]


def test_stop_orders_by_symbol_ignores_trailing_terminal_and_buys():
    flat = [
        {"id": "a", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "new", "stop_price": "95.5"},
        {"id": "b", "symbol": "MSFT", "side": "sell", "type": "trailing_stop", "status": "new", "trail_price": "2"},
        {"id": "c", "symbol": "NVDA", "side": "sell", "type": "stop", "status": "filled", "stop_price": "90"},
        {"id": "d", "symbol": "AMD", "side": "buy", "type": "stop", "status": "new", "stop_price": "80"},
        {"id": "e", "symbol": "CSCO", "side": "sell", "type": "stop_limit", "status": "held", "stop_price": "50"},
    ]
    assert runtime.stop_orders_by_symbol(flat) == {
        "AAPL": {"order_id": "a", "stop_price": 95.5},
        "CSCO": {"order_id": "e", "stop_price": 50.0},
    }


def test_alert_once_dedupes_and_retries_after_failed_push(store, monkeypatch):
    outcomes = iter([False, True, True])
    calls = []
    monkeypatch.setattr(alerts, "send_push", lambda *a, **k: calls.append(a) or next(outcomes))
    assert runtime.alert_once(store, "k1", "t", "m", "high") is False     # failed push: not recorded
    assert runtime.alert_once(store, "k1", "t", "m", "high") is True      # retried and sent
    assert runtime.alert_once(store, "k1", "t", "m", "high") is False     # deduped
    assert len(calls) == 2


# ----- daily ------------------------------------------------------------------

DAILY_TABLE = {"AAPL": {"r_med": 0.02, "agree": 0.7}, SPY: {"r_med": 0.02, "agree": 0.7}}
AFTER_CLOSE = datetime(2026, 11, 12, 17, 0, tzinfo=ET)


def daily_client(through=TODAY, overrides=None, hourly=None, **kwargs) -> FakeClient:
    bars = {"1Day": daily_table(through, overrides)}
    if hourly is not None:
        bars["1Hour"] = hourly
    return FakeClient(bars=bars, **kwargs)


def test_daily_happy_path_writes_state_email_and_metrics(store, emails, commits, pushes):
    client = daily_client(equity=101_000, last_equity=100_000)
    rc = run_daily.main(["--dry-run"], client=client, store=store,
                        forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE)
    assert rc == 0
    ranking = store.load_ranking(date(2026, 11, 13))
    assert ranking["as_of"] == "2026-11-12" and ranking["candidates"] == ["AAPL"]
    assert ranking["spy"]["regime"] == "BULLISH" and ranking["degraded"] is False
    daily_preds = [p for p in store.read(runtime.PREDICTIONS_FILE) if p["kind"] == "daily"]
    assert {p["symbol"] for p in daily_preds} == set(UNIVERSE)
    equity = store.read("equity.jsonl")
    assert len(equity) == 1 and equity[0]["date"] == "2026-11-12" and equity[0]["ew_index"] == 1.0
    with (store.dir / "metrics.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1
    run = [r for r in store.read("runs.jsonl") if r["kind"] == "daily"][-1]
    assert run["status"] == "ok" and run["key"] == "2026-11-13" and run["as_of"] == "2026-11-12"
    assert len(emails) == 1 and emails[0][0].startswith("[Kronos dry_run] 2026-11-12")
    assert emails[0][1].startswith("DRY RUN")
    assert len(commits) == 1
    assert client.order_attempts == []


def test_daily_second_run_is_a_noop(store, emails, commits, pushes):
    client = daily_client()
    args = dict(client=client, store=store, forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE)
    assert run_daily.main(["--dry-run"], **args) == 0
    snapshot = (len(store.read(runtime.PREDICTIONS_FILE)), len(store.read("equity.jsonl")), len(emails))
    assert run_daily.main(["--dry-run"], **args) == 0
    assert (len(store.read(runtime.PREDICTIONS_FILE)), len(store.read("equity.jsonl")), len(emails)) == snapshot


@pytest.mark.parametrize("now, last_final, target", [
    (datetime(2026, 11, 12, 17, 0, tzinfo=ET), date(2026, 11, 12), date(2026, 11, 13)),   # after the close
    (datetime(2026, 11, 12, 16, 10, tzinfo=ET), date(2026, 11, 11), date(2026, 11, 12)),  # before DAILY_READY_ET
    (datetime(2026, 11, 12, 9, 0, tzinfo=ET), date(2026, 11, 11), date(2026, 11, 12)),    # morning manual run
])
def test_daily_target_session_depends_on_time_of_day(store, emails, commits, pushes, now, last_final, target):
    client = daily_client(through=last_final)
    assert run_daily.main(["--dry-run"], client=client, store=store,
                          forecaster=FakeForecaster(DAILY_TABLE), now=now) == 0
    assert store.load_ranking(target)["as_of"] == last_final.isoformat()


def test_daily_isolates_stale_ticker(store, emails, commits, pushes):
    client = daily_client(overrides={"MSFT": date(2026, 11, 11)})
    run_daily.main(["--dry-run"], client=client, store=store, forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE)
    ranking = store.load_ranking(date(2026, 11, 13))
    assert ranking["errors"]["MSFT"] == "stale"
    assert "MSFT" not in {r["symbol"] for r in ranking["ranking"]}
    assert ranking["candidates"] == ["AAPL"]


def test_daily_degraded_without_spy_alerts_and_records_degraded(store, emails, commits, pushes):
    table = daily_table(TODAY)
    del table[SPY]
    client = FakeClient(bars={"1Day": table})
    assert run_daily.main(["--dry-run"], client=client, store=store,
                          forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE) == 0
    run = [r for r in store.read("runs.jsonl") if r["kind"] == "daily"][-1]
    assert run["status"] == "degraded"
    assert store.load_ranking(date(2026, 11, 13))["spy"]["regime"] == "UNKNOWN"
    assert [p for p in pushes if p[0] == "Kronos daily DEGRADED" and p[2] == "high"]


def test_daily_failure_records_error_and_pushes_urgent(store, emails, commits, pushes):
    client = daily_client()
    rc = run_daily.main(["--dry-run"], client=client, store=store,
                        forecaster=FakeForecaster(error=RuntimeError("model load failed")), now=AFTER_CLOSE)
    assert rc == 1
    run = [r for r in store.read("runs.jsonl") if r["kind"] == "daily"][-1]
    assert run["status"] == "error" and run["key"] == "2026-11-13" and "model load failed" in run["error"]
    assert [p for p in pushes if p[0] == "Kronos daily FAILED" and p[2] == "urgent"]
    assert emails == []                       # no report on failure
    assert commits == ["daily 2026-11-13: error"]   # the error record is synced to bot-state


def test_daily_scores_matured_daily_prediction(store, emails, commits, pushes):
    store.append(runtime.PREDICTIONS_FILE, {
        "kind": "daily", "as_of": "2026-10-29", "target_session": "2026-10-30", "symbol": "AAPL",
        "r_med": 0.02, "agree": 0.7, "last_close": 100.0, "horizon": 10,
    })
    client = daily_client()
    run_daily.main(["--dry-run"], client=client, store=store, forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE)
    outcome = [o for o in store.read("outcomes.jsonl") if o["kind"] == "daily"]
    assert len(outcome) == 1
    # Oct 29 + 10 sessions = Nov 12, the final bar.
    assert outcome[0]["as_of"] == "2026-10-29" and outcome[0]["horizon_end"] == "2026-11-12"
    assert outcome[0]["actual_r"] == pytest.approx(closes_for(TODAY)[TODAY] / 100.0 - 1.0)
    assert outcome[0]["hit"] is True


def test_hourly_prediction_scored_at_sixth_regular_bar(store, emails, commits, pushes):
    store.append(runtime.PREDICTIONS_FILE, {
        "kind": "hourly", "date": "2026-11-10", "slot": "2026-11-10-09", "bar_label": "2026-11-10T10:00:00",
        "symbol": "AAPL", "r_med": -0.01, "agree": 0.7, "last_close": 100.0, "horizon": 6,
    })
    hourly = {"AAPL": hourly_frame(datetime(2026, 11, 12, 15))}
    client = daily_client(hourly=hourly)
    run_daily.main(["--dry-run"], client=client, store=store, forecaster=FakeForecaster(DAILY_TABLE), now=AFTER_CLOSE)
    outcome = [o for o in store.read("outcomes.jsonl") if o["kind"] == "hourly"]
    # After Nov 10 10:00 the six regular bars are 11,12,13,14,15 on Nov 10, then 09:00 on Nov 11.
    assert len(outcome) == 1 and outcome[0]["horizon_end"] == "2026-11-11T09:00:00"
    expected = hourly_closes(datetime(2026, 11, 12, 15))[datetime(2026, 11, 11, 9)] / 100.0 - 1.0
    assert outcome[0]["actual_r"] == pytest.approx(expected)


def test_daily_forecast_is_seeded_by_last_final(store, emails, commits, pushes):
    fc = FakeForecaster(DAILY_TABLE)
    run_daily.main(["--dry-run"], client=daily_client(), store=store, forecaster=fc, now=AFTER_CLOSE)
    assert fc.calls[0]["seed"] == 20261112 and fc.calls[0]["pred_len"] == config.DAILY_PRED_LEN


# ----- trading ----------------------------------------------------------------

def test_dry_run_logs_entry_intent_and_places_no_orders(store, commits, pushes):
    save_ranking(store, [row("AAPL", 1, 0.02)])
    client = trading_client(prices={"AAPL": 100.0})
    rc = run_trading.main(["--dry-run"], client=client, store=store,
                          forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW)
    assert rc == 0
    assert client.order_attempts == []
    intents = [o for o in orders(store) if o["action"] == "entry"]
    assert len(intents) == 1 and intents[0]["status"] == "intent" and intents[0]["qty"] == 200
    assert intents[0]["stop"] == 96.0 and intents[0]["target"] == 106.0
    assert store.has_ok("trading", SLOT)
    assert store.read_json("positions.json")["symbols"] == []
    assert commits == [f"trading {SLOT} (dry_run)"]


def test_paper_places_bracket_buy_with_rounded_levels(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("AAPL", 1, 0.02)])
    client = trading_client(prices={"AAPL": 100.0}, allow_orders=True)
    assert run_trading.main([], client=client, store=store,
                            forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW) == 0
    assert client.trace == [("submit_bracket_buy", "AAPL", 200, 96.0, 106.0)]
    assert [o["status"] for o in orders(store) if o["action"] == "entry"] == ["submitted"]
    assert store.read_json("positions.json")["symbols"] == ["AAPL"]


def test_exit_cancels_then_confirms_then_sells(store, commits, pushes, sleeps, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("MSFT", 15, -0.002, agree=0.5)])   # rank 15 > ROTATION_RANK_EXIT
    client = trading_client(bar_syms=("MSFT",), positions=[MSFT_POS], open_orders=[MSFT_STOP],
                            prices={"MSFT": 101.0}, allow_orders=True)
    assert run_trading.main([], client=client, store=store,
                            forecaster=FakeForecaster(), now=TRADE_NOW) == 0
    assert client.trace == [("cancel_order", "stop-msft"), ("submit_market_sell", "MSFT", 10)]
    assert sleeps == []
    assert store.read_json("positions.json")["symbols"] == []


def test_failed_sell_restores_previous_stop(store, commits, pushes, sleeps, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("MSFT", 15, -0.002, agree=0.5)])
    client = trading_client(bar_syms=("MSFT",), positions=[MSFT_POS], open_orders=[MSFT_STOP],
                            prices={"MSFT": 101.0}, allow_orders=True,
                            sell_error=AlpacaError("market closed"))
    run_trading.main([], client=client, store=store, forecaster=FakeForecaster(), now=TRADE_NOW)
    assert client.trace[-2:] == [("submit_market_sell", "MSFT", 10), ("submit_stop_sell", "MSFT", 10, 95.0)]
    assert [p for p in pushes if p[0] == "Kronos exit FAILED" and p[2] == "urgent"]
    assert "MSFT" in store.read_json("positions.json")["symbols"]


def test_unconfirmed_cancel_skips_the_sell(store, commits, pushes, sleeps, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("MSFT", 15, -0.002, agree=0.5)])
    client = trading_client(bar_syms=("MSFT",), positions=[MSFT_POS], open_orders=[MSFT_STOP],
                            prices={"MSFT": 101.0}, allow_orders=True, stubborn_cancel=True)
    run_trading.main([], client=client, store=store, forecaster=FakeForecaster(), now=TRADE_NOW)
    assert client.trace == [("cancel_order", "stop-msft")]
    assert sleeps == [run_trading.CONFIRM_SLEEP_S] * run_trading.CONFIRM_READS
    assert [p for p in pushes if p[0] == "Kronos exit BLOCKED" and p[2] == "urgent"]
    assert "MSFT" in store.read_json("positions.json")["symbols"]


@pytest.mark.parametrize("env, allow, status, cancels", [
    (None, False, "intent", False),     # dry run: logged, never sent
    ("paper", True, "cancelled", True),
])
def test_halt_records_breach_cancels_buys_and_blocks_entries(store, commits, pushes, monkeypatch,
                                                             env, allow, status, cancels):
    if env:
        monkeypatch.setenv("BOT_MODE", env)
    save_ranking(store, [row("AAPL", 1, 0.02)])
    buy = {"id": "buy-nvda", "symbol": "NVDA", "side": "buy", "type": "market", "status": "new"}
    client = trading_client(equity=97_000, last_equity=100_000, open_orders=[buy],
                            prices={"AAPL": 100.0}, allow_orders=allow)
    assert run_trading.main([], client=client, store=store,
                            forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW) == 0
    assert store.halt_today(TODAY) is not None
    cancel_rows = [o for o in orders(store) if o["action"] == "cancel_buy"]
    assert [(o["symbol"], o["status"]) for o in cancel_rows] == [("NVDA", status)]
    assert [t for t in client.trace if t[0] == "submit_bracket_buy"] == []
    assert (("cancel_order", "buy-nvda") in client.trace) is cancels
    assert [p for p in pushes if p[0] == "Kronos HALTED" and p[2] == "urgent"]


def test_missing_ranking_blocks_entries_but_still_exits(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("MSFT", 15, -0.002, agree=0.5), row("AAPL", 1, 0.02)], day=date(2026, 11, 11))
    client = trading_client(bar_syms=("AAPL", "MSFT"), positions=[MSFT_POS], open_orders=[MSFT_STOP],
                            prices={"AAPL": 100.0, "MSFT": 101.0}, allow_orders=True)
    assert run_trading.main([], client=client, store=store,
                            forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW) == 0
    assert ("submit_market_sell", "MSFT", 10) in client.trace
    assert not [t for t in client.trace if t[0] == "submit_bracket_buy"]
    assert [p for p in pushes if p[0] == "Kronos: no ranking for today"]


def test_stale_hourly_ticker_is_isolated(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("AAPL", 1, 0.02), row("MSFT", 2, 0.015)])
    client = trading_client(bar_syms=("AAPL", "MSFT"), stale=("MSFT",),
                            prices={"AAPL": 100.0, "MSFT": 100.0}, allow_orders=True)
    run_trading.main([], client=client, store=store,
                     forecaster=FakeForecaster({"AAPL": HOURLY_LONG, "MSFT": HOURLY_LONG}), now=TRADE_NOW)
    buys = [t[1] for t in client.trace if t[0] == "submit_bracket_buy"]
    assert buys == ["AAPL"]
    run = [r for r in store.read("runs.jsonl") if r["kind"] == "trading"][-1]
    assert ["MSFT", "hourly_error"] in run["skipped"]


def test_slot_is_idempotent(store, commits, pushes):
    save_ranking(store, [row("AAPL", 1, 0.02)])
    client = trading_client(prices={"AAPL": 100.0})
    args = dict(client=client, store=store, forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW)
    assert run_trading.main(["--dry-run"], **args) == 0
    n_orders, n_requests = len(orders(store)), len(client.bar_requests)
    assert run_trading.main(["--dry-run"], **args) == 0
    assert len(orders(store)) == n_orders and len(client.bar_requests) == n_requests


def test_stopped_out_symbol_is_not_reentered_today(store, commits, pushes):
    store.write_json("positions.json", {"date": TODAY.isoformat(), "symbols": ["NVDA"]})
    save_ranking(store, [row("NVDA", 1, 0.02)])
    client = trading_client(bar_syms=("NVDA",), prices={"NVDA": 100.0})
    run_trading.main(["--dry-run"], client=client, store=store,
                     forecaster=FakeForecaster({"NVDA": HOURLY_LONG}), now=TRADE_NOW)
    assert [s["symbol"] for s in store.read("stopped_out.jsonl")] == ["NVDA"]
    run = [r for r in store.read("runs.jsonl") if r["kind"] == "trading"][-1]
    assert ["NVDA", "stopped_out_today"] in run["skipped"]
    assert client.order_attempts == []


def test_market_closed_writes_no_record_and_fetches_nothing(store, commits, pushes):
    save_ranking(store, [row("AAPL", 1, 0.02)])
    client = trading_client(is_open=False, prices={"AAPL": 100.0})
    assert run_trading.main(["--dry-run"], client=client, store=store,
                            forecaster=FakeForecaster({"AAPL": HOURLY_LONG}), now=TRADE_NOW) == 0
    assert store.read("runs.jsonl") == [] and client.bar_requests == []


def test_hourly_forecaster_outage_blocks_entries_and_alerts(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("AAPL", 1, 0.02)])
    client = trading_client(prices={"AAPL": 100.0}, allow_orders=True)
    rc = run_trading.main([], client=client, store=store,
                          forecaster=FakeForecaster(error=RuntimeError("weights missing")), now=TRADE_NOW)
    assert rc == 0 and client.trace == []
    assert [p for p in pushes if p[0] == "Kronos hourly forecast FAILED" and p[2] == "urgent"]


def test_trail_raises_stop_in_paper_mode(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("AAPL", 1, 0.02, atr=2.0)])
    aapl = {"symbol": "AAPL", "qty": 10, "avg_entry_price": 100.0, "current_price": 105.0, "market_value": 1050.0}
    stop = {"id": "stop-aapl", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "new", "stop_price": "95.0"}
    client = trading_client(positions=[aapl], open_orders=[stop], prices={"AAPL": 105.0}, allow_orders=True)
    run_trading.main([], client=client, store=store, forecaster=FakeForecaster(), now=TRADE_NOW)
    assert client.trace == [("replace_order", "stop-aapl", 102.0)]


def test_failed_trail_keeps_old_stop_and_alerts(store, commits, pushes, monkeypatch):
    monkeypatch.setenv("BOT_MODE", "paper")
    save_ranking(store, [row("AAPL", 1, 0.02, atr=2.0)])
    aapl = {"symbol": "AAPL", "qty": 10, "avg_entry_price": 100.0, "current_price": 105.0, "market_value": 1050.0}
    stop = {"id": "stop-aapl", "symbol": "AAPL", "side": "sell", "type": "stop", "status": "new", "stop_price": "95.0"}
    client = trading_client(positions=[aapl], open_orders=[stop], prices={"AAPL": 105.0}, allow_orders=True,
                            replace_error=AlpacaError("rejected"))
    run_trading.main([], client=client, store=store, forecaster=FakeForecaster(), now=TRADE_NOW)
    assert [p for p in pushes if p[0] == "Kronos trail FAILED" and p[2] == "high"]
    assert store.read_json("positions.json")["symbols"] == ["AAPL"]


# ----- watchdog ---------------------------------------------------------------

WATCH_NOON = datetime(2026, 11, 12, 13, 0, tzinfo=ET)
WATCH_EVENING = datetime(2026, 11, 12, 19, 0, tzinfo=ET)


def test_trading_gap_alerts_once_per_hour(store, commits, pushes):
    client = FakeClient(is_open=True)
    assert run_watchdog.main([], client=client, store=store, now=WATCH_NOON) == 0
    assert len(pushes) == 1 and pushes[0][2] == "urgent"
    assert commits == ["watchdog 2026-11-12"]
    assert run_watchdog.main([], client=client, store=store, now=WATCH_NOON) == 0
    assert len(pushes) == 1 and len(commits) == 1
    assert client.order_attempts == []


def test_recent_successful_trading_slot_suppresses_gap_alert(store, commits, pushes):
    ts = datetime(2026, 11, 12, 12, 30, tzinfo=ET).isoformat()
    store.append("runs.jsonl", {"kind": "trading", "key": "2026-11-12-11", "status": "ok", "ts": ts})
    run_watchdog.main([], client=FakeClient(is_open=True), store=store, now=WATCH_NOON)
    assert pushes == [] and commits == []


def test_missing_daily_run_alerts_after_deadline(store, commits, pushes):
    run_watchdog.main([], client=FakeClient(is_open=False), store=store, now=WATCH_EVENING)
    assert [p for p in pushes if p[0] == "Kronos: daily run missing" and p[2] == "high"]
    assert commits == ["watchdog 2026-11-12"]


def test_recorded_daily_run_suppresses_daily_alert(store, commits, pushes):
    store.record_run("daily", "2026-11-13", "ok", as_of="2026-11-12")
    run_watchdog.main([], client=FakeClient(is_open=False), store=store, now=WATCH_EVENING)
    assert pushes == [] and commits == []


def test_watchdog_is_quiet_before_deadline_when_market_closed(store, commits, pushes):
    run_watchdog.main([], client=FakeClient(is_open=False), store=store, now=WATCH_NOON)
    assert pushes == [] and commits == []
