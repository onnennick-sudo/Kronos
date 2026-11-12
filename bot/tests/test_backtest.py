import json
import math
from datetime import date, datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot import backtest, config, timeutil
from bot.state import StateStore

ET = timeutil.ET
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=ET)
START = date(2023, 1, 2)          # a Monday
END = date(2026, 7, 31)


# ----- fakes -----------------------------------------------------------------------------

def _close(sym: str, idx: int) -> float:
    """Deterministic synthetic close for `sym` at integer index `idx` (sessions or days)."""
    return 100.0 + 0.1 * len(sym) + 5.0 * math.sin(idx / 9.0 + len(sym)) + 0.1 * idx


def _stock_idx(d: date) -> int:
    return int(np.busday_count(START, d))


def _crypto_idx(d: date) -> int:
    return (d - START).days


def _stamp(d: date, hour: int, tz) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=tz)


class FakeClient:
    """In-memory stand-in for AlpacaClient market-data calls (no network)."""

    def __init__(self, symbols, crypto=()):
        self.symbols = list(symbols)
        self.crypto = list(crypto)
        self.sessions = [START + timedelta(days=i) for i in range((END - START).days + 1)]
        self.sessions = [d for d in self.sessions if d.weekday() < 5]
        self.days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
        self.bar_calls = []

    def get_calendar(self, start, end):
        return [{"date": d, "open": dtime(9, 30), "close": dtime(16, 0)}
                for d in self.sessions if start <= d <= end]

    def _frame(self, sym, dates, idx_fn, hour, start, end):
        rows = []
        for d in dates:
            ts = _stamp(d, hour, timezone.utc)
            if start <= ts <= end:
                c = _close(sym, idx_fn(d))
                rows.append((ts, c, c, c, c, 1000.0))
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["timestamps", "open", "high", "low", "close", "volume"])

    def get_bars(self, symbols, timeframe, start, end, feed=None):
        self.bar_calls.append(("stocks", list(symbols), start, end))
        out = {}
        for s in symbols:
            if s in self.symbols or s == config.MARKET_SYMBOL:
                df = self._frame(s, self.sessions, _stock_idx, 5, start, end)
                if df is not None:
                    out[s] = df
        return out

    def get_crypto_bars(self, symbols, timeframe, start, end):
        self.bar_calls.append(("crypto", list(symbols), start, end))
        out = {}
        for s in symbols:
            df = self._frame(s, self.days, _crypto_idx, 0, start, end)
            if df is not None:
                out[s] = df
        return out


class FakeForecaster:
    """Records what it was given; returns scripted r_med per symbol."""

    def __init__(self, r_by_sym=None, agree=0.7):
        self.r_by_sym = r_by_sym or {}
        self.agree = agree
        self.calls = []

    def forecast(self, series, future, pred_len, n_paths, lookback, seed):
        self.calls.append({
            "max_hist": {s: df["timestamps"].max() for s, df in series.items()},
            "min_future": {s: future[s].min() for s in series},
            "n_paths": n_paths, "pred_len": pred_len, "lookback": lookback,
        })
        out = {}
        for s, df in series.items():
            out[s] = {"r_med": self.r_by_sym.get(s, 0.0), "agree": self.agree, "n": n_paths,
                      "last_close": float(df["close"].iloc[-1]), "final_closes": []}
        return out


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(config, "ETF_UNIVERSE", ["SPY", "QQQ"])
    monkeypatch.setattr(config, "CRYPTO_UNIVERSE", ["BTC/USD", "ETH/USD"])
    monkeypatch.setattr(config, "DAILY_LOOKBACK", 30)
    monkeypatch.setattr(timeutil, "now_et", lambda: NOW)
    monkeypatch.setattr(StateStore, "commit_and_push", lambda self, message, retries=3: True)


def _store(tmp_path):
    return StateStore(tmp_path)


def _chunk_dates(store, universe, chunks):
    out = []
    for i in range(chunks):
        payload = store.read_json(f"backtest/{universe}/chunk_{i}.json")
        out.extend(r["date"] for r in payload["dates"])
    return out


# ----- chunking --------------------------------------------------------------------------

@pytest.mark.parametrize("n,chunks", [(0, 3), (1, 3), (7, 3), (50, 3), (10, 1), (5, 8)])
def test_split_chunks_covers_every_item_exactly_once(n, chunks):
    items = list(range(n))
    parts = [backtest.split_chunks(items, i, chunks) for i in range(chunks)]
    flat = [x for p in parts for x in p]
    assert flat == items
    assert len(flat) == len(set(flat))


def test_chunk_runs_together_cover_all_decision_dates_once(small, tmp_path):
    store = _store(tmp_path)
    fc = FakeForecaster()
    client = FakeClient(config.UNIVERSE + [config.MARKET_SYMBOL])
    assert backtest.main(["--universe", "stocks", "--chunk", "0", "--chunks", "1"],
                         client=client, forecaster=fc, store=store) == 0
    expected = _chunk_dates(store, "stocks", 1)
    assert len(expected) > 40 and len(expected) == len(set(expected))

    for i in range(3):
        assert backtest.main(["--universe", "stocks", "--chunk", str(i), "--chunks", "3"],
                             client=client, forecaster=fc, store=store) == 0
    got = _chunk_dates(store, "stocks", 3)
    assert got == expected                      # same dates, same order, none repeated


def test_invalid_chunk_arguments_return_2(small, tmp_path):
    assert backtest.main(["--universe", "stocks", "--chunk", "3", "--chunks", "3"],
                         client=FakeClient([]), forecaster=FakeForecaster(), store=_store(tmp_path)) == 2


# ----- no lookahead ----------------------------------------------------------------------

def test_forecaster_never_sees_bars_after_decision_date(small, tmp_path):
    fc = FakeForecaster()
    client = FakeClient(["QQQ", "SPY"])
    assert backtest.main(["--universe", "etfs", "--chunks", "1"],
                         client=client, forecaster=fc, store=_store(tmp_path)) == 0
    assert fc.calls, "forecaster should have been called"
    for call in fc.calls:
        for sym, last_bar in call["max_hist"].items():
            # the next forecast stamp is strictly after the last visible bar
            assert last_bar < call["min_future"][sym]
            assert last_bar.date() < NOW.date()
            assert call["min_future"][sym].date() > last_bar.date()


def test_forward_return_is_five_sessions_after_decision_close(small, tmp_path):
    fc = FakeForecaster()
    store = _store(tmp_path)
    assert backtest.main(["--universe", "stocks", "--chunks", "1"],
                         client=FakeClient(config.UNIVERSE + ["SPY"]), forecaster=fc, store=store) == 0
    payload = store.read_json("backtest/stocks/chunk_0.json")
    checked = 0
    for rec in payload["dates"]:
        d = date.fromisoformat(rec["date"])
        i = _stock_idx(d)
        for sym, s in rec["symbols"].items():
            if "error" in s or s["fwd_ret"] is None:
                continue
            expected = _close(sym, i + 5) / _close(sym, i) - 1.0
            assert s["fwd_ret"] == pytest.approx(expected)
            checked += 1
        assert rec["bench"]["SPY"] == pytest.approx(_close("SPY", i + 5) / _close("SPY", i) - 1.0)
    assert checked > 100
    # bars stop at now - delay (June 1), so the last decision's 5-session close is still visible
    assert payload["dates"][-1]["date"] < "2026-06-01"


def test_crypto_uses_daily_decisions_and_utc_dates(small, tmp_path):
    fc = FakeForecaster()
    store = _store(tmp_path)
    client = FakeClient([], crypto=config.CRYPTO_UNIVERSE)
    assert backtest.main(["--universe", "crypto", "--chunks", "1"],
                         client=client, forecaster=fc, store=store) == 0
    payload = store.read_json("backtest/crypto/chunk_0.json")
    dates = [date.fromisoformat(r["date"]) for r in payload["dates"]]
    assert all((b - a).days == 5 for a, b in zip(dates, dates[1:]))
    # bars were not shifted: no symbol errors from stale/insufficient history
    rec = payload["dates"][len(payload["dates"]) // 2]
    d = date.fromisoformat(rec["date"])
    btc = rec["symbols"]["BTC/USD"]
    assert "error" not in btc
    assert btc["fwd_ret"] == pytest.approx(_close("BTC/USD", _crypto_idx(d) + 5) /
                                           _close("BTC/USD", _crypto_idx(d)) - 1.0)
    assert all(call["n_paths"] == config.DAILY_PATHS for call in fc.calls)


def test_paths_per_universe(small, tmp_path):
    fc_stock = FakeForecaster()
    backtest.main(["--universe", "stocks", "--chunks", "1"],
                  client=FakeClient(config.UNIVERSE + ["SPY"]), forecaster=fc_stock, store=_store(tmp_path))
    assert fc_stock.calls and all(c["n_paths"] == 5 for c in fc_stock.calls)
    fc_etf = FakeForecaster()
    backtest.main(["--universe", "etfs", "--chunks", "1"],
                  client=FakeClient(["SPY", "QQQ"]), forecaster=fc_etf, store=_store(tmp_path / "e"))
    assert fc_etf.calls and all(c["n_paths"] == config.DAILY_PATHS for c in fc_etf.calls)


def test_missing_history_is_reported_not_forecast(small, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAILY_LOOKBACK", 10_000)   # more rows than exist
    fc = FakeForecaster()
    store = _store(tmp_path)
    backtest.main(["--universe", "etfs", "--chunks", "1"],
                  client=FakeClient(["SPY", "QQQ"]), forecaster=fc, store=store)
    assert fc.calls == []
    payload = store.read_json("backtest/etfs/chunk_0.json")
    assert all("error" in s for r in payload["dates"] for s in r["symbols"].values())


# ----- report aggregation -------------------------------------------------------------------

def _rec(date_s, syms, spy):
    return {"date": date_s,
            "symbols": {k: {"r_med": v[0], "agree": v[1], "last_close": 100.0, "fwd_ret": v[2]}
                        for k, v in syms.items()},
            "bench": {"SPY": spy}}


def _synthetic_etf_records():
    # Hand-computed: A/B are the two names, SPY is the benchmark. Thresholds are the config defaults.
    return [
        _rec("2026-01-05", {"A": (0.02, 0.7, 0.03), "B": (-0.02, 0.7, -0.01)}, 0.01),
        _rec("2026-01-12", {"A": (0.02, 0.7, -0.02), "B": (0.015, 0.65, 0.04)}, 0.0),
        _rec("2026-01-19", {"A": (0.0, 0.5, 0.05), "B": (0.0, 0.5, -0.01)}, 0.0),
        _rec("2026-01-26", {"A": (0.03, 0.8, 0.01), "B": (0.03, 0.8, 0.02)}, 0.02),
    ]


def test_report_aggregation_known_values():
    rep = backtest.build_report("etfs", _synthetic_etf_records())
    pooled = rep["hit_rate"]["pooled"]
    # 8 pairs, 6 hits (A2 and A3 miss); 5 of 8 actual returns are positive
    assert pooled["n"] == 8
    assert pooled["hit_rate"] == pytest.approx(6 / 8)
    assert pooled["base_rate"] == pytest.approx(5 / 8)
    assert rep["hit_rate"]["per_symbol"]["A"]["hit_rate"] == pytest.approx(2 / 4)

    # per-period strategy returns: 0.0295, 0.0095, -0.0005, 0.0145 (5 bps on each full re-allocation)
    rets = [0.0295, 0.0095, -0.0005, 0.0145]
    st = rep["strategy"]
    assert [d["holdings"] for d in st["detail"]] == [["A"], ["A", "B"], [], ["A", "B"]]
    assert st["cumulative_return"] == pytest.approx(np.prod(1 + np.array(rets)) - 1)
    assert st["max_drawdown"] == pytest.approx(0.0005)
    assert st["avg_holdings"] == pytest.approx(1.25)
    assert st["cash_periods"] == 1
    expected_sharpe = np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(252 / 5)
    assert st["sharpe"] == pytest.approx(expected_sharpe)

    spy = rep["benchmarks"]["SPY"]
    assert spy["cumulative_return"] == pytest.approx(1.01 * 1.0 * 1.0 * 1.02 - 1)   # 0.0302
    assert spy["periods"] == 4


def test_grid_default_cell_matches_main_strategy():
    rep = backtest.build_report("etfs", _synthetic_etf_records())
    assert len(rep["grid"]) == 9
    cell = next(c for c in rep["grid"] if c["eligible_r"] == 0.01 and c["agree"] == 0.60)
    assert cell["cumulative_return"] == pytest.approx(rep["strategy"]["cumulative_return"])
    assert cell["avg_holdings"] == pytest.approx(rep["strategy"]["avg_holdings"])


def test_stocks_report_picks_top_three_and_warns_about_survivorship():
    recs = [_rec("2026-02-02", {
        "AAA": (0.05, 0.7, 0.01), "BBB": (0.04, 0.7, 0.02),
        "CCC": (0.03, 0.7, 0.03), "DDD": (0.02, 0.7, 0.04),
    }, 0.0)]
    rep = backtest.build_report("stocks", recs)
    st = rep["strategy"]
    assert st["detail"][0]["holdings"] == ["AAA", "BBB", "CCC"]
    # equal weight: mean(0.01, 0.02, 0.03) = 0.02, minus 5 bps on a full entry from cash
    assert st["cumulative_return"] == pytest.approx(0.0195)
    assert any("survivorship" in w.lower() for w in rep["warnings"])
    md = backtest.render_markdown(rep)
    assert "survivorship" in md.lower()
    assert "small sample, not proof of edge" in md.lower()


def test_report_with_no_scored_periods_is_safe():
    rep = backtest.build_report("crypto", [_rec("2026-05-01", {"BTC/USD": (0.0, 0.0, None)}, None)])
    assert rep["scored_periods"] == 0
    assert rep["strategy"]["cumulative_return"] is None
    assert "small sample, not proof of edge" in backtest.render_markdown(rep).lower()


def test_report_run_writes_json_and_markdown(small, tmp_path):
    store = _store(tmp_path)
    fc = FakeForecaster({"AAA": 0.05, "BBB": 0.04, "CCC": 0.03, "DDD": -0.05})
    client = FakeClient(config.UNIVERSE + ["SPY"])
    assert backtest.main(["--universe", "stocks", "--chunks", "2"],
                         client=client, forecaster=fc, store=store) == 0   # chunk 0 only
    # chunk 1 missing on the first report attempt
    assert backtest.main(["--universe", "stocks", "--chunks", "2", "--report"], store=store) == 0
    assert not (tmp_path / "state" / "backtest" / "stocks" / "report.json").exists()

    assert backtest.main(["--universe", "stocks", "--chunk", "1", "--chunks", "2"],
                         client=client, forecaster=fc, store=store) == 0
    assert backtest.main(["--universe", "stocks", "--chunks", "2", "--report"], store=store) == 0
    rep = json.loads((tmp_path / "state" / "backtest" / "stocks" / "report.json").read_text())
    assert rep["universe"] == "stocks" and rep["dates"]["count"] > 40
    md = (tmp_path / "state" / "backtest" / "stocks" / "report.md").read_text()
    assert "survivorship" in md.lower()


def test_missing_chunk_exits_zero_without_report(small, tmp_path, capsys):
    store = _store(tmp_path)
    store.write_json("backtest/etfs/chunk_0.json", {"dates": _synthetic_etf_records()})
    store.write_json("backtest/etfs/chunk_2.json", {"dates": []})
    rc = backtest.main(["--universe", "etfs", "--chunks", "3", "--report"], store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1]" in out and "missing" in out
    assert not (tmp_path / "state" / "backtest" / "etfs" / "report.json").exists()


def test_chunk_run_returns_one_when_push_fails(small, tmp_path, monkeypatch):
    monkeypatch.setattr(StateStore, "commit_and_push", lambda self, message, retries=3: False)
    rc = backtest.main(["--universe", "etfs", "--chunks", "1"],
                       client=FakeClient(["SPY", "QQQ"]), forecaster=FakeForecaster(), store=_store(tmp_path))
    assert rc == 1
    assert (tmp_path / "state" / "backtest" / "etfs" / "chunk_0.json").exists()
