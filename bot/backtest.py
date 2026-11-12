"""Walk-forward backtest of the Kronos daily signal (stocks, ETFs, crypto).

Universes:
- stocks: config.UNIVERSE, hold up to 3 names by forecast return, SPY benchmark.
- etfs:   config.ETF_UNIVERSE, hold every BULLISH name, SPY benchmark.
- crypto: config.CRYPTO_UNIVERSE, 24/7 (calendar days), hold every BULLISH name,
          equal-weight buy-and-hold benchmark plus SPY when available.

Usage (from repo root, PYTHONPATH=.):
    python -m bot.backtest --universe stocks --chunk 0 --chunks 3   # one chunk of decision dates
    python -m bot.backtest --universe stocks --report --chunks 3    # aggregate once all chunks exist

Chunks go to the bot-state store as backtest/<universe>/chunk_<i>.json; the report goes to
backtest/<universe>/report.json and report.md. No lookahead: a decision on date d is given
only bars dated <= d, and forward returns are measured from the close on d.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from bot import config, metrics, strategy, timeutil
from bot.alpaca_client import AlpacaClient, AlpacaError
from bot.forecast import KronosForecaster, seed_for
from bot.state import StateStore

log = logging.getLogger(__name__)

UNIVERSES = ("stocks", "etfs", "crypto")
HORIZON = 5                        # sessions (crypto: days) held and scored
DECISION_STEP = HORIZON            # a decision every 5 sessions (crypto: 5 days)
LOOKBACK_CAL_DAYS = 800            # calendar days of bars fetched before a chunk's first decision
CALENDAR_FWD_DAYS = 60             # calendar requested past today so future sessions exist
STOCK_HISTORY_DAYS = 365
ETF_CRYPTO_HISTORY_DAYS = 730
STOCK_PATHS = 5                    # per the wave-2 contract (DAILY_PATHS override for stocks)
STOCK_HOLDINGS = 3
GRID_R = (0.005, 0.01, 0.015)
GRID_AGREE = (0.55, 0.60, 0.65)
DISCLAIMER = "Small sample, not proof of edge."
SURVIVORSHIP_WARNING = (
    "Survivorship bias: the stock universe is today's largest S&P 500 names, frozen at build. "
    "Companies that were large in the past but dropped out (or were removed) are absent, so "
    "historical stock results are biased upward."
)
RULE_TEXT = {
    "stocks": "Hold up to 3 stocks with the highest forecast return that clear the thresholds, "
              "equal weight; cash when none qualify.",
    "etfs": "Hold every name with a BULLISH daily signal, equal weight; cash when none qualify.",
    "crypto": "Hold every name with a BULLISH daily signal, equal weight; cash when none qualify.",
}


# ----- settings and helpers ----------------------------------------------------------

def _settings(universe: str) -> dict[str, Any]:
    if universe == "stocks":
        return {"symbols": list(config.UNIVERSE), "crypto": False,
                "history_days": STOCK_HISTORY_DAYS, "paths": STOCK_PATHS}
    if universe == "etfs":
        return {"symbols": list(config.ETF_UNIVERSE), "crypto": False,
                "history_days": ETF_CRYPTO_HISTORY_DAYS, "paths": config.DAILY_PATHS}
    if universe == "crypto":
        return {"symbols": list(config.CRYPTO_UNIVERSE), "crypto": True,
                "history_days": ETF_CRYPTO_HISTORY_DAYS, "paths": config.DAILY_PATHS}
    raise ValueError(f"unknown universe: {universe!r}")


def _chunk_name(universe: str, chunk: int) -> str:
    return f"backtest/{universe}/chunk_{chunk}.json"


def split_chunks(items: list, chunk: int, chunks: int) -> list:
    """Contiguous part `chunk` of `chunks`. The parts cover `items` exactly once."""
    n = len(items)
    return items[chunk * n // chunks:(chunk + 1) * n // chunks]


def _decision_dates(crypto: bool, decision_start: date, today: date, sessions: list[date]) -> list[date]:
    """Decision dates strictly before today, every DECISION_STEP sessions (days for crypto)."""
    if crypto:
        span = (today - decision_start).days
        return [decision_start + timedelta(days=k) for k in range(0, span, DECISION_STEP)]
    window = [d for d in sessions if decision_start <= d < today]
    return window[::DECISION_STEP]


def _daily_frames(raw: dict[str, pd.DataFrame], crypto: bool) -> dict[str, pd.DataFrame]:
    """Daily frames keyed by a naive date-at-midnight `timestamps` column.

    Stocks use the project convention (ET dates). Crypto bars sit at 00:00 UTC, so their
    date is the UTC date; converting to ET would shift every bar back one day.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym, df in raw.items():
        if df is None or len(df) == 0:
            continue
        if crypto:
            d = df.copy()
            d["timestamps"] = pd.to_datetime(d["timestamps"], utc=True).dt.tz_localize(None).dt.normalize()
            d = d.sort_values("timestamps").drop_duplicates("timestamps", keep="last")
        else:
            d = timeutil.daily_df_for_model(df).drop_duplicates("timestamps", keep="last")
        out[sym] = d.reset_index(drop=True)
    return out


def _closes_by_date(frame: pd.DataFrame | None) -> dict[date, float]:
    if frame is None:
        return {}
    return {ts.date(): float(c) for ts, c in zip(frame["timestamps"], frame["close"])}


def _target_date(d: date, crypto: bool, sess_pos: dict[date, int], sessions: list[date]) -> date | None:
    """The date HORIZON sessions after d (calendar days for crypto); None if not on the calendar."""
    if crypto:
        return d + timedelta(days=HORIZON)
    i = sess_pos.get(d)
    if i is None or i + HORIZON >= len(sessions):
        return None
    return sessions[i + HORIZON]


def _fwd(closes: dict[date, float], d: date, target: date | None) -> float | None:
    if target is None:
        return None
    c0 = closes.get(d)
    c1 = closes.get(target)
    if c0 is None or c1 is None or c0 <= 0:
        return None
    return c1 / c0 - 1.0


def _future_stamps(crypto: bool, d: date, cal: list[dict]) -> pd.Series:
    if crypto:
        return pd.Series(
            pd.to_datetime([datetime.combine(d + timedelta(days=i), dtime(0))
                            for i in range(1, config.DAILY_PRED_LEN + 1)]),
            name="timestamps",
        )
    return timeutil.future_daily_stamps(cal, d, config.DAILY_PRED_LEN)


def _fetch_bars(client, settings: dict[str, Any], first_decision: date, now: datetime) -> dict[str, pd.DataFrame]:
    """One bars fetch per run: from 800 days before the first decision to now - delay."""
    start = datetime.combine(first_decision - timedelta(days=LOOKBACK_CAL_DAYS), dtime(0), tzinfo=timeutil.ET)
    end = now.astimezone(timezone.utc) - timedelta(minutes=config.DATA_DELAY_MINUTES)
    symbols = settings["symbols"]
    if settings["crypto"]:
        raw = dict(client.get_crypto_bars(symbols, "1Day", start, end))
        try:
            spy = client.get_bars([config.MARKET_SYMBOL], "1Day", start, end)
            if spy.get(config.MARKET_SYMBOL) is not None:
                raw[config.MARKET_SYMBOL] = spy[config.MARKET_SYMBOL]
        except AlpacaError as exc:
            log.warning("SPY benchmark unavailable for crypto run: %s", exc)
    else:
        wanted = list(dict.fromkeys(symbols + [config.MARKET_SYMBOL]))
        raw = dict(client.get_bars(wanted, "1Day", start, end))
    return _daily_frames(raw, settings["crypto"])


# ----- chunk run -----------------------------------------------------------------------

def _decision_record(d: date, *, symbols: list[str], bars: dict[str, pd.DataFrame],
                     closes: dict[str, dict[date, float]], cal: list[dict], sess_pos: dict[date, int],
                     sessions: list[date], forecaster, paths: int, crypto: bool) -> dict[str, Any]:
    d_ts = pd.Timestamp(d)
    series: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        frame = bars.get(sym)
        if frame is None:
            errors[sym] = "no bars"
            continue
        hist = frame[frame["timestamps"] <= d_ts]          # no lookahead: only bars dated <= d
        if len(hist) < config.DAILY_LOOKBACK:
            errors[sym] = f"insufficient history ({len(hist)} rows)"
            continue
        last = hist["timestamps"].iloc[-1].date()
        if last != d:
            errors[sym] = f"stale: last bar {last.isoformat()}"
            continue
        series[sym] = hist

    results: dict[str, dict] = {}
    if series:
        stamps = _future_stamps(crypto, d, cal)
        future = {sym: stamps for sym in series}
        results = forecaster.forecast(series, future, config.DAILY_PRED_LEN, paths,
                                      config.DAILY_LOOKBACK, seed_for(d, "daily"))

    target = _target_date(d, crypto, sess_pos, sessions)
    out_syms: dict[str, dict] = {}
    for sym in symbols:
        if sym in errors:
            out_syms[sym] = {"error": errors[sym]}
            continue
        res = results.get(sym)
        if res is None or "error" in res:
            out_syms[sym] = {"error": (res or {}).get("error", "no forecast")}
            continue
        out_syms[sym] = {
            "r_med": float(res["r_med"]),
            "agree": float(res["agree"]),
            "last_close": float(res["last_close"]),
            "fwd_ret": _fwd(closes.get(sym, {}), d, target),
        }
    bench = {config.MARKET_SYMBOL: _fwd(closes.get(config.MARKET_SYMBOL, {}), d, target)}
    return {"date": d.isoformat(), "symbols": out_syms, "bench": bench}


def run_chunk(universe: str, chunk: int, chunks: int, client, forecaster, store: StateStore) -> int:
    settings = _settings(universe)
    crypto = settings["crypto"]
    now = timeutil.now_et()
    today = now.date()
    decision_start = today - timedelta(days=settings["history_days"])

    cal: list[dict] = []
    if not crypto:
        cal = client.get_calendar(decision_start, today + timedelta(days=CALENDAR_FWD_DAYS))
    sessions = sorted({c["date"] for c in cal if c["date"].weekday() < 5})
    sess_pos = {d: i for i, d in enumerate(sessions)}

    all_dates = _decision_dates(crypto, decision_start, today, sessions)
    my_dates = split_chunks(all_dates, chunk, chunks)
    log.info("backtest %s chunk %d/%d: %d of %d decision dates", universe, chunk, chunks,
             len(my_dates), len(all_dates))

    bars: dict[str, pd.DataFrame] = {}
    if my_dates:
        bars = _fetch_bars(client, settings, my_dates[0], now)
    closes = {sym: _closes_by_date(frame) for sym, frame in bars.items()}

    records = [
        _decision_record(d, symbols=settings["symbols"], bars=bars, closes=closes, cal=cal,
                         sess_pos=sess_pos, sessions=sessions, forecaster=forecaster,
                         paths=settings["paths"], crypto=crypto)
        for d in my_dates
    ]

    payload = {
        "universe": universe,
        "chunk": chunk,
        "chunks": chunks,
        "horizon": HORIZON,
        "pred_len": config.DAILY_PRED_LEN,
        "lookback": config.DAILY_LOOKBACK,
        "paths": settings["paths"],
        "generated_at": now.isoformat(),
        "dates": records,
    }
    store.write_json(_chunk_name(universe, chunk), payload)
    ok = store.commit_and_push(f"backtest {universe} chunk {chunk}/{chunks}")
    log.info("wrote %s with %d dates (push ok=%s)", _chunk_name(universe, chunk), len(records), ok)
    return 0 if ok else 1


# ----- strategy simulation ---------------------------------------------------------------

def _with_forecast(syms: dict[str, dict]) -> dict[str, dict]:
    return {s: r for s, r in syms.items() if "error" not in r and r.get("fwd_ret") is not None}


def _rule_default(universe: str) -> Callable[[dict[str, dict]], list[str]]:
    """Holdings using the project's own strategy functions at config thresholds."""
    if universe == "stocks":
        def pick(syms: dict[str, dict]) -> list[str]:
            stats = _with_forecast(syms)
            ranking = strategy.build_ranking(stats, {})
            return strategy.candidates(ranking, n=STOCK_HOLDINGS)
        return pick

    def pick_bullish(syms: dict[str, dict]) -> list[str]:
        return sorted(s for s, r in _with_forecast(syms).items()
                      if strategy.classify_daily(r["r_med"], r["agree"]) == "BULLISH")
    return pick_bullish


def _rule_custom(universe: str, min_r: float, min_agree: float) -> Callable[[dict[str, dict]], list[str]]:
    """Same rule as _rule_default with explicit thresholds (used by the sensitivity grid)."""
    def pick(syms: dict[str, dict]) -> list[str]:
        ok = [(r["r_med"], s) for s, r in _with_forecast(syms).items()
              if r["r_med"] >= min_r and r["agree"] >= min_agree]
        ok.sort(key=lambda t: (-t[0], t[1]))
        names = [s for _, s in ok]
        return names[:STOCK_HOLDINGS] if universe == "stocks" else sorted(names)
    return pick


def _perf(rets: list[float]) -> dict[str, Any]:
    if not rets:
        return {"cumulative_return": None, "sharpe": None, "max_drawdown": None}
    arr = np.asarray(rets, dtype=float)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + arr)])
    # metrics.sharpe annualises with sqrt(252); 5-session periods need sqrt(252/5).
    sharpe = metrics.sharpe(pd.Series(arr)) / math.sqrt(HORIZON)
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": float(metrics.max_drawdown(pd.Series(equity))),
    }


def _simulate(periods: list[dict], pick: Callable[[dict[str, dict]], list[str]]) -> dict[str, Any]:
    """Equal-weight holdings rebalanced each period; cost = bps/1e4 * traded fraction * 2 sides.

    Traded fraction is the one-way turnover sum(|w_new - w_old|) / 2, so the charge is
    bps/1e4 * sum(|w_new - w_old|): each side of each trade pays COST_BPS_PER_SIDE.
    """
    cost_rate = config.COST_BPS_PER_SIDE / 1e4
    prev: dict[str, float] = {}
    rets: list[float] = []
    counts: list[int] = []
    traded_all: list[float] = []
    detail: list[dict] = []
    for r in periods:
        syms = r["symbols"]
        hold = pick(syms)
        w = {s: 1.0 / len(hold) for s in hold} if hold else {}
        traded = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in set(w) | set(prev))
        gross = sum(w[s] * float(syms[s]["fwd_ret"]) for s in w)
        ret = gross - cost_rate * traded
        rets.append(ret)
        counts.append(len(hold))
        traded_all.append(traded)
        detail.append({"date": r["date"], "holdings": hold, "return": ret})
        prev = w
    out = _perf(rets)
    out.update({
        "periods": len(rets),
        "avg_holdings": float(np.mean(counts)) if counts else None,
        "cash_periods": sum(1 for c in counts if c == 0),
        "avg_turnover": float(np.mean(traded_all)) if traded_all else None,
        "detail": detail,
    })
    return out


def _hit(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    hit, base, n = metrics.hit_rate(pairs)
    return {"hit_rate": hit, "base_rate": base, "edge_vs_base": (hit - base) if n else None, "n": n}


def _ew_fwd(record: dict) -> float | None:
    vals = [float(s["fwd_ret"]) for s in record["symbols"].values()
            if "error" not in s and s.get("fwd_ret") is not None]
    return float(np.mean(vals)) if vals else None


def build_report(universe: str, records: list[dict]) -> dict[str, Any]:
    """Aggregate chunk date records into the report dict. Pure: no I/O."""
    crypto = universe == "crypto"
    records = sorted(records, key=lambda r: r["date"])
    settings = _settings(universe)

    pooled: list[tuple[float, float]] = []
    per_sym: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        for sym, s in r["symbols"].items():
            if "error" in s or s.get("fwd_ret") is None:
                continue
            pair = (float(s["r_med"]), float(s["fwd_ret"]))
            pooled.append(pair)
            per_sym.setdefault(sym, []).append(pair)

    if crypto:
        periods = [r for r in records if _ew_fwd(r) is not None]
    else:
        periods = [r for r in records if r["bench"].get(config.MARKET_SYMBOL) is not None]

    strat = _simulate(periods, _rule_default(universe))
    grid = []
    for min_r in GRID_R:
        for min_agree in GRID_AGREE:
            cell = _simulate(periods, _rule_custom(universe, min_r, min_agree))
            grid.append({
                "eligible_r": min_r,
                "agree": min_agree,
                "cumulative_return": cell["cumulative_return"],
                "sharpe": cell["sharpe"],
                "max_drawdown": cell["max_drawdown"],
                "avg_holdings": cell["avg_holdings"],
                "periods": cell["periods"],
            })

    benchmarks: dict[str, Any] = {}
    spy_periods = [r for r in records if r["bench"].get(config.MARKET_SYMBOL) is not None]
    spy_rets = [float(r["bench"][config.MARKET_SYMBOL]) for r in spy_periods]
    if crypto:
        ew_rets = [_ew_fwd(r) for r in periods]
        benchmarks["EW_BTC_ETH"] = {
            "label": "Equal-weight BTC/ETH, buy and hold rebalanced every 5 days, no costs",
            **_perf(ew_rets), "periods": len(ew_rets),
        }
        benchmarks[config.MARKET_SYMBOL] = {
            "label": "SPY buy and hold (same dates)", **_perf(spy_rets), "periods": len(spy_rets),
        }
    else:
        benchmarks[config.MARKET_SYMBOL] = {
            "label": "SPY buy and hold (same dates)",
            **_perf([float(r["bench"][config.MARKET_SYMBOL]) for r in periods]),
            "periods": len(periods),
        }

    warnings: list[str] = []
    if universe == "stocks":
        warnings.append(SURVIVORSHIP_WARNING)
    if config.DAILY_PRED_LEN != HORIZON:
        warnings.append(
            f"Forecasts look {config.DAILY_PRED_LEN} sessions ahead but returns are scored over "
            f"{HORIZON}; hit rates compare those two horizons."
        )
    if not periods:
        warnings.append("No scored decision dates yet; nothing to evaluate.")
    elif len(periods) < 20:
        warnings.append(f"Only {len(periods)} scored decision dates; treat every figure as noise.")
    warnings.append(DISCLAIMER)

    return {
        "universe": universe,
        "dates": {
            "count": len(records),
            "first": records[0]["date"] if records else None,
            "last": records[-1]["date"] if records else None,
        },
        "scored_periods": len(periods),
        "horizon": HORIZON,
        "forecast": {"pred_len": config.DAILY_PRED_LEN, "paths": settings["paths"],
                     "lookback": config.DAILY_LOOKBACK},
        "cost_bps_per_side": config.COST_BPS_PER_SIDE,
        "hit_rate": {
            "pooled": _hit(pooled),
            "per_symbol": {s: _hit(p) for s, p in sorted(per_sym.items())},
        },
        "strategy": {"rule": RULE_TEXT[universe], **strat},
        "benchmarks": benchmarks,
        "grid": grid,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render_markdown(rep: dict[str, Any]) -> str:
    u = rep["universe"]
    unit = "days" if u == "crypto" else "sessions"
    hr = rep["hit_rate"]["pooled"]
    st = rep["strategy"]
    lines = [
        f"# Kronos daily backtest: {u}",
        "",
        "## Setup",
        f"- Decision dates: {rep['dates']['count']} ({rep['dates']['first']} to {rep['dates']['last']}), "
        f"one every {HORIZON} {unit}.",
        f"- Scored decision dates: {rep['scored_periods']}.",
        f"- Forecast: {rep['forecast']['pred_len']} {unit} ahead, {rep['forecast']['paths']} sampled paths, "
        f"{rep['forecast']['lookback']}-bar lookback. Returns are scored over {HORIZON} {unit}.",
        f"- Costs: {rep['cost_bps_per_side']} bps per side on traded weight.",
        "",
        "## Direction hit rate",
        f"- Pooled: {hr['hit_rate'] * 100:.1f}% correct vs {hr['base_rate'] * 100:.1f}% base rate "
        f"(always predicting up), n={hr['n']}." if hr["n"] else "- Pooled: no scored forecasts.",
    ]
    for sym, h in rep["hit_rate"]["per_symbol"].items():
        lines.append(f"- {sym}: {h['hit_rate'] * 100:.1f}% vs {h['base_rate'] * 100:.1f}% base, n={h['n']}.")
    lines += [
        "",
        "## Strategy",
        f"- Rule: {st['rule']}",
        f"- Cumulative return: {_pct(st['cumulative_return'])} over {st['periods']} periods.",
        f"- Sharpe (annualised, {HORIZON}-{unit} periods): {_num(st['sharpe'])}.",
        f"- Max drawdown: {_pct(st['max_drawdown']).lstrip('+')}." if st['max_drawdown'] is not None else "- Max drawdown: n/a.",
        f"- Average holdings: {_num(st['avg_holdings'])}; periods in cash: {st['cash_periods']}.",
        "",
        "## Benchmarks (same dates)",
    ]
    for key, b in rep["benchmarks"].items():
        lines.append(f"- {b['label']}: cumulative {_pct(b['cumulative_return'])}, "
                     f"Sharpe {_num(b['sharpe'])}, max drawdown {_pct(b['max_drawdown']).lstrip('+')}."
                     if b["cumulative_return"] is not None else f"- {key}: no data.")
    lines += ["", "## Threshold sensitivity"]
    for cell in rep["grid"]:
        lines.append(
            f"- R >= {cell['eligible_r'] * 100:.1f}%, agree >= {cell['agree'] * 100:.0f}%: "
            f"cumulative {_pct(cell['cumulative_return'])}, average holdings {_num(cell['avg_holdings'])}."
        )
    lines += ["", "## Warnings"]
    lines += [f"- {w}" for w in rep["warnings"]]
    lines.append("")
    return "\n".join(lines)


# ----- report run ------------------------------------------------------------------------

def run_report(universe: str, chunks: int, store: StateStore) -> int:
    missing: list[int] = []
    records: list[dict] = []
    for i in range(chunks):
        payload = store.read_json(_chunk_name(universe, i))
        if payload is None:
            missing.append(i)
            continue
        records.extend(payload.get("dates", []))
    if missing:
        print(f"{universe}: missing chunk(s) {missing} of {chunks}; no report written.")
        return 0

    seen: dict[str, dict] = {}
    for r in records:
        if r["date"] in seen:
            log.warning("duplicate decision date %s across chunks; keeping the first", r["date"])
            continue
        seen[r["date"]] = r
    report = build_report(universe, [seen[k] for k in sorted(seen)])

    base = f"backtest/{universe}"
    store.write_json(f"{base}/report.json", report)
    md_path = store.dir / base / "report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {store.dir / base / 'report.json'} and {md_path}")

    ok = store.commit_and_push(f"backtest {universe} report")
    return 0 if ok else 1


# ----- CLI -------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bot.backtest", description=__doc__.split("\n")[0])
    p.add_argument("--universe", choices=UNIVERSES, required=True)
    p.add_argument("--chunk", type=int, default=0, help="chunk index, 0-based")
    p.add_argument("--chunks", type=int, default=3, help="total number of chunks")
    p.add_argument("--report", action="store_true", help="aggregate existing chunks into a report")
    return p


def main(argv=None, client=None, forecaster=None, store=None) -> int:
    args = _parser().parse_args(argv)
    if args.chunks < 1 or not 0 <= args.chunk < args.chunks:
        print(f"invalid chunk {args.chunk} of {args.chunks}", file=sys.stderr)
        return 2
    store = store if store is not None else StateStore()
    if args.report:
        return run_report(args.universe, args.chunks, store)
    client = client if client is not None else AlpacaClient.from_env()
    forecaster = forecaster if forecaster is not None else KronosForecaster()
    return run_chunk(args.universe, args.chunk, args.chunks, client, forecaster, store)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
