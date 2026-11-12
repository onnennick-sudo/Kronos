"""Daily runner (after the close): rank the universe for the next session, score matured
predictions, snapshot equity, update metrics and email a report.

Usage: python -m bot.run_daily [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import alerts, config, metrics, runtime, strategy, timeutil
from bot.alpaca_client import AlpacaClient
from bot.forecast import KronosForecaster, seed_for
from bot.state import StateStore

log = logging.getLogger(__name__)

ET = timeutil.ET
MARKET = config.MARKET_SYMBOL
UNIVERSE = list(config.UNIVERSE)
HISTORY_DAYS = 800
EQUITY_FILE = "equity.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"
METRICS_FILE = "metrics.csv"
METRICS_COLUMNS = [
    "date", "equity", "spy_close", "ew_index",
    "bot_ret", "spy_ret", "ew_ret", "sharpe_bot", "sharpe_spy", "maxdd_bot", "maxdd_spy",
    "daily_hit", "daily_base", "daily_n", "hourly_hit", "hourly_base", "hourly_n",
    "slots_expected", "slots_completed", "slots_missed",
]


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Kronos daily forecast and report")
    parser.add_argument("--dry-run", action="store_true", help="label the run as dry_run (no orders exist here)")
    parser.add_argument("--force", action="store_true", help="rerun even if the target session already succeeded")
    return parser.parse_args(argv)


def main(argv=None, client=None, store=None, forecaster=None, now=None) -> int:
    args = _parse_args(argv)
    runtime.setup_logging()
    mode = runtime.bot_mode(args.dry_run)
    client = client if client is not None else AlpacaClient.from_env()
    store = store if store is not None else StateStore()
    forecaster = forecaster if forecaster is not None else KronosForecaster()
    now = now if now is not None else timeutil.now_et()
    state = {"key": now.date().isoformat()}  # replaced with the target session once known
    try:
        return _run(args, mode, client, store, forecaster, now, state)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, then exit 1
        log.exception("daily run failed")
        runtime.fail(store, "daily", state["key"], exc)
        return 1


def _run(args, mode, client, store, forecaster, now, state) -> int:
    today = now.date()
    calendar = runtime.calendar_window(client, today)
    sessions = sorted({c["date"] for c in calendar})
    closes = {c["date"]: c["close"] for c in calendar}
    last_final, target = _target_sessions(sessions, now)
    key = target.isoformat()
    state["key"] = key
    if store.has_ok("daily", key) and not args.force:
        log.info("daily run for %s already ok; nothing to do", key)
        return 0

    symbols = UNIVERSE + [MARKET]
    bars = _fetch_daily(client, symbols, last_final, now)
    stats, errors = _forecast_daily(forecaster, symbols, bars, calendar, last_final)
    spy = _spy_summary(stats, errors)
    ranking = strategy.build_ranking({s: v for s, v in stats.items() if s in UNIVERSE},
                                     _atrs(bars, UNIVERSE))
    degraded = _is_degraded(spy, errors)
    payload = {
        "session": target.isoformat(), "as_of": last_final.isoformat(), "spy": spy,
        "ranking": ranking, "candidates": strategy.candidates(ranking), "errors": errors,
        "degraded": degraded, "paths": config.DAILY_PATHS, "model": config.MODEL_NAME,
    }
    store.save_ranking(target, payload)
    if degraded:
        runtime.alert_once(store, f"degraded-daily-{key}", "Kronos daily DEGRADED",
                           _degraded_message(spy, errors), "high")

    _write_daily_predictions(store, ranking, stats, last_final, target)
    scored_daily = _score_daily(store, sessions, bars, last_final)
    scored_hourly = _score_hourly(store, client, calendar, last_final, now)
    snap = _snapshot_equity(store, client, bars, last_final)
    rows = store.read(EQUITY_FILE)
    perf = _since_start(rows)
    outcomes = store.read(OUTCOMES_FILE)
    hits = {kind: _hit_stats(outcomes, kind) for kind in ("daily", "hourly")}
    health = _run_health(store, closes, last_final)
    _write_metrics_row(store, last_final, snap, perf, hits, health)

    ctx = {
        "mode": mode, "last_final": last_final, "target": target,
        "equity": snap["equity"], "last_equity": snap["last_equity"],
        "day_ret": _ratio_return(snap["equity"], snap["last_equity"]),
        "spy_day": _day_return(bars.get(MARKET), last_final),
        "trades": [r for r in store.read(runtime.ORDERS_FILE) if r.get("date") == last_final.isoformat()],
        "positions": client.get_positions(),
        "spy_regime": spy["regime"], "candidates": payload["candidates"],
        "daily": hits["daily"], "hourly": hits["hourly"], "latest": outcomes[-3:],
        "health": health, "halt": (store.halt_today(last_final) or {}).get("reason"),
        "n_errors": len(errors), "degraded": degraded, "perf": perf,
    }
    subject = (f"[Kronos {mode}] {last_final}: equity ${snap['equity']:,.2f} "
               f"(day {_pct(ctx['day_ret'])}) vs SPY {_pct(ctx['spy_day'])}")
    sent = alerts.send_email(subject, _render_email(ctx))

    status = "degraded" if degraded else "ok"
    store.record_run(
        "daily", key, status, as_of=last_final.isoformat(), mode=mode,
        n_candidates=len(payload["candidates"]), n_errors=len(errors),
        scored_daily=scored_daily, scored_hourly=scored_hourly, email_sent=bool(sent),
    )
    store.commit_and_push(f"daily {key} ({status})")
    return 0


# ----- sessions and data ------------------------------------------------------

def _target_sessions(sessions: list[date], now: datetime) -> tuple[date, date]:
    """(last_final, target). Today's bar counts as final only at or after DAILY_READY_ET."""
    today = now.date()
    ready = dtime.fromisoformat(config.DAILY_READY_ET)
    if today in sessions and now.time() >= ready:
        last_final = today
    else:
        before = [d for d in sessions if d < today]
        if not before:
            raise RuntimeError("calendar has no session before today")
        last_final = max(before)
    after = [d for d in sessions if d > last_final]
    if not after:
        raise RuntimeError(f"calendar has no session after {last_final}")
    return last_final, min(after)


def _fetch_daily(client, symbols, last_final, now) -> dict[str, pd.DataFrame]:
    start = datetime.combine(last_final - timedelta(days=HISTORY_DAYS), dtime(0), tzinfo=ET)
    end = now.astimezone(timezone.utc) - timedelta(minutes=config.DATA_DELAY_MINUTES)
    raw = client.get_bars(list(symbols), "1Day", start, end)
    cutoff = pd.Timestamp(last_final)
    out: dict[str, pd.DataFrame] = {}
    for sym, df in raw.items():
        daily = timeutil.daily_df_for_model(df)
        out[sym] = daily[daily["timestamps"] <= cutoff].reset_index(drop=True)
    return out


def _freshness_error(df, last_final) -> str | None:
    if df is None or df.empty:
        return "no bars"
    if df["timestamps"].iloc[-1].date() != last_final:
        return "stale"
    return None


def _forecast_daily(forecaster, symbols, bars, calendar, last_final):
    """Forecast every fresh symbol. Returns (stats, errors); errors maps symbol -> reason."""
    series, errors = {}, {}
    for sym in symbols:
        err = _freshness_error(bars.get(sym), last_final)
        if err:
            errors[sym] = err
        else:
            series[sym] = bars[sym]
    if not series:
        return {}, errors
    future = timeutil.future_daily_stamps(calendar, last_final, config.DAILY_PRED_LEN)
    raw = forecaster.forecast(series, {s: future for s in series}, config.DAILY_PRED_LEN,
                              config.DAILY_PATHS, config.DAILY_LOOKBACK, seed_for(last_final, "daily"))
    stats = {sym: raw[sym] for sym in series if sym in raw}
    for sym, s in stats.items():
        if "error" in s:
            errors[sym] = s["error"]
    return stats, errors


def _atrs(bars, symbols) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym in symbols:
        df = bars.get(sym)
        if df is None or df.empty:
            continue
        try:
            out[sym] = strategy.atr(df)
        except ValueError as exc:
            log.warning("no ATR for %s: %s", sym, exc)
    return out


def _spy_summary(stats, errors) -> dict:
    s = stats.get(MARKET)
    if MARKET in errors or s is None or "error" in s:
        return {"regime": "UNKNOWN", "error": errors.get(MARKET, "no forecast")}
    return {
        "r_med": float(s["r_med"]), "agree": float(s["agree"]), "last_close": float(s["last_close"]),
        "regime": strategy.classify_daily(float(s["r_med"]), float(s["agree"])),
    }


def _is_degraded(spy, errors) -> bool:
    failed = sum(1 for s in UNIVERSE if s in errors)
    return spy["regime"] == "UNKNOWN" or failed > config.MAX_FAILED_TICKERS


def _degraded_message(spy, errors) -> str:
    failed = sorted(s for s in UNIVERSE if s in errors)
    return (f"SPY regime {spy['regime']}. Failed universe tickers: {len(failed)}"
            f" ({', '.join(failed[:10]) or 'none'}).")


# ----- predictions and outcomes ----------------------------------------------

def _key(rec: dict) -> tuple:
    return (rec.get("kind"), rec.get("as_of") or rec.get("bar_label"), rec.get("symbol"))


def _write_daily_predictions(store, ranking, stats, last_final, target) -> None:
    existing = {_key(r) for r in store.read(runtime.PREDICTIONS_FILE)}
    for row in ranking:
        sym = row["symbol"]
        rec = {"kind": "daily", "as_of": last_final.isoformat(), "target_session": target.isoformat(),
               "symbol": sym, "r_med": float(row["r_med"]), "agree": float(row["agree"]),
               "last_close": float(stats[sym]["last_close"]), "horizon": config.DAILY_PRED_LEN}
        if _key(rec) not in existing:
            store.append(runtime.PREDICTIONS_FILE, rec)


def _nth_session_after(sessions: list[date], as_of: date, n: int) -> date | None:
    """The n-th session strictly after as_of, or None when the calendar cannot establish it."""
    if not sessions or as_of < sessions[0]:
        return None
    after = [d for d in sessions if d > as_of]
    return after[n - 1] if len(after) >= n else None


def _close_at(df, when) -> float | None:
    """Close of the bar stamped `when` (a date for daily bars, a naive datetime for hourly)."""
    if df is None or df.empty:
        return None
    rows = df[df["timestamps"] == pd.Timestamp(when)]
    return float(rows["close"].iloc[-1]) if not rows.empty else None


def _append_outcome(store, pred: dict, close: float, horizon_end) -> None:
    pred_r = float(pred["r_med"])
    actual_r = close / float(pred["last_close"]) - 1.0
    rec = {"kind": pred["kind"], "symbol": pred["symbol"], "pred_r": pred_r, "actual_r": actual_r,
           "hit": (pred_r > 0) == (actual_r > 0), "horizon_end": horizon_end.isoformat()}
    if pred["kind"] == "daily":
        rec["as_of"] = pred["as_of"]
    else:
        rec["bar_label"] = pred["bar_label"]
    store.append(OUTCOMES_FILE, rec)


def _score_daily(store, sessions, bars, last_final) -> int:
    done = {_key(r) for r in store.read(OUTCOMES_FILE)}
    written = 0
    for pred in store.read(runtime.PREDICTIONS_FILE):
        if pred.get("kind") != "daily" or _key(pred) in done:
            continue
        end = _nth_session_after(sessions, date.fromisoformat(pred["as_of"]), int(pred["horizon"]))
        if end is None or end > last_final:
            continue
        close = _close_at(bars.get(pred["symbol"]), end)
        if close is None:
            continue
        _append_outcome(store, pred, close, end)
        written += 1
    return written


def _hourly_target(calendar, sessions, pred, last_final) -> datetime | None:
    """The horizon bar for an hourly prediction, if its session is final and the calendar covers it."""
    label = datetime.fromisoformat(pred["bar_label"])
    if not sessions or label.date() < sessions[0] or label.date() > last_final:
        return None
    try:
        stamps = timeutil.future_hourly_stamps(calendar, label, int(pred["horizon"]))
    except ValueError:
        return None
    target = stamps.iloc[-1].to_pydatetime()
    return target if target.date() <= last_final else None


def _score_hourly(store, client, calendar, last_final, now) -> int:
    sessions = sorted({c["date"] for c in calendar})
    closes = {c["date"]: c["close"] for c in calendar}
    done = {_key(r) for r in store.read(OUTCOMES_FILE)}
    due = []
    for pred in store.read(runtime.PREDICTIONS_FILE):
        if pred.get("kind") != "hourly" or _key(pred) in done:
            continue
        target = _hourly_target(calendar, sessions, pred, last_final)
        if target is not None:
            due.append((pred, target))
    if not due:
        return 0
    start = min(datetime.fromisoformat(p["bar_label"]) for p, _ in due).replace(tzinfo=ET)
    end = now.astimezone(timezone.utc) - timedelta(minutes=config.DATA_DELAY_MINUTES)
    symbols = sorted({p["symbol"] for p, _ in due})
    raw = client.get_bars(symbols, "1Hour", start, end)
    frames = {s: timeutil.filter_regular_hourly(df, set(closes), closes) for s, df in raw.items()}
    written = 0
    for pred, target in due:
        close = _close_at(frames.get(pred["symbol"]), target)
        if close is not None:
            _append_outcome(store, pred, close, target)
            written += 1
    return written


def _hit_stats(outcomes, kind: str) -> tuple[float, float, int]:
    pairs = [(float(o["pred_r"]), float(o["actual_r"])) for o in outcomes if o.get("kind") == kind]
    return metrics.hit_rate(pairs)


# ----- equity, performance and health ----------------------------------------

def _day_return(df, day: date) -> float | None:
    """Close-to-close return ending on `day`, or None when the last two bars are not available."""
    if df is None or df.empty:
        return None
    upto = df[df["timestamps"] <= pd.Timestamp(day)]
    if len(upto) < 2 or upto["timestamps"].iloc[-1] != pd.Timestamp(day):
        return None
    return float(upto["close"].iloc[-1] / upto["close"].iloc[-2] - 1.0)


def _ratio_return(now_value, base_value) -> float | None:
    return None if not base_value else float(now_value) / float(base_value) - 1.0


def _ew_index(bars, base_day: date, last_final: date) -> float | None:
    """Mean over the universe of close(last_final) / close(base_day)."""
    ratios = []
    for sym in UNIVERSE:
        c0 = _close_at(bars.get(sym), base_day)
        c1 = _close_at(bars.get(sym), last_final)
        if c0 and c1:
            ratios.append(c1 / c0)
    return float(np.mean(ratios)) if ratios else None


def _snapshot_equity(store, client, bars, last_final) -> dict:
    """Write one equity.jsonl record per session (idempotent). Returns the record for last_final."""
    day = last_final.isoformat()
    rows = store.read(EQUITY_FILE)
    existing = next((r for r in rows if r.get("date") == day), None)
    if existing is not None:
        return existing
    acct = client.get_account()
    first_day = date.fromisoformat(rows[0]["date"]) if rows else None
    ew = 1.0 if first_day is None else _ew_index(bars, first_day, last_final)
    rec = {
        "date": day, "equity": float(acct["equity"]), "last_equity": float(acct["last_equity"]),
        "spy_close": _close_at(bars.get(MARKET), last_final), "ew_index": ew,
    }
    store.append(EQUITY_FILE, rec)
    return rec


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def _total_return(s: pd.Series) -> float | None:
    if len(s) < 2 or s.iloc[0] == 0:
        return None
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def _since_start(rows) -> dict:
    eq = pd.Series([float(r["equity"]) for r in rows], dtype=float)
    spy = pd.Series([_num(r.get("spy_close")) for r in rows], dtype=float).dropna()
    ew = pd.Series([_num(r.get("ew_index")) for r in rows], dtype=float).dropna()
    return {
        "days": len(rows),
        "bot_ret": _total_return(eq), "spy_ret": _total_return(spy), "ew_ret": _total_return(ew),
        "sharpe_bot": metrics.sharpe(metrics.daily_returns(eq)),
        "sharpe_spy": metrics.sharpe(metrics.daily_returns(spy)),
        "maxdd_bot": metrics.max_drawdown(eq), "maxdd_spy": metrics.max_drawdown(spy),
    }


def _expected_slots(day: date, closes: dict[date, dtime]) -> list[str]:
    close = closes.get(day)
    if close is None:
        return []
    return [f"{day.isoformat()}-{h - 1:02d}" for h in range(10, 16) if dtime(h, 20) < close]


def _run_health(store, closes, last_final) -> dict:
    expected = _expected_slots(last_final, closes)
    missed = [k for k in expected if not store.has_ok("trading", k)]
    return {"expected": len(expected), "completed": len(expected) - len(missed), "missed": missed}


def _write_metrics_row(store, last_final, snap, perf, hits, health) -> None:
    path = store.dir / METRICS_FILE
    day = last_final.isoformat()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            if any(row.get("date") == day for row in csv.DictReader(f)):
                return
    row = {
        "date": day, "equity": snap["equity"], "spy_close": snap.get("spy_close"), "ew_index": snap.get("ew_index"),
        "bot_ret": perf["bot_ret"], "spy_ret": perf["spy_ret"], "ew_ret": perf["ew_ret"],
        "sharpe_bot": perf["sharpe_bot"], "sharpe_spy": perf["sharpe_spy"],
        "maxdd_bot": perf["maxdd_bot"], "maxdd_spy": perf["maxdd_spy"],
        "daily_hit": hits["daily"][0], "daily_base": hits["daily"][1], "daily_n": hits["daily"][2],
        "hourly_hit": hits["hourly"][0], "hourly_base": hits["hourly"][1], "hourly_n": hits["hourly"][2],
        "slots_expected": health["expected"], "slots_completed": health["completed"],
        "slots_missed": len(health["missed"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ----- email ------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _pct_plain(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _num_plain(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _hit_line(label: str, stats) -> str:
    hit, base, n = stats
    if n == 0:
        return f"  {label}: no scored predictions yet"
    return f"  {label}: hit {hit:.1%} vs base rate {base:.1%} (n={n})"


def _trade_lines(trades) -> list[str]:
    return [f"  {t.get('action')} {t.get('symbol')} qty {t.get('qty')} [{t.get('status')}] "
            f"{t.get('reason') or ''}{' ' + str(t['error']) if t.get('error') else ''}".rstrip()
            for t in trades]


def _position_lines(positions) -> list[str]:
    return [f"  {p['symbol']} qty {p['qty']}  avg {p['avg_entry_price']:.2f}  "
            f"now {p['current_price']:.2f}  value ${p['market_value']:,.2f}" for p in positions]


def _latest_lines(outcomes) -> list[str]:
    lines = []
    for o in outcomes:
        when = o.get("as_of") or o.get("bar_label")
        verdict = "HIT" if o.get("hit") else "miss"
        lines.append(f"  {o['kind']} {when} {o['symbol']}: predicted {o['pred_r']:+.2%}, "
                     f"actual {o['actual_r']:+.2%} [{verdict}]")
    return lines


def _perf_lines(perf) -> list[str]:
    return [
        f"  {perf['days']} sessions",
        f"  Return: bot {_pct_plain(perf['bot_ret'])}, SPY {_pct_plain(perf['spy_ret'])}, "
        f"EW {_pct_plain(perf['ew_ret'])}",
        f"  Sharpe: bot {_num_plain(perf['sharpe_bot'])}, SPY {_num_plain(perf['sharpe_spy'])}",
        f"  Max drawdown: bot -{perf['maxdd_bot'] * 100:.2f}%, SPY -{perf['maxdd_spy'] * 100:.2f}%",
    ]


def _render_email(ctx: dict) -> str:
    health = ctx["health"]
    lines: list[str] = []
    if ctx["mode"] == runtime.DRY_RUN:
        lines += ["DRY RUN — no orders placed", ""]
    lines += [
        f"TODAY ({ctx['last_final']})",
        f"  Equity ${ctx['equity']:,.2f} (last ${ctx['last_equity']:,.2f}), day {_pct(ctx['day_ret'])}; "
        f"SPY day {_pct(ctx['spy_day'])}",
        "",
        "TRADES",
        *(_trade_lines(ctx["trades"]) or ["  none"]),
        "",
        "OPEN POSITIONS",
        *(_position_lines(ctx["positions"]) or ["  none"]),
        "",
        f"NEXT SESSION ({ctx['target']})",
        f"  SPY regime: {ctx['spy_regime']}",
        f"  Candidates: {', '.join(ctx['candidates']) or 'none'}",
        "",
        "KRONOS VS ACTUAL",
        _hit_line("Daily", ctx["daily"]),
        _hit_line("Hourly", ctx["hourly"]),
        *(_latest_lines(ctx["latest"]) or ["  no scored outcomes yet"]),
        "",
        "RUN HEALTH",
        f"  Trading slots: expected {health['expected']}, completed {health['completed']}, "
        f"missed {', '.join(health['missed']) or 'none'}",
        f"  Halt today: {ctx['halt'] or 'no'}",
        f"  Ticker errors: {ctx['n_errors']} (degraded: {'yes' if ctx['degraded'] else 'no'})",
        "",
        "SINCE START",
        *_perf_lines(ctx["perf"]),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
