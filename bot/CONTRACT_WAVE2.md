# Wave 2 contract: runners, workflows, backtest (binding)

Read CONTRACT.md + config.py + the existing modules first. Reuse them; do not duplicate logic. Do not modify wave-1 modules except where this file says so.

## Shared: bot/runtime.py (owned by runner builder)
- `setup_logging()`; `bot_mode(cli_dry_run: bool) -> "dry_run"|"paper"` (CLI flag forces dry_run; else env BOT_MODE; anything not exactly "paper" => "dry_run").
- `flatten_orders(open_orders) -> list[dict]` top-level orders + every nested `legs` entry, deduped by id.
- `stop_orders_by_symbol(flat) -> dict[sym, {"order_id","stop_price"}]` sell orders with type in (stop, stop_limit, trailing_stop excluded) and status not in (filled, canceled, expired, rejected).
- `orders_for_symbol(flat, sym)`.
- `calendar_window(client, today, back_days=40, fwd_days=40)`.
- `alert_once(store, key, title, msg, priority)` dedupe via state `alerts_sent.jsonl` (key unique per day/slot).
- `fail(store, kind, key, exc)`: urgent push, record_run(kind,key,"error",error=str(exc)), commit_and_push.
- `AlpacaClient.get_latest_prices(symbols) -> dict[sym,float]` ADD this method to alpaca_client.py: GET {data_url}/v2/stocks/trades/latest?symbols=..&feed=iex (free real-time IEX) -> price "p". Add a test in test_alpaca_client.py.

## bot/run_daily.py  (`python -m bot.run_daily [--dry-run] [--force]`)
1. client=AlpacaClient.from_env(); store=StateStore(); now=now_et(); calendar=calendar_window.
2. last_final = today if (today is a session and now >= DAILY_READY_ET) else last session date < today. target_session = first session date > last_final.
3. If store.has_ok("daily", str(target_session)) and not --force: log + exit 0.
4. Daily bars UNIVERSE + [MARKET_SYMBOL]: start = last_final - 800 days, end = now_utc - DATA_DELAY_MINUTES. Keep rows with ET date <= last_final. Per-ticker stale if last row date != last_final -> error for that ticker.
5. daily_df_for_model; future = future_daily_stamps(calendar, last_final, DAILY_PRED_LEN) (same for all).
6. stats = KronosForecaster().forecast(..., DAILY_PRED_LEN, DAILY_PATHS, DAILY_LOOKBACK, seed_for(last_final,"daily")).
7. atrs (skip errors). ranking = build_ranking(stats minus SPY, atrs). spy = {"r_med","agree","regime"} or {"regime":"UNKNOWN","error"}.
8. degraded = SPY error or (#errors in universe > MAX_FAILED_TICKERS) -> push high alert.
9. store.save_ranking(target_session, {session, as_of: last_final, spy, ranking, candidates: candidates(ranking), errors, degraded, paths, model}).
10. predictions.jsonl: one record per symbol {kind:"daily", as_of, target_session, symbol, r_med, agree, last_close, horizon: DAILY_PRED_LEN}.
11. outcomes.jsonl: for each earlier daily prediction whose horizon end (as_of + DAILY_PRED_LEN sessions, via calendar) <= last_final and not yet scored: actual_r = close(horizon end)/last_close - 1 from the fetched bars; record {kind:"daily", as_of, symbol, pred_r, actual_r, hit: sign(pred)==sign(actual)}. Hourly predictions (kind "hourly", bar_label, horizon 6) from sessions <= last_final not yet scored: fetch hourly bars for those symbols (one call), score the same way at the 6th regular bar after bar_label; skip if not available yet.
12. Account snapshot: equity.jsonl {date:last_final, equity, last_equity, spy_close, ew_index} (ew_index = mean over universe of close/close_at_first_equity_date, computed from bars; first record sets base 1.0). Skip if a record for that date exists.
13. Metrics (bot.metrics) over equity.jsonl since first record: bot return, SPY return, EW return, Sharpe bot & SPY, max DD; daily+hourly hit rate with base rate from outcomes.jsonl. Append metrics.csv row (create header if missing).
14. Run health for session last_final: expected trading slot keys = for h in 10..15 where time(h,20) < that day's close: f"{last_final}-{h-1:02d}"; missed = those without has_ok("trading", key). Record in metrics.
15. Email (alerts.send_email) subject `[Kronos {mode}] {last_final}: equity $X (day ±Y%) vs SPY ±Z%`; plain-text body sections: Today, Trades (orders.jsonl records for last_final), Open positions (client), Next session candidates + SPY regime, Kronos vs actual (hit rates + latest scored), Run health (expected/completed/missed, halts, errors), Since start (returns, Sharpe, max DD vs SPY & EW). Prefix body "DRY RUN — no orders placed" when dry_run.
16. record_run("daily", str(target_session), "ok" or "degraded", ...); commit_and_push. Any exception -> runtime.fail(...) and exit 1.

## bot/run_trading.py  (`python -m bot.run_trading [--dry-run]`)
1. mode=bot_mode; client; store; now; slot=slot_key(now); today=now.date(). If has_ok("trading", slot): exit 0.
2. account=get_account(); clock=get_clock(); if not clock["is_open"]: log, exit 0 (no record).
3. payload = store.load_ranking(today). If None: payload=latest_ranking() (may be None); entries_blocked="no_ranking_for_today"; alert_once(key=f"noranking-{today}", high).
   ranking = payload["ranking"] if payload else []. spy_regime = payload["spy"]["regime"] if payload else "UNKNOWN". If spy_regime == "UNKNOWN" or entries_blocked or payload.get("degraded"): pass spy_regime="BEARISH" into decide (blocks entries) and note why.
4. Halt: halted = bool(store.halt_today(today)) or is_halted(equity,last_equity). If newly halted: set_halt(today, reason), cancel open BUY orders (paper mode only), alert_once urgent.
5. positions, flat=flatten_orders(get_open_orders()), stop_orders=stop_orders_by_symbol(flat).
6. stopped_out_today: prev = store.read_json("positions.json", {}) (written by last run, {date, symbols}); if prev date == today: symbols in prev but not held now and not exited by us (orders.jsonl today with side sell and reason in exit reasons) -> append to store "stopped_out.jsonl" {date, symbol}. stopped_out_today = all symbols in stopped_out.jsonl for today.
7. symbols = candidates(ranking) ∪ held symbols. Hourly bars start=now-140 days, end=now_utc-DATA_DELAY_MINUTES; filter_regular_hourly with calendar. Per-ticker: last label must equal datetime(today, now.hour-1) else error "stale". future_hourly_stamps(calendar, last_label, HOURLY_PRED_LEN).
8. hourly = forecaster.forecast(..., HOURLY_PRED_LEN, HOURLY_PATHS, HOURLY_LOOKBACK, seed_for(now,"hourly")). If forecaster load fails => system-wide: entries blocked, exits from daily reasons only (pass hourly={} -> decide treats as hourly_error), urgent alert.
9. last_prices = get_latest_prices(symbols) (fallback: last hourly close; then position current_price).
10. plan = decide(ranking, spy_regime, hourly, positions, stop_orders, equity, halted, stopped_out_today, last_prices).
11. Execute only if mode == "paper":
   - Exit: cancel every open order for sym (from flat); re-read open orders up to 3 times (sleep 1s) until none for sym; submit_market_sell(qty, client_order_id=f"kx-{sym}-{slot}"). If the sell fails after cancel: submit_stop_sell at the previous stop_price (if known) and alert urgent. If cancel can't be confirmed: alert, skip sell.
   - Entry: submit_bracket_buy(sym, qty, stop, target, client_order_id=f"ke-{sym}-{slot}").
   - Trail: replace_order(order_id, new_stop); failure -> alert high, keep old.
   Each order result (or dry-run intent) appended to orders.jsonl {date, slot, mode, action, symbol, qty, reason, stop, target, status, error}.
12. predictions.jsonl hourly records {kind:"hourly", date, slot, bar_label, symbol, r_med, agree, last_close, horizon}. store.write_json("positions.json", {date: today, symbols: held after plan (held - exits + entries if paper)}).
13. record_run("trading", slot, "ok", mode, n_exits, n_entries, n_trails, skipped, notes); commit_and_push. Order failures don't make the run "error" but trigger alerts. Unhandled exception -> runtime.fail, exit 1.

## bot/run_watchdog.py  (`python -m bot.run_watchdog`)
- clock + calendar via client. If market open: last ok trading record ts today; gap = now - max(last_ok_ts, today 09:30 ET); if gap > WATCHDOG_TRADING_GAP_MIN min -> alert_once(key=f"wd-trading-{today}-{now.hour}", urgent).
- If today is a session and now >= DAILY_DEADLINE_ET and no ok/degraded daily record whose as_of == today -> alert_once(key=f"wd-daily-{today}", high). (daily record must store as_of.)
- commit_and_push only if an alert was sent.

## Tests (bot/tests/test_runners.py)
Fakes for client (in-memory account/orders/positions/bars), forecaster (returns given stats), StateStore on tmp_path with commit_and_push monkeypatched. Cover: daily happy path writes ranking/predictions/equity/metrics/email & idempotent second run; daily target-session logic (after close vs morning manual); trading dry_run places no orders but logs intents; paper places bracket buy; exit sequence cancel→confirm→sell, and sell failure re-places stop; halt triggers set_halt + cancel buys + no entries; missing ranking blocks entries but allows exits; stale hourly ticker isolated; slot idempotency; stopped_out detection; watchdog alerts + dedupe.

## Workflows (.github/workflows/) — owned by workflow builder
Common job skeleton: runs-on ubuntu-latest; permissions contents: write (watchdog too, for dedupe commits); concurrency {group: kronos-state, cancel-in-progress: false}; env BOT_ENABLED: ${{ vars.BOT_ENABLED }}, ALPACA_API_KEY_ID/SECRET from secrets, GMAIL_ADDRESS, GMAIL_APP_PASSWORD, ALERT_TO_EMAIL, NTFY_TOPIC secrets, BOT_STATE_DIR: state-repo, PYTHONPATH: ${{ github.workspace }}.
Steps: checkout (ref master by default = default branch) → gate with system python3 (`python3 -m bot.gate <mode>`, id gate) → all later steps `if: steps.gate.outputs.proceed == 'true'` → bootstrap bot-state (if `git ls-remote --exit-code --heads origin bot-state` fails: create orphan branch with README + state/.keep, push) → checkout bot-state into state-repo (actions/checkout path: state-repo, ref: bot-state) → setup-python 3.11 with pip cache (cache-dependency-path bot/requirements.txt) → cache ~/.cache/huggingface key hf-${{ hashFiles('bot/config.py') }} → `pip install torch --index-url https://download.pytorch.org/whl/cpu` + `pip install -r bot/requirements.txt` → run.
- kronos-trading.yml name "Kronos Trading": cron '20 14-20 * * 1-5' and '50 14-20 * * 1-5'; workflow_dispatch input dry_run (boolean, default true); BOT_MODE = dispatch && dry_run ? 'dry_run' : vars.BOT_MODE; gate mode trading (manual dispatch uses gate mode manual); timeout 20.
- kronos-daily.yml "Kronos Daily": cron '20 20-22 * * 1-5'; dispatch inputs dry_run (default true), force (default false); gate daily (manual for dispatch); timeout 45.
- kronos-watchdog.yml "Kronos Watchdog": cron '15 13-22 * * 1-5' and '45 23 * * 1-5'; no torch install (pip install -r bot/requirements.txt only); gate watchdog; timeout 5. If watchdog itself fails, GitHub emails.
- kronos-backtest.yml "Kronos Backtest": dispatch only; inputs universe (choice stocks|etfs|crypto), chunk (number, default 0), chunks (number, default 3); no gate except BOT_ENABLED not required; timeout 350; results committed to bot-state under backtest/.
- kronos-tests.yml "Kronos Tests": on push (paths bot/**) and pull_request; installs CPU torch? NO — tests are offline; install bot/requirements.txt + pytest only; run `python -m pytest bot/tests -q`.
- bot/requirements.txt: pandas, numpy, requests, einops==0.8.1, huggingface_hub==0.33.1, safetensors==0.6.2, pytest (no torch line).

## bot/backtest.py (`python -m bot.backtest --universe stocks|etfs|crypto --chunk i --chunks n`) — owned by backtest builder
- Uses alpaca client data (stocks get_bars SIP; crypto get_crypto_bars), KronosForecaster, strategy.classify_daily/build_ranking, metrics. Daily only. No lookahead: at each decision date use only bars <= date.
- stocks: last ~1 year, decision every 5 sessions (~50 points), DAILY_PATHS=5 override, rank universe, hold top-3 eligible equal-weight for 5 sessions; ETFs/crypto: ~2 years, every 5 sessions, paths 10, hold all BULLISH names equal-weight. Costs COST_BPS_PER_SIDE on turnover.
- Chunking: decision dates split into `chunks` contiguous parts; chunk i processes its part and writes `backtest/{universe}/chunk_{i}.json` (per date: per symbol r_med, agree, forward 5-session return) to BOT_STATE_DIR state; when all chunks exist, `--report` aggregates: hit rate vs base rate, threshold sensitivity (eligible R in {0.5%,1%,1.5%} x agree {0.55,0.6,0.65}), strategy vs SPY cumulative return, max DD, Sharpe → backtest/{universe}/report.md + report.json. Survivorship-bias warning in stocks report.
- Tests (bot/tests/test_backtest.py) with fake client/forecaster: no-lookahead (forecaster only sees bars <= decision date), chunk split, aggregation math.
