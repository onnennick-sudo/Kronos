# Bot module contract (binding for all builders)

Repo root: /home/user/Kronos. Package: `bot/`. Run from repo root with `PYTHONPATH=.`.
Kronos upstream (`model/`) is imported as `from model import Kronos, KronosTokenizer, KronosPredictor` and must NOT be modified.
All tunables live in `bot/config.py` (import as `from bot import config`). Never hardcode numbers that exist there.
Python 3.11+. Deps: pandas, numpy, requests, torch (CPU), einops, huggingface_hub, safetensors, pytest.
`bot/__init__.py` stays empty (gate.py must import without pandas/torch).
Timezone: `ET = zoneinfo.ZoneInfo("America/New_York")`. Model inputs use tz-NAIVE ET timestamps.
Tests: `bot/tests/test_<module>.py`, pytest, fully offline (mock HTTP with a fake session / monkeypatch; mock Kronos with a fake predictor). Run: `cd /home/user/Kronos && python -m pytest bot/tests -q`.

## bot/alpaca_client.py
- `class AlpacaError(Exception)`, `class PaperGuardError(AlpacaError)`.
- `AlpacaClient(key_id, secret_key, base_url=config.PAPER_BASE_URL, data_url=config.DATA_BASE_URL, session=None, max_retries=3, backoff=1.0)`
  - Constructor raises PaperGuardError if `base_url != config.PAPER_BASE_URL` (exact match) BEFORE any request.
  - `from_env()` classmethod: reads ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY.
  - Retries on 429/5xx/timeouts/connection errors with exponential backoff; raises AlpacaError after.
  - `get_account() -> dict` (floats for equity, last_equity, cash, buying_power)
  - `get_clock() -> dict` (is_open: bool, timestamp/next_open/next_close: aware datetimes)
  - `get_calendar(start: date, end: date) -> list[dict]` each `{"date": date, "open": time, "close": time}` (ET)
  - `get_positions() -> list[dict]` each `{"symbol","qty":int,"avg_entry_price":float,"current_price":float,"market_value":float}`
  - `get_open_orders() -> list[dict]` (raw Alpaca dicts, `status=open&nested=true`, includes bracket `legs`)
  - `submit_bracket_buy(symbol, qty:int, stop_price:float, take_profit_price:float, client_order_id:str) -> dict` market, time_in_force="gtc", order_class="bracket", prices rounded to cents
  - `submit_market_sell(symbol, qty:int, client_order_id:str) -> dict` (time_in_force="day")
  - `submit_stop_sell(symbol, qty:int, stop_price:float, client_order_id:str) -> dict` (gtc)
  - `cancel_order(order_id:str) -> None` (204/200 ok; 404/422 already gone -> no error)
  - `replace_order(order_id:str, stop_price:float) -> dict` (PATCH /v2/orders/{id})
  - `get_bars(symbols:list[str], timeframe:"1Day"|"1Hour", start:datetime, end:datetime, feed=config.DATA_FEED) -> dict[str, pd.DataFrame]`
    GET {data_url}/v2/stocks/bars, adjustment=all, limit=10000, follows next_page_token. DataFrame columns: `timestamps` (tz-aware UTC), open, high, low, close, volume; sorted, deduped. Symbols with no bars are omitted.
  - `get_crypto_bars(symbols, timeframe, start, end) -> dict[str, pd.DataFrame]` via /v1beta3/crypto/us/bars (backtest only; no auth needed but send headers anyway).

## bot/timeutil.py
- `ET`, `now_et() -> datetime`
- `to_naive_et(ts_series_utc) -> pd.Series` (tz-aware UTC -> naive ET)
- `filter_regular_hourly(df, session_dates:set[date], closes:dict[date,time]) -> df` keep hourly bars whose ET label date is a session and label hour in config.REGULAR_HOUR_LABELS and label < that day's close.
- `daily_df_for_model(df) -> df` daily bars with `timestamps` naive ET date at 00:00.
- `future_daily_stamps(calendar:list[dict], after: date, n:int) -> pd.Series` next n session dates strictly after `after`, naive 00:00.
- `future_hourly_stamps(calendar:list[dict], last_label: datetime(naive ET), n:int) -> pd.Series` next n clock-hour labels after last_label within sessions: for each session, hours h in REGULAR_HOUR_LABELS where datetime(date, h:00) < datetime(date, close) (09:00 label covers 09:30-10:00).
- `slot_key(now_et) -> str` "YYYY-MM-DD-HH" where HH = decision-bar label hour = now_et.hour - 1 (e.g. 11:20 run -> "...-10").

## bot/forecast.py
- `path_stats(final_closes: np.ndarray, last_close: float) -> tuple[float, float]` returns (r_med, agree). r = final/last_close - 1; r_med = median; agree = share of paths with sign == sign(r_med) (r_med==0 -> share of r==0... treat as 0.0 agreement).
- `class KronosForecaster(predictor=None)`: if predictor None, lazily loads Kronos-small + tokenizer (config names/revision), `torch.set_num_threads(config.TORCH_THREADS)`, device cpu.
  - `forecast(series: dict[str, pd.DataFrame], future: dict[str, pd.Series], pred_len:int, n_paths:int, lookback:int, seed:int) -> dict[str, dict]`
    For each symbol: last `lookback` rows (needs >= lookback rows else symbol reported in result as `{"error": "..."}`), duplicates the series n_paths times, calls `predictor.predict_batch(..., sample_count=1, T/top_k/top_p from config, verbose=False)` in chunks of config.BATCH_SIZE (all sequences in one call must share lookback & pred_len). Seeds torch/numpy/random with `seed` once before inference. Returns `{sym: {"r_med", "agree", "n", "last_close", "final_closes": list}}`. Per-symbol exceptions are isolated into `{"error": str}`.
  - `seed_for(dt, kind) -> int` daily: YYYYMMDD; hourly: YYYYMMDDHH.

## bot/strategy.py (pure, no I/O)
Dataclasses: `Entry(symbol, qty, ref_price, stop, target, rank, reason)`, `Exit(symbol, qty, reason)`, `Trail(symbol, order_id, old_stop, new_stop)`, `Plan(exits, entries, trails, skipped: list[tuple[str,str]], notes: list[str])`.
- `classify_daily(r_med, agree) -> "BULLISH"|"BEARISH"|"NEUTRAL"`
- `classify_hourly(r_med, agree) -> "CONFIRM_LONG"|"STRONG_NEG"|"NONE"`
- `atr(daily_df, period=config.ATR_PERIOD) -> float` (Wilder or simple mean TR; document)
- `build_ranking(stats: dict[sym, dict(r_med, agree)], atrs: dict[sym,float]) -> list[dict]` all symbols without error, sorted by r_med desc, each `{symbol, rank (1=best), r_med, agree, eligible, regime, atr}`; eligible = r_med>=DAILY_ELIGIBLE_R and agree>=DAILY_AGREE.
- `candidates(ranking, n=config.N_CANDIDATES) -> list[str]` top-n by rank that are eligible.
- `trail_stop(entry_price, current_stop, last_price, atr) -> float|None` new higher stop or None (only moves up).
- `decide(ranking, spy_regime, hourly: dict[sym, dict(r_med, agree) | {"error"}], positions, stop_orders: dict[sym, dict(order_id, stop_price)], equity: float, halted: bool, stopped_out_today: set[str], last_prices: dict[sym,float], sectors=config.SECTORS) -> Plan`
  Order: exits, then entries, then trails; total orders (exits+entries+trails) <= MAX_ORDERS_PER_RUN.
  Exit if held and any: hourly STRONG_NEG; rank > ROTATION_RANK_EXIT or symbol missing from ranking... (missing => skip, not exit); r_med <= 0; daily regime BEARISH. Held symbol with hourly error -> no exit decision from hourly (other daily reasons still apply), add to skipped.
  Entries only if not halted and spy_regime != "BEARISH": symbol in candidates, not held, not in stopped_out_today, hourly CONFIRM_LONG, positions (after exits) < MAX_POSITIONS, sector count < MAX_PER_SECTOR, qty = floor(MAX_POSITION_FRAC*equity/price) >=1, total exposure after entry <= MAX_EXPOSURE_FRAC*equity. ref=last price, stop = ref - STOP_ATR_MULT*atr, target = ref + TARGET_ATR_MULT*atr, rounded to cents. Fill in rank order.
  Trails for held (not exiting) positions with a known stop order.
- `risk.py`: `is_halted(equity, last_equity) -> bool` (equity <= last_equity*(1-DAILY_HALT_FRAC)); `size_qty(equity, price, existing_value=0.0) -> int`.

## bot/state.py
- `StateStore(root: str|Path)` default root from env BOT_STATE_DIR or "./state". Files under `root/state/`.
  - `append(name, record: dict)` JSONL append (adds `ts` iso UTC if missing); `read(name) -> list[dict]` (missing -> []).
  - `write_json(name, obj)`, `read_json(name, default=None)`.
  - `has_ok(kind:str, key:str) -> bool` from runs.jsonl records `{kind, key, status:"ok"}`.
  - `record_run(kind, key, status, **extra)`.
  - `save_ranking(session_date: date, payload: dict)` -> rankings/YYYY-MM-DD.json ; `load_ranking(session_date) -> dict|None`; `latest_ranking() -> (date, dict)|(None,None)`.
  - `halt_today(date) -> dict|None`, `set_halt(date, reason)`.
  - `commit_and_push(message, retries=3) -> bool` runs git in root: add -A, commit (skip if nothing), pull --rebase, push; returns False on failure (never raises).
## bot/alerts.py
- `send_email(subject, text, html=None) -> bool` SMTP_SSL smtp.gmail.com:465 using GMAIL_ADDRESS / GMAIL_APP_PASSWORD, to ALERT_TO_EMAIL. Missing env -> log + False. Never raises.
- `send_push(title, message, priority="default"|"high"|"urgent", tags=None) -> bool` POST https://ntfy.sh/{NTFY_TOPIC}. Never raises.
## bot/gate.py (STDLIB ONLY: os, sys, json, urllib, datetime, zoneinfo)
- CLI `python -m bot.gate <trading|daily|watchdog|manual>`; prints reason; writes `proceed=true|false` to $GITHUB_OUTPUT if set; always exit 0 except paper-guard violation (exit 1).
- Checks: BOT_ENABLED env == "true" (else proceed=false); config.PAPER_BASE_URL unchanged; for trading: Alpaca /v2/clock is_open and ET minute window ok; for daily: today is a session (calendar) and now_et >= DAILY_READY_ET (or manual); watchdog: always proceed if enabled. `manual` always proceeds if enabled.
## bot/metrics.py
- `daily_returns(equity: pd.Series) -> pd.Series`; `sharpe(returns) -> float` (mean/std*sqrt(252), 0 if <2 pts or std 0); `max_drawdown(equity) -> float` (positive fraction); `hit_rate(preds_vs_actual: list[(pred_r, actual_r)]) -> (hit, base_rate, n)`.

## Runners (wave 2): bot/run_daily.py, bot/run_trading.py, bot/run_watchdog.py, bot/backtest.py; workflows in .github/workflows/.
