# Kronos bot: setup and operating guide

This bot trades fake money (Alpaca paper account). It runs by itself on GitHub Actions. Nothing runs on your computer.

## Setup checklist

Never paste keys or passwords into chat. Enter them only in GitHub.

1. Install the ntfy app on your phone. Subscribe to a long random topic (example: `kronos-7f3a9q2x8m`).
2. Google Account, Security: turn on 2-Step Verification.
3. Google Account, Security, App passwords: create "Kronos". Copy the 16 characters.
4. Alpaca: switch to Paper Trading and generate a paper API key. The Key ID starts with `PK`. The secret shows only once, so keep the page open.
5. On your fork, go to Settings, Secrets and variables, Actions, Secrets tab. Add `ALPACA_API_KEY_ID`.
6. Same page: add `ALPACA_API_SECRET_KEY`.
7. Same page: add `GMAIL_ADDRESS` (your Gmail address).
8. Same page: add `GMAIL_APP_PASSWORD` (the 16 characters).
9. Same page: add `ALERT_TO_EMAIL` (where the summaries go).
10. Same page: add `NTFY_TOPIC` (the topic from step 1).
11. Variables tab: add `BOT_ENABLED` = `true`.
12. Variables tab: add `BOT_MODE` = `dry_run`.
13. Day 1 morning: approve and merge the PR Claude opens into `master`. Schedules run only from `master`.
14. Actions tab: click "I understand my workflows, go ahead and enable them."
15. Day 1 morning, before 10:20 ET: Actions, "Kronos Daily", Run workflow, `dry_run` = true. This computes today's ranking and market filter from the last final daily bar.
16. Confirm the daily email and the phone push notification arrive.
17. Day 1: "Kronos Trading" runs every hour in dry run. Watch its logs and alerts through the day. Confirm it logs decisions and places no orders.
18. Day 2 morning, before 10:20 ET, if the Day 1 dry run is clean: Variables tab, change `BOT_MODE` to `paper`. Then reply "go paper" to Claude.

## How to stop the bot

- Go to Settings, Secrets and variables, Actions, Variables tab.
- Set `BOT_ENABLED` to `false`.
- Every scheduled run now exits right away and places no orders. Existing stop orders at Alpaca stay in place.
- To start again, set `BOT_ENABLED` back to `true`.

## How to switch dry run to paper

- Dry run: `BOT_MODE` = `dry_run`. The bot logs what it would do and places no orders.
- Paper: `BOT_MODE` = `paper`. The bot places fake-money orders at Alpaca.
- Manual runs from the Actions tab also have a `dry_run` box. Ticking it forces a dry run for that one run, even if `BOT_MODE` is `paper`.

## What each workflow does

- **Kronos Trading** (hourly, weekdays 10:20 to 15:20 ET, backup at :50): reads the ranking, checks the hourly forecast, and places or exits trades. Each slot runs once, so a duplicate run is harmless.
- **Kronos Daily** (weekdays, after the market closes): forecasts the 50 stocks and SPY, saves tomorrow's ranking, sends the daily email. It does nothing until the day's final price bar is ready.
- **Kronos Watchdog** (weekdays, during market hours and after 18:30 ET): checks that trading and daily runs actually happened. It sends a phone alert if not.
- **Kronos Backtest** (manual only): runs a historical test for the stocks, ETFs or crypto universe. It is started from the Actions tab and writes its report to the `bot-state` branch.
- **Kronos Tests** (on every change to `bot/`): runs the automated tests. A red mark here means do not merge.

## Where logs and state live

- **Run logs:** Actions tab. Click a workflow, then a run, to see its output.
- **State:** the `bot-state` branch, folder `state/`. Main files:
  - `runs.jsonl`: one line per run (ok, degraded or error).
  - `orders.jsonl`: every order or order intent.
  - `predictions.jsonl` and `outcomes.jsonl`: what Kronos predicted and what happened.
  - `equity.jsonl` and `metrics.csv`: account value and performance over time.
  - `halts.jsonl`: days the bot stopped taking new trades.
- Alpaca's paper account is the source of truth for positions and orders.

## Daily email and phone alerts

- **Daily email:** subject starts with `[Kronos dry_run]` or `[Kronos paper]`, followed by the date, equity and comparison with SPY. In dry run, the first line says "DRY RUN, no orders placed."
- **Phone alerts (ntfy):**
  - Urgent: the market is open and there has been no successful trading run for 150 minutes. Check the Actions tab.
  - High: no daily run by 18:30 ET, or no ranking for today. New trades are blocked until this is fixed.
  - Urgent: the bot halted new entries for the rest of the day after the account fell 2% from yesterday's close.
- If the Watchdog workflow itself fails, GitHub emails you.

## Timeline

- **Day 0:** Claude builds the bot and opens a PR. You do setup steps 1 to 12.
- **Day 1:** Merge the PR. Run the daily job by hand in dry run. Trading runs hourly in dry run. Check the email and the phone alert.
- **Day 2:** If the dry run was clean, switch `BOT_MODE` to `paper` before 10:20 ET.
- **Day 14:** Early read. About 10 trading days. This is information, not a pass or fail, because 10 days cannot separate skill from luck.
- **Day 28:** Go-live decision. All three must pass: no missed runs, Sharpe above SPY over the same window, and max drawdown below 8%.
- If Day 28 fails, the paper account keeps running. Fix the cause first through a reviewed PR. The 4-week window restarts from the fix date.
- Real money is out of scope for now.
