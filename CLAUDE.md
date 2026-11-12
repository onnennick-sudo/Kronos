# Kronos repo notes for Claude

## Project
- Kronos paper-trading bot lives in `bot/`. It forecasts stocks with the Kronos model and trades a paper Alpaca account.
- Upstream Kronos code in `model/` (and the rest of the repo) is untouched. Do not edit it.
- Bot runs on GitHub Actions workflows in `.github/workflows/`.

## Working standard
- Opus is manager and reviewer. Haiku agents do the building (code, tests, drafts).
- Opus reviews all agent work and sends it back with corrections until it is at least 95% confident before anything goes to Nick.
- Ask intake questions before starting any big new project.
- Nick's preferences: bullets, direct recommendations, minimal filler, one question at a time.

## Bot rules
- Paper only. Never change `PAPER_BASE_URL`.
- All tunables live in `bot/config.py`. Config and strategy changes go only through a reviewed PR to `master`.
- Never push directly to `master`.
- Never weaken risk rules without Nick's explicit approval.
- Runners must stay idempotent (a repeat run must not double-trade or double-send).
- Tests must pass before merging: `python -m pytest bot/tests -q`.

## Pointers
- Contracts: `bot/CONTRACT.md`, `bot/CONTRACT_WAVE2.md`.
- Nick's setup and operating guide: `bot/SETUP.md`.
- Bot state (logs, rankings, metrics, backtests) lives on the `bot-state` branch, not on `master`.
