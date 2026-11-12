import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bot import config, gate

ET = ZoneInfo("America/New_York")
SESSION_DAY = "2026-11-12"          # a Thursday, regular session
REPO_ROOT = Path(__file__).resolve().parents[2]


def et(hour, minute=0, day=12):
    return datetime(2026, 11, day, hour, minute, tzinfo=ET)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("BOT_ENABLED", "true")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(config, "PAPER_BASE_URL", gate.PAPER_URL)
    return monkeypatch


def fake_http(monkeypatch, *, is_open=True, sessions=(SESSION_DAY,), fail=False):
    """Route HTTP calls by path. Records every URL it was asked for."""
    calls = []

    def fake(url, headers):
        calls.append(url)
        if fail:
            raise OSError("network down")
        if "/v2/clock" in url:
            return {"is_open": is_open, "timestamp": "2026-11-12T10:20:00-05:00"}
        if "/v2/calendar" in url:
            return [{"date": d, "open": "09:30", "close": "16:00"} for d in sessions]
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(gate, "http_get_json", fake)
    return calls


def set_now(monkeypatch, dt):
    monkeypatch.setattr(gate, "now", lambda: dt)


# ----- disabled / manual / watchdog ---------------------------------------

def test_disabled_is_false_without_http(env):
    env.setenv("BOT_ENABLED", "false")
    calls = fake_http(env)
    set_now(env, et(10, 20))
    assert gate.evaluate("trading") == (False, "BOT_ENABLED is not 'true'")
    assert gate.evaluate("manual")[0] is False
    assert calls == []


def test_unset_bot_enabled_is_disabled(env):
    env.delenv("BOT_ENABLED")
    assert gate.evaluate("watchdog")[0] is False


def test_manual_proceeds_whenever_enabled(env):
    calls = fake_http(env)
    set_now(env, et(3, 0))          # middle of the night, market closed, not a session
    assert gate.evaluate("manual")[0] is True
    assert calls == []              # no API needed


def test_watchdog_proceeds_whenever_enabled(env):
    fake_http(env, is_open=False)
    set_now(env, et(3, 0))
    assert gate.evaluate("watchdog")[0] is True


def test_unknown_mode_does_not_proceed(env):
    fake_http(env)
    set_now(env, et(10, 20))
    assert gate.evaluate("bogus")[0] is False


# ----- trading ------------------------------------------------------------

def test_trading_open_inside_window_proceeds(env):
    fake_http(env, is_open=True)
    set_now(env, et(10, 20))
    proceed, reason = gate.evaluate("trading")
    assert proceed is True and "open" in reason


def test_trading_closed_market_inside_window_does_not_proceed(env):
    fake_http(env, is_open=False)
    set_now(env, et(10, 20))
    assert gate.evaluate("trading") == (False, "market closed")


def test_trading_open_but_outside_window_does_not_proceed(env):
    fake_http(env, is_open=True)
    set_now(env, et(9, 30))         # before 10:05
    assert gate.evaluate("trading")[0] is False
    set_now(env, et(16, 10))        # after 16:00
    assert gate.evaluate("trading")[0] is False


def test_trading_window_boundaries(env):
    fake_http(env, is_open=True)
    set_now(env, et(10, 5))
    assert gate.evaluate("trading")[0] is True
    set_now(env, et(10, 4))
    assert gate.evaluate("trading")[0] is False
    set_now(env, et(15, 50))
    assert gate.evaluate("trading")[0] is True


def test_trading_converts_aware_time_to_et(env):
    fake_http(env, is_open=True)
    utc = datetime(2026, 11, 12, 15, 20, tzinfo=ZoneInfo("UTC"))  # 10:20 ET
    set_now(env, utc)
    assert gate.evaluate("trading")[0] is True


def test_trading_clock_failure_inside_window_proceeds_with_warning(env, capsys):
    fake_http(env, fail=True)
    set_now(env, et(10, 20))
    proceed, _ = gate.evaluate("trading")
    assert proceed is True
    assert "WARNING" in capsys.readouterr().err


def test_trading_clock_failure_outside_window_does_not_proceed(env, capsys):
    fake_http(env, fail=True)
    set_now(env, et(22, 0))
    assert gate.evaluate("trading")[0] is False
    assert "WARNING" in capsys.readouterr().err


def test_clock_request_uses_paper_url_and_auth_headers(env):
    env.setenv("ALPACA_API_KEY_ID", "kid")
    env.setenv("ALPACA_API_SECRET_KEY", "ksecret")
    seen = {}

    def fake(url, headers):
        seen.update(url=url, headers=headers)
        return {"is_open": True}

    env.setattr(gate, "http_get_json", fake)
    set_now(env, et(10, 20))
    gate.evaluate("trading")
    assert seen["url"] == f"{gate.PAPER_URL}/v2/clock"
    assert seen["headers"]["APCA-API-KEY-ID"] == "kid"
    assert seen["headers"]["APCA-API-SECRET-KEY"] == "ksecret"


# ----- daily --------------------------------------------------------------

def test_daily_session_before_ready_does_not_proceed(env):
    fake_http(env)
    set_now(env, et(16, 15))        # DAILY_READY_ET is 16:16
    assert gate.evaluate("daily")[0] is False


def test_daily_session_at_or_after_ready_proceeds(env):
    fake_http(env)
    set_now(env, et(16, 16))
    assert gate.evaluate("daily")[0] is True
    set_now(env, et(18, 0))
    assert gate.evaluate("daily")[0] is True


def test_daily_non_session_day_does_not_proceed(env):
    fake_http(env, sessions=())     # calendar says today is not a session
    set_now(env, et(17, 0))
    proceed, reason = gate.evaluate("daily")
    assert proceed is False and "not a trading session" in reason


def test_daily_calendar_query_is_for_today(env):
    calls = fake_http(env)
    set_now(env, et(17, 0))
    gate.evaluate("daily")
    assert any("start=2026-11-12&end=2026-11-12" in u for u in calls)


def test_daily_calendar_failure_after_ready_proceeds_with_warning(env, capsys):
    fake_http(env, fail=True)
    set_now(env, et(17, 0))
    assert gate.evaluate("daily")[0] is True
    assert "WARNING" in capsys.readouterr().err


def test_daily_calendar_failure_before_ready_does_not_proceed(env, capsys):
    fake_http(env, fail=True)
    set_now(env, et(9, 0))
    assert gate.evaluate("daily")[0] is False


# ----- main(): CLI behaviour ----------------------------------------------

def test_main_writes_github_output_true(env, tmp_path, capsys):
    out = tmp_path / "gh_output"
    env.setenv("GITHUB_OUTPUT", str(out))
    fake_http(env, is_open=True)
    set_now(env, et(10, 20))
    assert gate.main(["trading"]) == 0
    assert out.read_text() == "proceed=true\n"
    assert "proceed=true" in capsys.readouterr().out


def test_main_writes_github_output_false_and_exits_zero(env, tmp_path):
    out = tmp_path / "gh_output"
    env.setenv("GITHUB_OUTPUT", str(out))
    fake_http(env, is_open=False)
    set_now(env, et(10, 20))
    assert gate.main(["trading"]) == 0
    assert out.read_text() == "proceed=false\n"


def test_main_no_github_output_when_unset(env, tmp_path):
    fake_http(env)
    set_now(env, et(10, 20))
    assert gate.main(["manual"]) == 0   # must not crash without GITHUB_OUTPUT


def test_paper_guard_violation_exits_1_without_http(env, monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER_BASE_URL", "https://api.alpaca.markets")
    calls = fake_http(env)
    set_now(env, et(10, 20))
    assert gate.main(["trading"]) == 1
    assert calls == []
    assert "Refusing to run" in capsys.readouterr().err


def test_gate_imports_only_stdlib_and_config():
    # importing gate must not drag in pandas/numpy/torch
    code = ("import sys, bot.gate; "
            "bad = [m for m in ('pandas', 'numpy', 'torch', 'requests') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
