import json
import subprocess
from datetime import date, datetime

import pytest

from bot import config
from bot.state import StateStore


def test_default_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path / "botstate"))
    s = StateStore()
    assert s.root == tmp_path / "botstate"
    assert s.dir == tmp_path / "botstate" / "state"


def test_append_and_read(tmp_path):
    s = StateStore(tmp_path)
    assert s.read("events.jsonl") == []
    s.append("events.jsonl", {"a": 1})
    s.append("events.jsonl", {"a": 2, "ts": "fixed"})
    rows = s.read("events.jsonl")
    assert [r["a"] for r in rows] == [1, 2]
    assert "ts" in rows[0] and rows[0]["ts"]          # auto-added
    assert rows[1]["ts"] == "fixed"                    # preserved


def test_append_serialises_dates_with_default_str(tmp_path):
    s = StateStore(tmp_path)
    s.append("events.jsonl", {"d": date(2026, 11, 12)})
    assert s.read("events.jsonl")[0]["d"] == "2026-11-12"


def test_write_json_read_json_and_atomic(tmp_path):
    s = StateStore(tmp_path)
    assert s.read_json("cfg.json", default={"x": 0}) == {"x": 0}
    s.write_json("cfg.json", {"x": 1, "when": date(2026, 1, 2)})
    assert s.read_json("cfg.json") == {"x": 1, "when": "2026-01-02"}
    # no leftover tmp file after an atomic replace
    assert not (s.dir / "cfg.json.tmp").exists()


def test_has_ok_and_record_run(tmp_path):
    s = StateStore(tmp_path)
    assert s.has_ok("daily", "2026-11-12") is False
    s.record_run("daily", "2026-11-12", "error", error="boom")
    assert s.has_ok("daily", "2026-11-12") is False
    s.record_run("daily", "2026-11-12", "ok", n=5)
    assert s.has_ok("daily", "2026-11-12") is True
    assert s.has_ok("trading", "2026-11-12") is False   # kind must match
    assert s.has_ok("daily", "2026-11-13") is False     # key must match
    recs = s.read("runs.jsonl")
    assert recs[-1]["n"] == 5 and recs[-1]["status"] == "ok"


def test_ranking_save_load_latest(tmp_path):
    s = StateStore(tmp_path)
    assert s.latest_ranking() == (None, None)
    assert s.load_ranking(date(2026, 11, 10)) is None

    s.save_ranking(date(2026, 11, 10), {"rows": ["old"]})
    s.save_ranking(datetime(2026, 11, 12, 16, 30), {"rows": ["new"]})  # datetime normalised
    assert s.load_ranking(date(2026, 11, 10)) == {"rows": ["old"]}
    assert (s.dir / "rankings" / "2026-11-12.json").exists()

    d, payload = s.latest_ranking()
    assert d == date(2026, 11, 12)
    assert payload == {"rows": ["new"]}


def test_latest_ranking_ignores_non_date_files(tmp_path):
    s = StateStore(tmp_path)
    s.save_ranking(date(2026, 11, 9), {"ok": True})
    (s.dir / "rankings" / "notes.json").write_text("{}")
    d, payload = s.latest_ranking()
    assert d == date(2026, 11, 9) and payload == {"ok": True}


def test_halt_today_and_set_halt(tmp_path):
    s = StateStore(tmp_path)
    assert s.halt_today(date(2026, 11, 12)) is None
    s.set_halt(date(2026, 11, 12), "drawdown")
    h = s.halt_today(date(2026, 11, 12))
    assert h["reason"] == "drawdown"
    assert s.halt_today(date(2026, 11, 13)) is None
    s.set_halt(datetime(2026, 11, 12, 15, 0), "second reason")
    assert s.halt_today(date(2026, 11, 12))["reason"] == "second reason"


# ----- commit_and_push against a temporary bare remote --------------------

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture
def git_env(tmp_path):
    """bare remote + clone that acts as the bot-state checkout. Nothing outside tmp_path."""
    bare = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "-b", "main", "--bare", str(bare)], check=True)
    seed.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    (seed / "README.md").write_text("seed\n")
    _git(seed, "-c", "user.name=seed", "-c", "user.email=seed@example.invalid",
         "add", "-A")
    _git(seed, "-c", "user.name=seed", "-c", "user.email=seed@example.invalid",
         "commit", "-q", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(bare))
    _git(seed, "push", "-q", "-u", "origin", "main")
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    return bare, work


def test_commit_and_push_success(git_env):
    bare, work = git_env
    s = StateStore(work)
    s.append("runs.jsonl", {"kind": "daily", "key": "k", "status": "ok"})
    assert s.commit_and_push("bot: test commit") is True

    # the commit reached the bare remote
    log = _git(bare, "log", "--format=%s", "main")
    assert log.splitlines()[0] == "bot: test commit"
    # local identity defaults were applied (unless a global identity exists)
    assert _git(work, "config", "user.name")
    assert _git(work, "config", "user.email")
    assert _git(work, "status", "--porcelain") == ""


def test_commit_and_push_nothing_to_commit(git_env):
    bare, work = git_env
    before = _git(bare, "rev-parse", "main")
    s = StateStore(work)
    assert s.commit_and_push("bot: nothing") is True
    assert _git(bare, "rev-parse", "main") == before   # no new commit pushed


def test_commit_and_push_returns_false_when_remote_unreachable(git_env, tmp_path):
    _, work = git_env
    _git(work, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))
    s = StateStore(work)
    s.append("runs.jsonl", {"kind": "x", "key": "y", "status": "ok"})
    assert s.commit_and_push("bot: will fail", retries=2) is False


def test_commit_and_push_never_raises_outside_repo(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    s = StateStore(not_a_repo)
    # git may or may not be able to walk up; either way the call must not raise
    assert s.commit_and_push("bot: x") in (True, False)
