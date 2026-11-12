"""Persistent bot state: JSONL run log, JSON snapshots, rankings, halts, git sync.

Layout: ``<root>/state/<name>``. ``root`` is the bot-state checkout (a git repo)
and is the working directory for ``commit_and_push``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bot import config

log = logging.getLogger(__name__)

RUNS_FILE = "runs.jsonl"
HALTS_FILE = "halts.jsonl"
RANKINGS_DIR = "rankings"
GIT_USER_NAME = "kronos-bot"
GIT_USER_EMAIL = "kronos-bot@users.noreply.github.com"


def _as_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _ensure_identity(cwd: str) -> None:
    """Set a local git identity only where none is configured."""
    for key, value in (("user.name", GIT_USER_NAME), ("user.email", GIT_USER_EMAIL)):
        current = _git(cwd, "config", key)
        if current.returncode != 0 or not current.stdout.strip():
            _git(cwd, "config", key, value)


class StateStore:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get(config.STATE_DIR_ENV) or "./state"
        self.root = Path(root)
        self.dir = self.root / "state"

    # ----- generic file helpers -------------------------------------------
    def _path(self, name: str) -> Path:
        return self.dir / name

    def append(self, name: str, record: dict[str, Any]) -> None:
        rec = dict(record)
        if "ts" not in rec:
            rec["ts"] = datetime.now(timezone.utc).isoformat()
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def write_json(self, name: str, obj: Any) -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, default=str, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self._path(name)
        if not path.exists():
            return default
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    # ----- run log ----------------------------------------------------------
    def has_ok(self, kind: str, key: str) -> bool:
        return any(
            r.get("kind") == kind and r.get("key") == key and r.get("status") == "ok"
            for r in self.read(RUNS_FILE)
        )

    def record_run(self, kind: str, key: str, status: str, **extra: Any) -> None:
        self.append(RUNS_FILE, {"kind": kind, "key": key, "status": status, **extra})

    # ----- rankings ---------------------------------------------------------
    @staticmethod
    def _ranking_name(session_date: date | datetime) -> str:
        return f"{RANKINGS_DIR}/{_as_date(session_date).isoformat()}.json"

    def save_ranking(self, session_date: date | datetime, payload: dict[str, Any]) -> None:
        self.write_json(self._ranking_name(session_date), payload)

    def load_ranking(self, session_date: date | datetime) -> dict[str, Any] | None:
        return self.read_json(self._ranking_name(session_date), default=None)

    def latest_ranking(self) -> tuple[date, dict[str, Any]] | tuple[None, None]:
        rdir = self.dir / RANKINGS_DIR
        if not rdir.is_dir():
            return None, None
        dated: list[date] = []
        for p in rdir.glob("*.json"):
            try:
                dated.append(date.fromisoformat(p.stem))
            except ValueError:
                log.warning("ignoring non-date ranking file %s", p.name)
        if not dated:
            return None, None
        latest = max(dated)
        return latest, self.load_ranking(latest)

    # ----- halts ------------------------------------------------------------
    def halt_today(self, day: date | datetime) -> dict[str, Any] | None:
        key = _as_date(day).isoformat()
        matches = [r for r in self.read(HALTS_FILE) if r.get("date") == key]
        return matches[-1] if matches else None

    def set_halt(self, day: date | datetime, reason: str) -> None:
        self.append(HALTS_FILE, {"date": _as_date(day).isoformat(), "reason": reason})

    # ----- git sync ---------------------------------------------------------
    def commit_and_push(self, message: str, retries: int = 3) -> bool:
        """Stage all, commit if anything changed, then pull --rebase and push.

        Returns True on success (including nothing-to-commit), False on any
        failure. Never raises.
        """
        try:
            cwd = str(self.root)
            if _git(cwd, "add", "-A").returncode != 0:
                log.error("git add failed in %s", cwd)
                return False

            staged = _git(cwd, "diff", "--cached", "--quiet")
            if staged.returncode == 1:
                _ensure_identity(cwd)
                commit = _git(cwd, "commit", "-m", message)
                if commit.returncode != 0:
                    log.error("git commit failed: %s", commit.stderr.strip())
                    return False
            elif staged.returncode == 0:
                log.info("nothing to commit in %s", cwd)
            else:
                log.error("git diff --cached failed: %s", staged.stderr.strip())
                return False

            for attempt in range(1, retries + 1):
                pull = _git(cwd, "pull", "--rebase")
                if pull.returncode != 0:
                    log.warning("git pull --rebase failed (attempt %d): %s",
                                attempt, pull.stderr.strip())
                    _git(cwd, "rebase", "--abort")
                    continue
                push = _git(cwd, "push")
                if push.returncode == 0:
                    return True
                log.warning("git push failed (attempt %d): %s",
                            attempt, push.stderr.strip())
            return False
        except Exception:  # noqa: BLE001 - contract: never raise
            log.exception("commit_and_push failed")
            return False
