"""Time helpers for the Kronos bot: ET conversion, session calendar, bar labels, slot keys.

Conventions:
- Model inputs and all labels are tz-NAIVE ET datetimes.
- Helpers that receive tz-aware timestamps convert them to ET first.
- Naive timestamps passed to `filter_regular_hourly` / `daily_df_for_model` are taken as ET already.
- `to_naive_et` is the only helper that treats naive input as UTC (it is for broker UTC timestamps).
"""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from bot import config

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def to_naive_et(ts_series_utc: pd.Series) -> pd.Series:
    """Tz-aware UTC (or naive, assumed UTC) -> tz-naive ET."""
    s = pd.to_datetime(ts_series_utc, utc=True)
    return s.dt.tz_convert(ET).dt.tz_localize(None)


def _to_naive_et_any(ts: pd.Series) -> pd.Series:
    """Tz-aware -> naive ET; already-naive input is assumed to be ET."""
    s = pd.to_datetime(ts)
    if s.dt.tz is None:
        return s
    return s.dt.tz_convert(ET).dt.tz_localize(None)


def _as_date(d) -> date:
    return d.date() if isinstance(d, datetime) else d


def _session_rows(calendar: list[dict]) -> list[tuple[date, time]]:
    """Sorted unique (date, close) pairs from an Alpaca-style calendar."""
    by_date: dict[date, time] = {}
    for c in calendar:
        by_date[_as_date(c["date"])] = c["close"]
    return sorted(by_date.items())


def filter_regular_hourly(df: pd.DataFrame, session_dates: set[date], closes: dict[date, time]) -> pd.DataFrame:
    """Keep regular-session hourly bars only.

    A bar is kept when its ET label date is a session date, its label hour is in
    config.REGULAR_HOUR_LABELS, and the label is before that day's close.
    The `timestamps` column is returned as tz-naive ET. Result is sorted and re-indexed.
    """
    out = df.copy()
    if out.empty:
        return out.reset_index(drop=True)
    ts = _to_naive_et_any(out["timestamps"])
    out["timestamps"] = ts
    sessions = {_as_date(d) for d in session_dates}

    def keep(label: datetime) -> bool:
        d = label.date()
        if d not in sessions or d not in closes:
            return False
        if label.hour not in config.REGULAR_HOUR_LABELS:
            return False
        return label < datetime.combine(d, closes[d])

    mask = [keep(t) for t in ts]
    return out.loc[mask].sort_values("timestamps").reset_index(drop=True)


def daily_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars with `timestamps` as naive ET dates at 00:00. Sorted and re-indexed."""
    out = df.copy()
    ts = _to_naive_et_any(out["timestamps"])
    out["timestamps"] = ts.dt.normalize()
    return out.sort_values("timestamps").reset_index(drop=True)


def future_daily_stamps(calendar: list[dict], after: date, n: int) -> pd.Series:
    """Next n session dates strictly after `after`, as naive 00:00 datetimes.

    Sessions come from the calendar list (holidays are absent from it); weekends are skipped defensively.
    Raises ValueError if the calendar does not cover n sessions.
    """
    after = _as_date(after)
    dates = [d for d, _ in _session_rows(calendar) if d > after and d.weekday() < 5]
    if len(dates) < n:
        raise ValueError(f"calendar has {len(dates)} sessions after {after}, need {n}")
    return pd.Series(pd.to_datetime([datetime.combine(d, time(0)) for d in dates[:n]]), name="timestamps")


def future_hourly_stamps(calendar: list[dict], last_label: datetime, n: int) -> pd.Series:
    """Next n clock-hour labels strictly after `last_label`, within regular sessions.

    For each session, hours h in config.REGULAR_HOUR_LABELS with datetime(date, h:00) < close.
    `last_label` is naive ET (tz-aware input is converted to ET). Raises ValueError if the
    calendar does not cover n labels.
    """
    if last_label.tzinfo is not None:
        last_label = last_label.astimezone(ET).replace(tzinfo=None)
    labels: list[datetime] = []
    if n > 0:
        for d, close in _session_rows(calendar):
            close_dt = datetime.combine(d, close)
            for h in config.REGULAR_HOUR_LABELS:
                label = datetime.combine(d, time(h))
                if label <= last_label or label >= close_dt:
                    continue
                labels.append(label)
                if len(labels) == n:
                    return pd.Series(pd.to_datetime(labels), name="timestamps")
        raise ValueError(f"calendar yields fewer than {n} hourly labels after {last_label}")
    return pd.Series(pd.to_datetime(labels), name="timestamps")


def slot_key(now: datetime) -> str:
    """'YYYY-MM-DD-HH' where HH is the decision-bar label hour (now.hour - 1 in ET).

    Example: an 11:20 ET run -> '...-10'.
    """
    if now.tzinfo is not None:
        now = now.astimezone(ET)
    return f"{now:%Y-%m-%d}-{now.hour - 1:02d}"
