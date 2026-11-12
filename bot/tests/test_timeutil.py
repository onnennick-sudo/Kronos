from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from bot import config
from bot.timeutil import (
    ET,
    daily_df_for_model,
    filter_regular_hourly,
    future_daily_stamps,
    future_hourly_stamps,
    slot_key,
    to_naive_et,
)


def cal(d, close="16:00", open_="09:30"):
    return {"date": d, "open": time.fromisoformat(open_), "close": time.fromisoformat(close)}


def bars_for(day, labels):
    """Hourly bars labelled in ET on `day`, returned as tz-aware UTC."""
    ts = [pd.Timestamp(datetime.combine(day, time(h)), tz=ET).tz_convert("UTC") for h in labels]
    return pd.DataFrame({"timestamps": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})


# --- to_naive_et ---

def test_to_naive_et_across_dst_boundaries():
    s = pd.Series(pd.to_datetime([
        "2026-03-08T06:30:00Z",  # 01:30 EST (before spring-forward at 07:00Z)
        "2026-03-08T07:30:00Z",  # 03:30 EDT
        "2026-11-01T05:30:00Z",  # 01:30 EDT (first pass, before fall-back at 06:00Z)
        "2026-11-01T06:30:00Z",  # 01:30 EST (second pass)
    ], utc=True))
    out = to_naive_et(s)
    assert out.dt.tz is None
    assert out.tolist() == [
        datetime(2026, 3, 8, 1, 30),
        datetime(2026, 3, 8, 3, 30),
        datetime(2026, 11, 1, 1, 30),
        datetime(2026, 11, 1, 1, 30),
    ]


# --- filter_regular_hourly ---

def test_filter_keeps_regular_hours_and_drops_extended_labels():
    day = date(2026, 11, 10)  # normal Tuesday, close 16:00
    labels = [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19]
    out = filter_regular_hourly(bars_for(day, labels), {day}, {day: time(16)})
    assert out["timestamps"].dt.tz is None
    assert [t.hour for t in out["timestamps"]] == [9, 10, 11, 12, 13, 14, 15]
    assert all(t.date() == day for t in out["timestamps"])


def test_filter_early_close_drops_labels_at_or_after_close():
    day = date(2026, 11, 27)  # early close 13:00
    labels = [8, 9, 10, 11, 12, 13, 14, 15]
    out = filter_regular_hourly(bars_for(day, labels), {day}, {day: time(13)})
    assert [t.hour for t in out["timestamps"]] == [9, 10, 11, 12]


def test_filter_drops_non_session_dates():
    sat = date(2026, 11, 14)
    tue = date(2026, 11, 10)
    thanksgiving = date(2026, 11, 26)
    df = pd.concat([
        bars_for(sat, [10]),
        bars_for(thanksgiving, [10]),
        bars_for(tue, [10]),
    ], ignore_index=True)
    out = filter_regular_hourly(df, {tue}, {tue: time(16), sat: time(16)})
    assert [t.date() for t in out["timestamps"]] == [tue]


def test_filter_handles_empty_frame():
    empty = pd.DataFrame({"timestamps": pd.Series([], dtype="datetime64[ns, UTC]"), "close": []})
    assert filter_regular_hourly(empty, set(), {}).empty


# --- daily_df_for_model ---

def test_daily_df_for_model_maps_to_et_midnight():
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2026-11-10T05:00:00Z", "2026-11-11T05:00:00Z"], utc=True),
        "close": [1.0, 2.0],
    })
    out = daily_df_for_model(df)
    assert out["timestamps"].tolist() == [datetime(2026, 11, 10), datetime(2026, 11, 11)]
    assert out["timestamps"].dt.tz is None


# --- future_daily_stamps ---

def test_future_daily_skips_weekends_and_holidays():
    calendar = [
        cal(date(2026, 11, 24)),
        cal(date(2026, 11, 25)),
        # 2026-11-26 is Thanksgiving: absent from the calendar
        cal(date(2026, 11, 27), close="13:00"),
        cal(date(2026, 11, 30)),
    ]
    out = future_daily_stamps(calendar, after=date(2026, 11, 24), n=2)
    assert out.tolist() == [datetime(2026, 11, 25), datetime(2026, 11, 27)]
    out3 = future_daily_stamps(calendar, after=date(2026, 11, 25), n=2)
    assert out3.tolist() == [datetime(2026, 11, 27), datetime(2026, 11, 30)]


def test_future_daily_raises_when_calendar_too_short():
    with pytest.raises(ValueError):
        future_daily_stamps([cal(date(2026, 11, 10))], after=date(2026, 11, 9), n=3)


# --- future_hourly_stamps ---

def test_future_hourly_normal_day_labels_09_to_15():
    calendar = [cal(date(2026, 11, 9)), cal(date(2026, 11, 10))]
    out = future_hourly_stamps(calendar, datetime(2026, 11, 9, 14), n=8)
    assert out.tolist() == [
        datetime(2026, 11, 9, 15),
        *[datetime(2026, 11, 10, h) for h in range(9, 16)],
    ]


def test_future_hourly_early_close_and_crossing_to_next_session():
    calendar = [
        cal(date(2026, 11, 25)),
        cal(date(2026, 11, 27), close="13:00"),
        cal(date(2026, 11, 30)),
    ]
    out = future_hourly_stamps(calendar, datetime(2026, 11, 25, 15), n=5)
    assert out.tolist() == [
        datetime(2026, 11, 27, 9),
        datetime(2026, 11, 27, 10),
        datetime(2026, 11, 27, 11),
        datetime(2026, 11, 27, 12),
        datetime(2026, 11, 30, 9),
    ]


def test_future_hourly_excludes_last_label_and_raises_when_short():
    calendar = [cal(date(2026, 11, 10))]
    with pytest.raises(ValueError):
        future_hourly_stamps(calendar, datetime(2026, 11, 10, 14), n=3)


def test_future_hourly_uses_config_labels(monkeypatch):
    monkeypatch.setattr(config, "REGULAR_HOUR_LABELS", (10, 11))
    out = future_hourly_stamps([cal(date(2026, 11, 10))], datetime(2026, 11, 10, 9), n=2)
    assert out.tolist() == [datetime(2026, 11, 10, 10), datetime(2026, 11, 10, 11)]


# --- slot_key ---

def test_slot_key_uses_previous_hour_label():
    assert slot_key(datetime(2026, 11, 12, 11, 20, tzinfo=ET)) == "2026-11-12-10"


def test_slot_key_converts_aware_utc_to_et():
    # 16:20 UTC on 2026-11-12 is 11:20 EST
    assert slot_key(datetime(2026, 11, 12, 16, 20, tzinfo=timezone.utc)) == "2026-11-12-10"


def test_slot_key_naive_is_taken_as_et():
    assert slot_key(datetime(2026, 11, 12, 15, 5)) == "2026-11-12-14"
