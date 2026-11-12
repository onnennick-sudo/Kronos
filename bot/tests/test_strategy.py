import pandas as pd
import pytest

from bot import config
from bot.strategy import (
    Entry,
    Exit,
    Trail,
    atr,
    build_ranking,
    candidates,
    classify_daily,
    classify_hourly,
    decide,
    trail_stop,
)

CONFIRM = {"r_med": 0.005, "agree": 0.70}
NONE = {"r_med": 0.0, "agree": 0.50}
STRONG_NEG = {"r_med": -0.01, "agree": 0.80}


def R(sym, rank, r=0.02, agree=0.8, atr_val=2.0):
    return {
        "symbol": sym,
        "rank": rank,
        "r_med": r,
        "agree": agree,
        "eligible": r >= config.DAILY_ELIGIBLE_R and agree >= config.DAILY_AGREE,
        "regime": classify_daily(r, agree),
        "atr": atr_val,
    }


def P(sym, qty, entry=100.0, cur=100.0):
    return {
        "symbol": sym,
        "qty": qty,
        "avg_entry_price": entry,
        "current_price": cur,
        "market_value": qty * cur,
    }


def run(**kw):
    args = dict(
        ranking=[],
        spy_regime="BULLISH",
        hourly={},
        positions=[],
        stop_orders={},
        equity=100000.0,
        halted=False,
        stopped_out_today=set(),
        last_prices={},
        sectors={},
    )
    args.update(kw)
    return decide(**args)


# --- classification -----------------------------------------------------------

def test_classify_daily_boundaries():
    assert classify_daily(config.DAILY_ELIGIBLE_R, config.DAILY_AGREE) == "BULLISH"
    assert classify_daily(0.01, 0.60) == "BULLISH"
    assert classify_daily(0.00999, 0.60) == "NEUTRAL"
    assert classify_daily(0.02, 0.59) == "NEUTRAL"
    assert classify_daily(config.DAILY_BEARISH_R, config.DAILY_AGREE) == "BEARISH"
    assert classify_daily(-0.01, 0.60) == "BEARISH"
    assert classify_daily(-0.00999, 0.60) == "NEUTRAL"
    assert classify_daily(-0.02, 0.59) == "NEUTRAL"


def test_classify_hourly_boundaries():
    assert classify_hourly(0.003, 0.60) == "CONFIRM_LONG"
    assert classify_hourly(0.00299, 0.60) == "NONE"
    assert classify_hourly(0.01, 0.59) == "NONE"
    assert classify_hourly(-0.005, 0.65) == "STRONG_NEG"
    assert classify_hourly(-0.005, 0.64) == "NONE"
    assert classify_hourly(-0.00499, 0.90) == "NONE"
    assert classify_hourly(-0.02, 0.60) == "NONE"  # STRONG_NEG needs 65% agreement


# --- atr ----------------------------------------------------------------------

def test_atr_constant_range():
    df = pd.DataFrame({"high": [101.0] * 5, "low": [99.0] * 5, "close": [100.0] * 5})
    assert atr(df, period=3) == pytest.approx(2.0)


def test_atr_wilder_smoothing_hand_computed():
    # TR1=1.5, TR2=1.5, TR3=2.5. Seed = mean(TR1, TR2) = 1.5. Then (1.5*1 + 2.5)/2 = 2.0.
    df = pd.DataFrame({
        "high": [10.0, 11.0, 12.0, 14.0],
        "low": [9.0, 10.0, 11.0, 12.0],
        "close": [9.5, 10.5, 11.5, 13.0],
    })
    assert atr(df, period=2) == pytest.approx(2.0)
    assert atr(df.iloc[:3], period=2) == pytest.approx(1.5)


def test_atr_returns_float_and_defaults_to_config(monkeypatch):
    monkeypatch.setattr(config, "ATR_PERIOD", 2)
    df = pd.DataFrame({"high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0], "close": [10.0, 11.0, 12.0]})
    value = atr(df)
    assert isinstance(value, float)
    assert value == pytest.approx(2.0)


def test_atr_raises_with_too_few_rows():
    df = pd.DataFrame({"high": [11.0, 12.0], "low": [9.0, 10.0], "close": [10.0, 11.0]})
    with pytest.raises(ValueError):
        atr(df, period=2)


# --- ranking ------------------------------------------------------------------

def test_build_ranking_order_eligibility_and_errors_excluded():
    stats = {
        "A": {"r_med": 0.03, "agree": 0.80},
        "B": {"r_med": 0.05, "agree": 0.50},  # best r but not eligible (agree)
        "C": {"error": "boom"},                # excluded
        "D": {"r_med": 0.01, "agree": 0.60},  # exact eligibility boundary
    }
    atrs = {"A": 1.0, "B": 2.0, "D": 3.0}
    rows = build_ranking(stats, atrs)
    assert [r["symbol"] for r in rows] == ["B", "A", "D"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert [r["eligible"] for r in rows] == [False, True, True]
    assert [r["regime"] for r in rows] == ["NEUTRAL", "BULLISH", "BULLISH"]
    assert [r["atr"] for r in rows] == [2.0, 1.0, 3.0]
    assert rows[1]["r_med"] == 0.03 and rows[1]["agree"] == 0.80


def test_build_ranking_ties_broken_by_symbol():
    stats = {"ZZZ": {"r_med": 0.02, "agree": 0.9}, "AAA": {"r_med": 0.02, "agree": 0.9}}
    rows = build_ranking(stats, {})
    assert [r["symbol"] for r in rows] == ["AAA", "ZZZ"]
    assert rows[0]["atr"] is None


def test_candidates_top_five_eligible_only():
    ranking = [
        R("R1", 1, r=0.005),  # not eligible
        R("R2", 2),
        R("R3", 3, agree=0.1),  # not eligible
        R("R4", 4),
        R("R5", 5),
        R("R6", 6),
        R("R7", 7),
        R("R8", 8),
    ]
    assert candidates(ranking) == ["R2", "R4", "R5", "R6", "R7"]
    assert candidates(ranking, n=2) == ["R2", "R4"]


def test_candidates_fewer_eligible_than_n():
    ranking = [R("X", 1), R("Y", 2, r=0.0)]
    assert candidates(ranking) == ["X"]


def test_candidates_default_n_reads_config(monkeypatch):
    monkeypatch.setattr(config, "N_CANDIDATES", 1)
    ranking = [R("X", 1), R("Y", 2)]
    assert candidates(ranking) == ["X"]


# --- trail_stop ---------------------------------------------------------------

def test_trail_none_below_breakeven_threshold():
    # unrealized 0.5 < 1 ATR (2.0)
    assert trail_stop(100.0, 95.0, 100.5, 2.0) is None


def test_trail_breakeven():
    # unrealized 2.5: >= 1 ATR and < 2 ATR (4.0) -> stop to entry 100.0
    assert trail_stop(100.0, 95.0, 102.5, 2.0) == 100.0


def test_trail_breakeven_no_change_when_stop_already_there():
    assert trail_stop(100.0, 100.0, 102.5, 2.0) is None


def test_trail_trailing_distance():
    # unrealized 10 >= 2 ATR -> 110 - 1.5*2 = 107.0
    assert trail_stop(100.0, 100.0, 110.0, 2.0) == 107.0


def test_trail_rounds_to_cents():
    # 110.126 - 1.5*1.0 = 108.626 -> 108.63
    assert trail_stop(100.0, 100.0, 110.126, 1.0) == 108.63


def test_trail_never_lowers_stop():
    # candidate 107.0 is below current stop 108.0
    assert trail_stop(100.0, 108.0, 108.5, 2.0) is None


# --- decide: basics -----------------------------------------------------------

def test_entry_sizing_and_stop_target_math():
    # 20% of 100000 = 20000 -> 200 shares at 100. stop = 100 - 2*2 = 96, target = 100 + 3*2 = 106.
    plan = run(ranking=[R("A", 1, atr_val=2.0)], hourly={"A": CONFIRM}, last_prices={"A": 100.0})
    assert len(plan.entries) == 1
    e = plan.entries[0]
    assert (e.symbol, e.qty, e.ref_price, e.stop, e.target, e.rank) == ("A", 200, 100.0, 96.0, 106.0, 1)
    assert plan.exits == [] and plan.trails == []


def test_stop_target_round_to_cents():
    plan = run(ranking=[R("A", 1, atr_val=1.111)], hourly={"A": CONFIRM}, last_prices={"A": 33.333})
    e = plan.entries[0]
    assert e.qty == 600
    assert e.stop == 31.11      # 33.333 - 2*1.111 = 31.111
    assert e.target == 36.67    # 33.333 + 3*1.111 = 36.666
    assert e.rank == 1


def test_qty_below_one_skipped():
    plan = run(ranking=[R("A", 1)], hourly={"A": CONFIRM}, last_prices={"A": 60000.0})
    assert plan.entries == []
    assert ("A", "qty_lt_1") in plan.skipped


def test_no_price_skipped():
    plan = run(ranking=[R("A", 1)], hourly={"A": CONFIRM}, last_prices={})
    assert plan.entries == []
    assert ("A", "no_price") in plan.skipped


def test_candidate_requires_hourly_confirm():
    plan = run(ranking=[R("A", 1)], hourly={"A": NONE}, last_prices={"A": 100.0})
    assert ("A", "no_hourly_confirm") in plan.skipped
    plan = run(ranking=[R("A", 1)], hourly={"A": {"error": "timeout"}}, last_prices={"A": 100.0})
    assert ("A", "hourly_error") in plan.skipped


def test_stopped_out_today_blocks_reentry():
    plan = run(ranking=[R("A", 1)], hourly={"A": CONFIRM}, last_prices={"A": 100.0},
               stopped_out_today={"A"})
    assert plan.entries == []
    assert ("A", "stopped_out_today") in plan.skipped


# --- decide: exits ------------------------------------------------------------

def test_exit_hourly_strong_neg():
    plan = run(ranking=[R("X", 1)], hourly={"X": STRONG_NEG}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "hourly_strong_neg")]


def test_exit_rank_rotation_boundary():
    plan = run(ranking=[R("X", 11)], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "rank_rotation")]
    plan = run(ranking=[R("X", 10)], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == []


def test_exit_r_med_nonpositive_boundary():
    plan = run(ranking=[R("X", 3, r=0.0, agree=0.8)], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "r_med_nonpositive")]
    plan = run(ranking=[R("X", 3, r=0.001, agree=0.8)], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == []


def test_exit_daily_bearish_reported_before_r_med_nonpositive():
    row = R("X", 3, r=-0.02, agree=0.8)
    assert row["regime"] == "BEARISH"
    plan = run(ranking=[row], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "daily_bearish")]


def test_exit_reason_priority_hourly_first():
    plan = run(ranking=[R("X", 11, r=-0.02)], hourly={"X": STRONG_NEG}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "hourly_strong_neg")]


def test_missing_forecast_no_daily_exit_but_recorded():
    plan = run(ranking=[], hourly={"X": NONE}, positions=[P("X", 10)])
    assert plan.exits == []
    assert ("X", "missing_forecast") in plan.skipped


def test_missing_forecast_still_allows_hourly_exit():
    plan = run(ranking=[], hourly={"X": STRONG_NEG}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "hourly_strong_neg")]


def test_held_hourly_error_no_hourly_exit_but_daily_still_applies():
    plan = run(ranking=[R("X", 11)], hourly={"X": {"error": "boom"}}, positions=[P("X", 10)])
    assert plan.exits == [Exit("X", 10, "rank_rotation")]
    assert ("X", "hourly_error") in plan.skipped


def test_held_hourly_error_with_healthy_daily_no_exit():
    plan = run(ranking=[R("X", 2)], hourly={"X": {"error": "boom"}}, positions=[P("X", 10)])
    assert plan.exits == []
    assert ("X", "hourly_error") in plan.skipped


def test_exited_symbol_is_not_trailed():
    plan = run(
        ranking=[R("X", 11, atr_val=2.0)],
        hourly={"X": NONE},
        positions=[P("X", 10, entry=100.0, cur=110.0)],
        stop_orders={"X": {"order_id": "o1", "stop_price": 95.0}},
        last_prices={"X": 110.0},
    )
    assert [e.symbol for e in plan.exits] == ["X"]
    assert plan.trails == []


# --- decide: halt / SPY -------------------------------------------------------

def test_spy_bearish_blocks_entries_but_not_exits():
    plan = run(
        ranking=[R("X", 11), R("A", 1)],
        spy_regime="BEARISH",
        hourly={"X": NONE, "A": CONFIRM},
        positions=[P("X", 10)],
        last_prices={"A": 100.0},
    )
    assert plan.exits == [Exit("X", 10, "rank_rotation")]
    assert plan.entries == []
    assert ("A", "spy_bearish") in plan.skipped


def test_halt_blocks_entries_not_exits_or_trails():
    plan = run(
        ranking=[R("X", 11), R("Z", 2, atr_val=2.0), R("A", 1)],
        halted=True,
        hourly={"X": NONE, "Z": NONE, "A": CONFIRM},
        positions=[P("X", 10), P("Z", 10, entry=100.0, cur=110.0)],
        stop_orders={"Z": {"order_id": "oz", "stop_price": 95.0}},
        last_prices={"A": 100.0, "Z": 110.0},
    )
    assert plan.exits == [Exit("X", 10, "rank_rotation")]
    assert plan.entries == []
    assert plan.trails == [Trail("Z", "oz", 95.0, 107.0)]
    assert ("A", "halted") in plan.skipped


# --- decide: caps and sizing --------------------------------------------------

def test_max_positions_blocks_entry(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITIONS", 3)
    positions = [P("P1", 1), P("P2", 1), P("P3", 1)]
    ranking = [R("P1", 2), R("P2", 3), R("P3", 4), R("A", 1)]
    plan = run(ranking=ranking, hourly={"A": CONFIRM}, positions=positions, last_prices={"A": 100.0})
    assert plan.entries == []
    assert ("A", "max_positions") in plan.skipped


def test_exit_frees_slot_for_entry(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITIONS", 3)
    positions = [P("X", 1), P("P2", 1), P("P3", 1)]
    ranking = [R("X", 11), R("P2", 2), R("P3", 3), R("A", 1)]
    plan = run(ranking=ranking, hourly={"X": NONE, "A": CONFIRM}, positions=positions,
               last_prices={"A": 100.0})
    assert [e.symbol for e in plan.exits] == ["X"]
    assert [e.symbol for e in plan.entries] == ["A"]


def test_exposure_cap_blocks_entry(monkeypatch):
    monkeypatch.setattr(config, "MAX_EXPOSURE_FRAC", 0.60)
    # remaining market value 50000 + planned 200*100 = 70000 > 60000
    plan = run(ranking=[R("H", 2), R("A", 1)], hourly={"H": NONE, "A": CONFIRM},
               positions=[P("H", 500, cur=100.0)], last_prices={"A": 100.0})
    assert plan.entries == []
    assert ("A", "exposure_cap") in plan.skipped


def test_exposure_cap_exact_boundary_allowed(monkeypatch):
    monkeypatch.setattr(config, "MAX_EXPOSURE_FRAC", 0.60)
    # 40000 + 200*100 = 60000 == cap -> allowed
    plan = run(ranking=[R("H", 2), R("A", 1)], hourly={"H": NONE, "A": CONFIRM},
               positions=[P("H", 400, cur=100.0)], last_prices={"A": 100.0})
    assert [e.symbol for e in plan.entries] == ["A"]
    assert plan.entries[0].qty == 200


def test_sector_cap_limits_to_two_per_sector(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITIONS", 3)
    sectors = {"AAA": "IT", "BBB": "IT", "CCC": "IT", "DDD": "FIN"}
    ranking = [R("BBB", 1), R("AAA", 2), R("CCC", 3), R("DDD", 4)]
    plan = run(
        ranking=ranking,
        hourly={"AAA": NONE, "BBB": CONFIRM, "CCC": CONFIRM, "DDD": CONFIRM},
        positions=[P("AAA", 10, cur=100.0)],
        last_prices={"BBB": 100.0, "CCC": 100.0, "DDD": 100.0},
        sectors=sectors,
    )
    assert [e.symbol for e in plan.entries] == ["BBB", "DDD"]
    assert ("CCC", "sector_cap") in plan.skipped


def test_order_cap_priority_exits_then_entries_then_trails(monkeypatch):
    monkeypatch.setattr(config, "MAX_ORDERS_PER_RUN", 3)
    ranking = [
        R("A", 1, atr_val=2.0),
        R("Z", 2, atr_val=2.0),
        R("B", 3, atr_val=2.0),
        R("X", 11, atr_val=2.0),
        R("Y", 12, atr_val=2.0),
    ]
    plan = run(
        ranking=ranking,
        hourly={"A": CONFIRM, "B": CONFIRM, "X": NONE, "Y": NONE, "Z": NONE},
        positions=[P("X", 1), P("Y", 1), P("Z", 10, entry=100.0, cur=110.0)],
        stop_orders={"Z": {"order_id": "oz", "stop_price": 95.0}},
        last_prices={"A": 100.0, "B": 100.0, "Z": 110.0},
    )
    assert [e.symbol for e in plan.exits] == ["X", "Y"]
    assert [e.symbol for e in plan.entries] == ["A"]
    assert plan.trails == []
    assert ("B", "order_cap") in plan.skipped
    assert ("Z", "order_cap") in plan.skipped


def test_order_cap_allows_everything_within_budget(monkeypatch):
    # cap 5: 2 exits, 2 entries (A, B), 1 trail (Z) = exactly 5 orders.
    monkeypatch.setattr(config, "MAX_ORDERS_PER_RUN", 5)
    ranking = [R("A", 1, atr_val=2.0), R("Z", 2, atr_val=2.0), R("B", 3, atr_val=2.0),
               R("X", 11, atr_val=2.0), R("Y", 12, atr_val=2.0)]
    plan = run(
        ranking=ranking,
        hourly={"A": CONFIRM, "B": CONFIRM, "X": NONE, "Y": NONE, "Z": NONE},
        positions=[P("X", 1), P("Y", 1), P("Z", 10, entry=100.0, cur=110.0)],
        stop_orders={"Z": {"order_id": "oz", "stop_price": 95.0}},
        last_prices={"A": 100.0, "B": 100.0, "Z": 110.0},
    )
    assert [e.symbol for e in plan.exits] == ["X", "Y"]
    assert [e.symbol for e in plan.entries] == ["A", "B"]
    assert plan.trails == [Trail("Z", "oz", 95.0, 107.0)]
    assert not [s for s in plan.skipped if s[1] == "order_cap"]


def test_exits_beyond_cap_are_skipped(monkeypatch):
    monkeypatch.setattr(config, "MAX_ORDERS_PER_RUN", 1)
    plan = run(
        ranking=[R("X", 11), R("Y", 12), R("A", 1)],
        hourly={"X": NONE, "Y": NONE, "A": CONFIRM},
        positions=[P("X", 1), P("Y", 1)],
        last_prices={"A": 100.0},
    )
    assert [e.symbol for e in plan.exits] == ["X"]
    assert ("Y", "order_cap") in plan.skipped
    assert plan.entries == []


# --- decide: trails -----------------------------------------------------------

def test_trail_emitted_for_held_with_stop_order():
    plan = run(
        ranking=[R("Z", 2, atr_val=2.0)],
        hourly={"Z": NONE},
        positions=[P("Z", 10, entry=100.0, cur=110.0)],
        stop_orders={"Z": {"order_id": "oz", "stop_price": 95.0}},
        last_prices={"Z": 110.0},
    )
    assert plan.trails == [Trail("Z", "oz", 95.0, 107.0)]


def test_held_without_stop_order_is_skipped():
    plan = run(ranking=[R("Z", 2, atr_val=2.0)], hourly={"Z": NONE},
               positions=[P("Z", 10, entry=100.0, cur=110.0)], last_prices={"Z": 110.0})
    assert plan.trails == []
    assert ("Z", "no_stop_order") in plan.skipped


def test_trail_uses_last_price_then_falls_back_to_current_price():
    plan = run(
        ranking=[R("Z", 2, atr_val=2.0)],
        hourly={"Z": NONE},
        positions=[P("Z", 10, entry=100.0, cur=110.0)],
        stop_orders={"Z": {"order_id": "oz", "stop_price": 95.0}},
    )
    assert plan.trails == [Trail("Z", "oz", 95.0, 107.0)]
