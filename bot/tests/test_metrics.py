import math

import numpy as np
import pandas as pd
import pytest

from bot import metrics


def test_daily_returns_known_values():
    eq = pd.Series([100.0, 110.0, 99.0, 99.0])
    r = metrics.daily_returns(eq)
    assert len(r) == 3                       # leading NaN dropped
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)
    assert r.iloc[2] == pytest.approx(0.0)


def test_daily_returns_drops_infinities():
    eq = pd.Series([0.0, 5.0, 10.0])         # pct_change from 0 -> inf
    r = metrics.daily_returns(eq)
    assert np.isfinite(r).all()
    assert list(r) == [pytest.approx(1.0)]


def test_sharpe_known_value():
    rets = pd.Series([0.01, 0.03])
    # mean 0.02, sample std sqrt(0.0002) (ddof=1)
    expected = 0.02 / math.sqrt(0.0002) * math.sqrt(252)
    assert metrics.sharpe(rets) == pytest.approx(expected)
    assert metrics.sharpe(rets) == pytest.approx(22.4499, rel=1e-4)


def test_sharpe_zero_std_and_short_inputs_are_zero():
    assert metrics.sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0
    assert metrics.sharpe(pd.Series([0.05])) == 0.0
    assert metrics.sharpe(pd.Series([], dtype=float)) == 0.0


def test_sharpe_sign_follows_mean():
    assert metrics.sharpe(pd.Series([-0.01, -0.03])) < 0


def test_max_drawdown_known_value():
    eq = pd.Series([100.0, 120.0, 90.0, 110.0])
    # worst: peak 120 -> trough 90 = 0.25
    assert metrics.max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotonic_up_is_zero():
    assert metrics.max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_max_drawdown_takes_largest_of_several():
    eq = pd.Series([100.0, 80.0, 120.0, 60.0, 130.0, 117.0])
    # drawdowns: 20% (100->80), 50% (120->60), 10% (130->117)
    assert metrics.max_drawdown(eq) == pytest.approx(0.5)


def test_max_drawdown_is_positive_fraction_and_empty_safe():
    assert metrics.max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert metrics.max_drawdown(pd.Series([50.0, 25.0])) == pytest.approx(0.5)


def test_hit_rate_known_values():
    pairs = [
        (0.02, 0.01),    # pred up, actual up   -> hit
        (-0.01, 0.03),   # pred down, actual up -> miss
        (0.01, -0.02),   # pred up, actual down -> miss
        (-0.02, -0.01),  # pred down, actual down -> hit
        (0.005, 0.004),  # hit
    ]
    hit, base, n = metrics.hit_rate(pairs)
    assert n == 5
    assert hit == pytest.approx(3 / 5)
    # actual up in 3 of 5 pairs
    assert base == pytest.approx(3 / 5)


def test_hit_rate_empty():
    assert metrics.hit_rate([]) == (0.0, 0.0, 0)


def test_hit_rate_base_rate_is_always_up_hit_rate():
    pairs = [(0.1, -0.1), (0.1, 0.1), (0.1, -0.2), (0.1, 0.3)]
    hit, base, n = metrics.hit_rate(pairs)
    assert n == 4
    assert base == pytest.approx(0.5)
    assert hit == pytest.approx(0.5)
