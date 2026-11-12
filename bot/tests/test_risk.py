from bot import config, risk


def test_is_halted_exact_boundary_is_halted():
    last = 1000.0
    equity = last * 0.98
    assert risk.is_halted(equity, last) is True


def test_is_halted_just_above_boundary_is_not_halted():
    last = 1000.0
    assert risk.is_halted(last * 0.98 + 0.01, last) is False


def test_is_halted_below_boundary_is_halted():
    assert risk.is_halted(900.0, 1000.0) is True


def test_is_halted_respects_config(monkeypatch):
    monkeypatch.setattr(config, "DAILY_HALT_FRAC", 0.05)
    assert risk.is_halted(950.0, 1000.0) is True
    assert risk.is_halted(951.0, 1000.0) is False


def test_size_qty_basic(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION_FRAC", 0.20)
    assert risk.size_qty(10000.0, 50.0) == 40
    assert risk.size_qty(10000.0, 33.0) == 60  # 2000 / 33 = 60.6 -> 60


def test_size_qty_subtracts_existing_value(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION_FRAC", 0.20)
    assert risk.size_qty(10000.0, 50.0, existing_value=1000.0) == 20


def test_size_qty_floors_at_zero(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION_FRAC", 0.20)
    assert risk.size_qty(10000.0, 50.0, existing_value=5000.0) == 0
    assert risk.size_qty(10000.0, 5000.0) == 0  # price too high for one share
    assert risk.size_qty(10000.0, 0.0) == 0  # non-positive price is never sized
