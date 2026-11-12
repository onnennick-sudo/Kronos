import sys
import types
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from bot import config
from bot.forecast import KronosForecaster, path_stats, seed_for


# --- path_stats ---

def test_path_stats_mixed_signs():
    # r = [0.10, 0.05, -0.05] -> median 0.05; two of three agree in sign
    r_med, agree = path_stats([110.0, 105.0, 95.0], 100.0)
    assert r_med == pytest.approx(0.05)
    assert agree == pytest.approx(2 / 3)


def test_path_stats_all_negative():
    r_med, agree = path_stats([90.0, 95.0, 80.0], 100.0)
    assert r_med == pytest.approx(-0.10)
    assert agree == pytest.approx(1.0)


def test_path_stats_even_count_median_and_agreement():
    # r = [0.1, 0.2, -0.1, -0.05] -> median (-0.05 + 0.1)/2 = 0.025; 2 of 4 positive
    r_med, agree = path_stats([1.1, 1.2, 0.9, 0.95], 1.0)
    assert r_med == pytest.approx(0.025)
    assert agree == pytest.approx(0.5)


def test_path_stats_zero_median_gives_zero_agreement():
    r_med, agree = path_stats([100.0, 100.0, 110.0, 90.0], 100.0)
    assert r_med == 0.0
    assert agree == 0.0


def test_path_stats_flat_path_disagrees_with_positive_median():
    # r = [0.1, 0.0, 0.2] -> median 0.1; the flat path does not share the sign
    r_med, agree = path_stats([110.0, 100.0, 120.0], 100.0)
    assert r_med == pytest.approx(0.1)
    assert agree == pytest.approx(2 / 3)


def test_path_stats_rejects_empty():
    with pytest.raises(ValueError):
        path_stats([], 100.0)


# --- seed_for ---

def test_seed_for_daily_and_hourly_formats():
    assert seed_for(date(2026, 11, 12), "daily") == 20261112
    assert seed_for(datetime(2026, 11, 12, 11, 20), "hourly") == 2026111211
    assert seed_for(datetime(2026, 1, 3, 0, 0), "hourly") == 2026010300


def test_seed_for_rejects_unknown_kind():
    with pytest.raises(ValueError):
        seed_for(date(2026, 11, 12), "weekly")


# --- fake predictor and helpers ---

class FakePredictor:
    """Deterministic stand-in for KronosPredictor.predict_batch.

    final close = input last close * factors.get(last close, 1.0), optionally perturbed
    with numpy noise (so seeding is observable). Raises for inputs whose last close is in fail_on.
    """

    def __init__(self, factors=None, noise=False, fail_on=()):
        self.factors = factors or {}
        self.noise = noise
        self.fail_on = set(fail_on)
        self.calls = []   # batch sizes per call
        self.kwargs = []
        self.seen = []    # every input DataFrame received

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len,
                      T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        self.calls.append(len(df_list))
        self.kwargs.append(dict(pred_len=pred_len, T=T, top_k=top_k, top_p=top_p,
                                sample_count=sample_count, verbose=verbose))
        out = []
        for df, xts, yts in zip(df_list, x_timestamp_list, y_timestamp_list):
            self.seen.append(df)
            last = float(df["close"].iloc[-1])
            if last in self.fail_on:
                raise RuntimeError("boom")
            factor = self.factors.get(last, 1.0)
            if self.noise:
                factor *= 1.0 + 0.05 * np.random.standard_normal()
            closes = [0.0] * (pred_len - 1) + [last * factor]
            out.append(pd.DataFrame({
                "open": closes, "high": closes, "low": closes, "close": closes,
                "volume": [0.0] * pred_len, "amount": [0.0] * pred_len,
            }, index=yts))
        return out


def make_df(n, last_close):
    ts = pd.date_range("2026-11-01 09:00", periods=n, freq="h")
    close = np.linspace(last_close - 1.0, last_close, n)
    return pd.DataFrame({
        "timestamps": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    })


def future_for(pred_len):
    return pd.Series(pd.date_range("2026-11-10 10:00", periods=pred_len, freq="h"))


# --- forecast ---

def test_forecast_duplicates_each_symbol_n_paths_times_with_lookback_slice():
    fp = FakePredictor()
    f = KronosForecaster(predictor=fp)
    out = f.forecast({"AAA": make_df(50, 100.0)}, {"AAA": future_for(4)},
                     pred_len=4, n_paths=7, lookback=30, seed=1)
    assert len(fp.seen) == 7
    assert sum(fp.calls) == 7
    for df in fp.seen:
        assert len(df) == 30
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[-1] == 100.0
    assert out["AAA"]["n"] == 7
    assert len(out["AAA"]["final_closes"]) == 7


def test_forecast_passes_config_sampling_kwargs():
    fp = FakePredictor()
    KronosForecaster(predictor=fp).forecast({"AAA": make_df(40, 50.0)}, {"AAA": future_for(3)},
                                            pred_len=3, n_paths=2, lookback=20, seed=5)
    assert fp.kwargs and all(
        k == dict(pred_len=3, T=config.SAMPLE_T, top_k=config.SAMPLE_TOP_K, top_p=config.SAMPLE_TOP_P,
                  sample_count=1, verbose=False)
        for k in fp.kwargs
    )


def test_forecast_chunks_at_batch_size(monkeypatch):
    monkeypatch.setattr(config, "BATCH_SIZE", 3)
    fp = FakePredictor()
    out = KronosForecaster(predictor=fp).forecast({"AAA": make_df(40, 100.0)}, {"AAA": future_for(2)},
                                                  pred_len=2, n_paths=7, lookback=20, seed=1)
    assert fp.calls == [3, 3, 1]
    assert out["AAA"]["n"] == 7


def test_forecast_maps_results_to_correct_symbol():
    fp = FakePredictor(factors={100.0: 1.05, 200.0: 0.90})
    series = {"AAA": make_df(50, 100.0), "BBB": make_df(50, 200.0)}
    fut = {"AAA": future_for(4), "BBB": future_for(4)}
    out = KronosForecaster(predictor=fp).forecast(series, fut, pred_len=4, n_paths=4, lookback=30, seed=1)
    assert out["AAA"]["last_close"] == 100.0
    assert out["AAA"]["r_med"] == pytest.approx(0.05)
    assert out["AAA"]["agree"] == pytest.approx(1.0)
    assert out["BBB"]["last_close"] == 200.0
    assert out["BBB"]["r_med"] == pytest.approx(-0.10)
    assert out["BBB"]["agree"] == pytest.approx(1.0)


def test_forecast_isolates_per_symbol_errors():
    fp = FakePredictor(factors={100.0: 1.02}, fail_on={300.0})
    series = {
        "GOOD": make_df(50, 100.0),
        "SHORT": make_df(10, 100.0),
        "NANS": make_df(50, 100.0).assign(close=lambda d: d["close"].where(d.index != 45, np.nan)),
        "BOOM": make_df(50, 300.0),
        "NOFUT": make_df(50, 100.0),
        "GOOD2": make_df(50, 100.0),
    }
    fut = {k: future_for(4) for k in series if k != "NOFUT"}
    out = KronosForecaster(predictor=fp).forecast(series, fut, pred_len=4, n_paths=3, lookback=30, seed=1)

    assert "need 30 rows" in out["SHORT"]["error"]
    assert "NaN" in out["NANS"]["error"]
    assert "RuntimeError" in out["BOOM"]["error"]
    assert "no future" in out["NOFUT"]["error"]
    for sym in ("GOOD", "GOOD2"):
        assert "error" not in out[sym]
        assert out[sym]["r_med"] == pytest.approx(0.02)


def test_forecast_rejects_wrong_future_length():
    out = KronosForecaster(predictor=FakePredictor()).forecast(
        {"AAA": make_df(40, 100.0)}, {"AAA": future_for(5)}, pred_len=4, n_paths=2, lookback=20, seed=1)
    assert "need pred_len=4" in out["AAA"]["error"]


def test_forecast_same_seed_reproduces_and_different_seed_differs():
    series = {"AAA": make_df(50, 100.0)}
    fut = {"AAA": future_for(3)}

    def run(seed):
        return KronosForecaster(predictor=FakePredictor(noise=True)).forecast(
            series, fut, pred_len=3, n_paths=5, lookback=30, seed=seed)["AAA"]["final_closes"]

    assert run(20261112) == run(20261112)
    assert run(20261112) != run(20261113)


# --- loader (fake `model` package, no weights) ---

def test_loader_passes_revision_and_falls_back_without_it(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    created = {}

    class _Base:
        accepts_revision = True

        @classmethod
        def from_pretrained(cls, name, **kw):
            if "revision" in kw and not cls.accepts_revision:
                raise TypeError("unexpected keyword 'revision'")
            created.setdefault(cls.__name__, []).append((name, kw))
            return cls()

        def to(self, device):
            return self

    class KronosTokenizer(_Base):
        pass

    class Kronos(_Base):
        accepts_revision = False  # forces the TypeError fallback path

    class KronosPredictor:
        def __init__(self, model, tokenizer, device=None, max_context=None):
            created["predictor"] = dict(device=device, max_context=max_context)

    fake = types.ModuleType("model")
    fake.Kronos, fake.KronosTokenizer, fake.KronosPredictor = Kronos, KronosTokenizer, KronosPredictor
    monkeypatch.setitem(sys.modules, "model", fake)

    KronosForecaster()._load_predictor()

    assert created["KronosTokenizer"] == [(config.TOKENIZER_NAME, {"revision": config.MODEL_REVISION})]
    assert created["Kronos"] == [(config.MODEL_NAME, {})]
    assert created["predictor"] == dict(device="cpu", max_context=config.MAX_CONTEXT)
