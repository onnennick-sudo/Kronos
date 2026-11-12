"""Kronos forecasting wrapper: Monte-Carlo path sampling and path statistics.

The torch/model imports happen only inside the loader, so importing this module
(and running it with a fake predictor) does not require weights or torch.
"""
from __future__ import annotations

import logging
import math
import random
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

from bot import config

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def path_stats(final_closes, last_close: float) -> tuple[float, float]:
    """Return (r_med, agree) for sampled final closes.

    r = final / last_close - 1 per path; r_med = median(r);
    agree = share of paths whose sign matches sign(r_med).
    If r_med == 0, returns (0.0, 0.0).
    """
    finals = np.asarray(final_closes, dtype=float)
    if finals.size == 0:
        raise ValueError("no paths")
    r = finals / float(last_close) - 1.0
    r_med = float(np.median(r))
    if r_med == 0.0:
        return 0.0, 0.0
    agree = float(np.mean(np.sign(r) == np.sign(r_med)))
    return r_med, agree


def seed_for(dt, kind: str) -> int:
    """Deterministic seed: daily -> YYYYMMDD, hourly -> YYYYMMDDHH."""
    if kind == "daily":
        return int(dt.strftime("%Y%m%d"))
    if kind == "hourly":
        return int(dt.strftime("%Y%m%d%H"))
    raise ValueError(f"unknown kind: {kind!r}")


def _from_pretrained(cls, name: str):
    try:
        return cls.from_pretrained(name, revision=config.MODEL_REVISION)
    except TypeError:
        return cls.from_pretrained(name)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # Seed torch only if it is already loaded (a real predictor imports it); do not import it here.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


class KronosForecaster:
    def __init__(self, predictor=None):
        self._predictor = predictor

    def _load_predictor(self):
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor

        torch.set_num_threads(config.TORCH_THREADS)
        tokenizer = _from_pretrained(KronosTokenizer, config.TOKENIZER_NAME)
        model = _from_pretrained(Kronos, config.MODEL_NAME)
        return KronosPredictor(model, tokenizer, device="cpu", max_context=config.MAX_CONTEXT)

    def forecast(self, series: dict[str, pd.DataFrame], future: dict[str, pd.Series],
                 pred_len: int, n_paths: int, lookback: int, seed: int) -> dict[str, dict]:
        """Sample n_paths futures per symbol. Per-symbol failures become {"error": str}."""
        if pred_len < 1 or n_paths < 1 or lookback < 1:
            raise ValueError("pred_len, n_paths and lookback must be >= 1")
        if self._predictor is None:
            self._predictor = self._load_predictor()
        _seed_all(seed)

        results: dict[str, dict] = {}
        for sym, df in series.items():
            try:
                results[sym] = self._forecast_one(df, future.get(sym), pred_len, n_paths, lookback)
            except Exception as exc:  # isolate per-symbol failures
                log.warning("forecast failed for %s: %s", sym, exc)
                results[sym] = {"error": f"{type(exc).__name__}: {exc}"}
        return results

    def _forecast_one(self, df: pd.DataFrame, y_stamps, pred_len: int, n_paths: int, lookback: int) -> dict:
        if y_stamps is None:
            raise ValueError("no future timestamps for symbol")
        if len(df) < lookback:
            raise ValueError(f"need {lookback} rows, have {len(df)}")
        hist = df.tail(lookback)
        if hist[OHLCV].isna().to_numpy().any():
            raise ValueError("NaN in OHLCV history")

        x_df = hist[OHLCV].astype(float).reset_index(drop=True)
        x_ts = pd.to_datetime(hist["timestamps"]).reset_index(drop=True)
        if x_ts.dt.tz is not None:
            raise ValueError("timestamps must be tz-naive ET")
        y_ts = pd.Series(pd.to_datetime(y_stamps)).reset_index(drop=True)
        if len(y_ts) != pred_len:
            raise ValueError(f"future has {len(y_ts)} stamps, need pred_len={pred_len}")

        last_close = float(hist["close"].iloc[-1])
        if not (math.isfinite(last_close) and last_close > 0):
            raise ValueError(f"invalid last close {last_close}")

        finals: list[float] = []
        for start in range(0, n_paths, config.BATCH_SIZE):
            k = min(config.BATCH_SIZE, n_paths - start)
            preds = self._predictor.predict_batch(
                [x_df] * k, [x_ts] * k, [y_ts] * k, pred_len,
                T=config.SAMPLE_T, top_k=config.SAMPLE_TOP_K, top_p=config.SAMPLE_TOP_P,
                sample_count=1, verbose=False,
            )
            if len(preds) != k:
                raise RuntimeError(f"predict_batch returned {len(preds)} results for {k} inputs")
            finals.extend(float(p["close"].iloc[-1]) for p in preds)

        if not all(math.isfinite(v) for v in finals):
            raise ValueError("non-finite forecast")
        r_med, agree = path_stats(finals, last_close)
        return {
            "r_med": r_med,
            "agree": agree,
            "n": len(finals),
            "last_close": last_close,
            "final_closes": finals,
        }
