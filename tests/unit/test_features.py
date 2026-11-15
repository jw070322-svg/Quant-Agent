"""Feature-library invariants: causality, edge cases, shared implementation."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.features.indicators import (FEATURE_NAMES, FEATURE_NAMES_ALL,
                                             FEATURE_NAMES_V2, compute_features)
from quant_agent.features.pipeline import assert_prefix_causality, build_symbol_frame


def make_bars(n=400, seed=7, flat=False, zero_volume=False):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, size=n)
    if flat:
        steps = np.zeros(n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.minimum(high, np.maximum(low, close * (1 + rng.normal(0, 0.002, n))))
    vol = np.zeros(n) if zero_volume else rng.integers(1e6, 5e6, n).astype(float)
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "adj_factor": 1.0,
        "open_adj": open_, "high_adj": high, "low_adj": low, "close_adj": close,
    })


def test_prefix_causality_random_walk():
    df = make_bars()
    # appending future rows must never change past feature values
    assert_prefix_causality(df, FEATURE_NAMES_ALL)


def test_v2_pack_warmup_is_nan_then_finite():
    df = make_bars()
    feats = compute_features(df, FEATURE_NAMES_V2)
    # long-warmup factors: defined only after their full lookback
    assert feats["mom_12_1"].iloc[:239].isna().all()
    assert np.isfinite(feats["mom_12_1"].iloc[240])
    assert feats["dist_52w_high"].iloc[:250].isna().all()
    assert np.isfinite(feats["dist_52w_high"].iloc[251])
    # every v2 factor: unknown early (NaN, not 0), finite after full warm-up
    for name in FEATURE_NAMES_V2:
        assert feats[name].iloc[:10].isna().all(), name
        assert np.isfinite(feats[name].iloc[300]), name


def test_amihud_is_finite_with_positive_volume_and_nan_with_zero():
    df = make_bars()
    feats = compute_features(df, ["amihud_20"])
    assert np.isfinite(feats["amihud_20"].iloc[252])
    zero = make_bars(zero_volume=True)
    feats0 = compute_features(zero, ["amihud_20"])
    assert feats0["amihud_20"].iloc[252:].isna().all()


def test_warmup_is_nan_not_zero():
    df = make_bars()
    feats = compute_features(df, ["close_over_ma60", "ret_1", "macd_norm"])
    assert feats["close_over_ma60"].iloc[:59].isna().all()
    assert np.isfinite(feats["close_over_ma60"].iloc[59])
    assert feats["ret_1"].iloc[0] != 0 or pd.isna(feats["ret_1"].iloc[0])
    assert pd.isna(feats["ret_1"].iloc[0])  # unknown, never fabricated 0


def test_rsi_rising_series_is_not_oversold():
    # legacy bug: zero avg_loss -> rs=0 -> RSI 0 (oversold) on a rising series
    n = 60
    close = np.linspace(100, 160, n)
    df = make_bars(n)
    df["close"] = close
    df["close_adj"] = close
    rsi = compute_features(df, ["rsi14"])["rsi14"]
    assert rsi.iloc[-1] == 100.0  # rising series is maximally strong, not oversold


def test_rsi_flat_series_is_neutral():
    df = make_bars(80, flat=True)
    rsi = compute_features(df, ["rsi14"])["rsi14"]
    assert rsi.iloc[-1] == 50.0


def test_zero_volume_series_yields_nan_ratio_not_inf():
    df = make_bars(100, zero_volume=True)
    out = compute_features(df, ["volume_ratio20"])
    assert out["volume_ratio20"].iloc[30:].isna().all()


def test_flat_price_series_features_are_finite():
    df = make_bars(400, flat=True)
    out = compute_features(df, FEATURE_NAMES_ALL)
    tail = out.iloc[300:]
    # zero dispersion: band WIDTH is genuinely 0.0 (real information), while
    # band POSITION 0/0 is undefined -> NaN, never a fabricated 0/1
    assert tail["bb_pos20"].isna().all()
    assert (tail["bb_width20"] == 0.0).all()
    # zero-variance returns make band position, distribution shape, and the
    # return/volume correlation undefined -> NaN (never fabricated constants)
    assert tail[["bb_pos20", "skew_20", "kurt_20", "corr_pv_20"]].isna().all().all()
    others = [c for c in FEATURE_NAMES_ALL
              if c not in ("bb_pos20", "skew_20", "kurt_20", "corr_pv_20")]
    assert np.isfinite(tail[others].to_numpy(dtype=float)).all()


def test_build_symbol_frame_carries_executable_columns():
    df = make_bars()
    sf = build_symbol_frame(df, ["ret_1", "close_over_ma20"])
    for col in ("date", "open_adj", "high_adj", "low_adj", "close_adj",
                "volume", "adj_factor", "ret_1", "close_over_ma20"):
        assert col in sf.columns
