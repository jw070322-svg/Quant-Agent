"""Shared causal indicator library.

This is the ONE feature module for research, replay, paper, and live modes
(handover 8.1 principle 2). All features are:

- causal: value at row t uses only rows <= t (rolling/ewm semantics), so
  appending future data never changes past values (prefix causality, tested);
- dimensionless: nominal price scale is divided out, so pooled cross-symbol
  fitting cannot learn price-level associations (handover 4.8, 8.4);
- NaN during warm-up: unknown is NaN, never zero and never a fabricated value.

Fixes vs legacy scripts (handover 6.2): real EMA-based MACD, K/D distinct,
RSI with zero-loss handled (rising series is NOT oversold), volatility
published, and no "K and D both assigned RSV" defect by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sessions required before every registered feature is defined (max window).
# dist_52w_high needs 252 sessions; mom_12_1 needs 240.
WARMUP_SESSIONS = 252

FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_20",
    "close_over_ma5", "close_over_ma20", "close_over_ma60",
    "ma5_over_ma20",
    "macd_norm", "macd_hist_norm",
    "rsi14",
    "bb_pos20", "bb_width20",
    "vol_5", "vol_20",
    "atr14_norm",
    "volume_ratio20",
]

# Hypothesis H1 factor pack v2 (pre-registered in docs/DECISIONS.md 2.4):
# causal, dimensionless, per-symbol. Definitions are FROZEN — changes require
# a new pre-registered hypothesis.
FEATURE_NAMES_V2 = [
    "mom_12_1",            # 12-month momentum excluding the last month
    "dist_52w_high",       # distance from the 252-session high
    "skew_20",             # skewness of daily returns
    "kurt_20",             # excess kurtosis of daily returns
    "vol_of_vol_20",       # volatility of the 5-day volatility
    "amihud_20",           # |ret| per unit traded value (x1e8 scale const)
    "corr_pv_20",          # correlation of returns and volume changes
    "vol_ratio_60",        # volume relative to its 60-session mean
]

FEATURE_NAMES_ALL = FEATURE_NAMES + FEATURE_NAMES_V2


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Zero-loss guard: avg_loss == 0 & avg_gain > 0 -> 100;
    both zero (flat) -> 50. Legacy bug returned 0 for a rising series."""
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _atr_norm(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr / close


def compute_features(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Compute registered features from a normalized frame.

    Requires columns: date, close_adj (all features), high_adj/low_adj (atr),
    volume (liquidity). Uses ADJUSTED prices only, so returns are
    corporate-action consistent under the recorded adjustment status.
    """
    names = names or FEATURE_NAMES
    close = df["close_adj"].astype(float)
    high = df["high_adj"].astype(float)
    low = df["low_adj"].astype(float)
    volume = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    std20 = close.rolling(20).std(ddof=0)
    ret1 = close.pct_change(1)
    ret5 = ret1.rolling(5).std(ddof=1)
    var20 = ret1.rolling(20).std(ddof=1)
    log_vol = np.log(volume.where(volume > 0))
    dv = log_vol.diff()

    computed: dict[str, pd.Series] = {
        "ret_1": ret1,
        "ret_5": close.pct_change(5),
        "ret_20": close.pct_change(20),
        "close_over_ma5": close / ma5 - 1.0,
        "close_over_ma20": close / ma20 - 1.0,
        "close_over_ma60": close / ma60 - 1.0,
        "ma5_over_ma20": ma5 / ma20 - 1.0,
        "macd_norm": (_ema(close, 12).mean() - _ema(close, 26).mean()) / close,
        # signal line = EMA9 of the MACD line (legacy used a 9-day price mean)
        "macd_hist_norm": ((_ema(close, 12).mean() - _ema(close, 26).mean())
                           - _ema(_ema(close, 12).mean() - _ema(close, 26).mean(), 9).mean()) / close,
        "rsi14": _rsi(close),
        "bb_pos20": (close - ma20) / (2.0 * std20.replace(0.0, np.nan)),
        "bb_width20": (4.0 * std20) / close,
        "vol_5": ret5,
        "vol_20": var20,
        "atr14_norm": _atr_norm(high, low, close),
        "volume_ratio20": volume / volume.rolling(20).mean().replace(0.0, np.nan),
    }

    # ---- H1 factor pack v2 (frozen definitions; DECISIONS.md 2.4) ----------
    dollar_volume = close * volume
    computed.update({
        "mom_12_1": close.shift(20) / close.shift(240) - 1.0,
        "dist_52w_high": close / close.rolling(252).max() - 1.0,
        # zero-variance windows have an UNDEFINED distribution shape: pandas
        # would return degenerate 0/-3 constants, which we mask to NaN
        # (unknown is NaN, never a fabricated value)
        "skew_20": ret1.rolling(20).skew().where(var20 > 0),
        "kurt_20": ret1.rolling(20).kurt().where(var20 > 0),
        "vol_of_vol_20": ret5.rolling(20).std(ddof=1),
        # Amihud illiquidity, scaled by a fixed 1e8 constant so values stay
        # numerically small; the scale is a constant, not a fitted parameter
        "amihud_20": (ret1.abs() / dollar_volume.replace(0.0, np.nan)).rolling(20).mean() * 1e8,
        "corr_pv_20": ret1.rolling(20).corr(dv).where(var20 > 0),
        "vol_ratio_60": volume / volume.rolling(60).mean().replace(0.0, np.nan),
    })

    missing = [n for n in names if n not in computed]
    if missing:
        raise KeyError(f"unregistered features requested: {missing}")
    for n in names:
        out[n] = computed[n].replace([np.inf, -np.inf], np.nan)
    return out
