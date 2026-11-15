"""Data validation: OHLC quarantine matches audit evidence; volume units are
inferred from data, not assumed; schemas are normalized deterministically."""
import numpy as np
import pandas as pd

from quant_agent.data.ingest import (_normalize_simple, _normalize_tushare,
                                     _resolve_volume_unit, detect_schema)
from quant_agent.data.schema import (QUARANTINE_OHLC_INCONSISTENT, validate_bars)


def _simple_df(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low",
                                       "close", "volume"])


def test_ohlc_inconsistent_rows_quarantined_with_reasons():
    # reproduces the audit's AMZN/GOOGL defect class: open outside [low, high]
    df = _simple_df([
        ("2026-02-03", "X", 10.0, 10.5, 9.8, 10.2, 1000.0),
        ("2026-02-04", "X", 238.86, 238.77, 231.83, 235.0, 2000.0),  # open > high
        ("2026-02-05", "X", 11.0, 11.4, 10.9, 11.2, 1500.0),
    ])
    df["adj_factor"] = 1.0
    df["adjustment_status"] = "unknown_no_factor"
    df["date"] = pd.to_datetime(df["date"])
    clean, quarantine, review = validate_bars(df)
    assert len(clean) == 2
    assert len(quarantine) == 1
    assert quarantine[0]["date"] == "2026-02-04"
    assert QUARANTINE_OHLC_INCONSISTENT in quarantine[0]["reasons"]
    # silently clipped repair is forbidden: the row is gone, not fixed
    assert not (clean["open"] > clean["high"]).any()


def test_duplicate_dates_keep_last_and_quarantine_earlier():
    df = _simple_df([
        ("2026-01-05", "X", 10.0, 10.5, 9.8, 10.2, 1000.0),
        ("2026-01-06", "X", 11.0, 11.5, 10.8, 11.2, 1100.0),
        ("2026-01-06", "X", 11.2, 11.6, 11.0, 11.4, 1200.0),  # vendor correction
    ])
    df["adj_factor"] = 1.0
    df["adjustment_status"] = "unknown_no_factor"
    df["date"] = pd.to_datetime(df["date"])
    clean, quarantine, review = validate_bars(df)
    assert len(clean) == 2
    assert clean["date"].is_unique
    assert clean["close"].iloc[-1] == 11.4  # last occurrence wins


def test_extreme_move_is_flagged_not_dropped():
    df = _simple_df([
        ("2026-01-05", "X", 10.0, 10.5, 9.8, 10.2, 1000.0),
        ("2026-01-06", "X", 14.0, 15.0, 13.8, 14.8, 9000.0),  # +45%: review, keep
    ])
    df["adj_factor"] = 1.0
    df["adjustment_status"] = "unknown_no_factor"
    df["date"] = pd.to_datetime(df["date"])
    clean, quarantine, review = validate_bars(df)
    assert len(clean) == 2  # retained in the clean frame for reconciliation
    assert len(review) == 1 and review[0]["date"] == "2026-01-06"
    assert any("extreme" in r for r in review[0]["reasons"])


def test_tushare_cn_volume_unit_resolved_as_lots():
    # Tushare CN convention: volume in lots of 100 shares, amount in thousands of CNY
    n = 30
    rng = np.random.default_rng(3)
    close = np.full(n, 10.0) + rng.normal(0, 0.05, n)
    vol_hand = rng.uniform(5e5, 1e6, n)          # lots of 100 shares
    amount_k = vol_hand * 100 * close / 1000.0   # thousands of CNY
    df = pd.DataFrame({
        "trade_date": pd.date_range("2025-01-01", periods=n).strftime("%Y%m%d"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "vol": vol_hand, "amount": amount_k,
        "adj_factor": 1.0,
    })
    norm = _normalize_tushare(df, "000001.SZ")
    norm, unit, amount_unit = _resolve_volume_unit(norm)
    assert unit == "lots_x100_converted"
    assert amount_unit == "thousands"
    # converted share volume tracks amount/close within tolerance
    implied = norm["amount"] * 1000 / norm["volume"]
    assert (np.abs(implied / norm["close"] - 1) < 0.05).all()


def test_integer_dates_never_parsed_as_epoch_offsets():
    df = _simple_df([])
    df = pd.DataFrame({
        "trade_date": [20250101, 20250102],
        "open": [10.0, 10.1], "high": [10.2, 10.3], "low": [9.9, 10.0],
        "close": [10.1, 10.2], "vol": [1000.0, 1100.0], "adj_factor": 1.0,
        "amount": [np.nan, np.nan],
    })
    norm = _normalize_tushare(df, "X")
    assert norm["date"].iloc[0] == pd.Timestamp("2025-01-01")
    assert norm["date"].iloc[1] == pd.Timestamp("2025-01-02")


def test_schema_detection():
    assert detect_schema(["date", "symbol", "open", "high", "low", "close", "volume"]) == "simple"
    assert detect_schema(["ts_code", "trade_date", "open", "close", "vol"]) == "tushare"
    import pytest
    with pytest.raises(ValueError):
        detect_schema(["foo", "bar"])
