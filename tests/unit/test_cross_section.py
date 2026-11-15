"""Cross-section study helpers (NO network): NW t-stat, winsorize/z-score,
factor row construction with as-of valuation merge, coverage gates."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.research.cross_section import (_factor_rows, _technical_factor_rows,
                                                _winsorize_zscore, newey_west_t,
                                                REBALANCE_SESSIONS,
                                                MIN_HISTORY_SESSIONS,
                                                MIN_HISTORY_TECHNICAL)


def test_newey_west_t_on_white_noise_is_bounded():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 200))
    assert abs(newey_west_t(x)) < 3.0


def test_newey_west_t_handles_serial_dependence():
    # block structure: overlapping information -> NW t differs from naive t
    rng = np.random.default_rng(1)
    blocks = np.repeat(rng.normal(0.2, 0.3, 40), 5)  # 40 independent draws
    x = pd.Series(blocks)
    nw = newey_west_t(x, lag=3)
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert np.isfinite(nw)
    # naive t overstates precision when draws are serially dependent
    assert abs(nw) < abs(naive)


def test_winsorize_zscore_clips_outliers():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0, 1, 200))
    s.iloc[0] = 50.0  # one extreme outlier among a realistic cross-section
    z = _winsorize_zscore(s)
    assert len(z) == len(s)
    # after 1%/99% winsorize, no point dominates the cross-section
    assert z.abs().max() < 3.5
    # the outlier is pulled to the winsorized boundary
    assert z.iloc[0] == pytest.approx(z.abs().max())


def test_winsorize_zscore_insufficient_cross_section_is_empty():
    assert _winsorize_zscore(pd.Series([1.0, 2.0])).empty
    assert _winsorize_zscore(pd.Series([5.0] * 50)).empty  # zero dispersion


def _make_frames(n_sessions=320, symbols=("ALPHA", "BETA", "GAMMA"), seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_sessions)
    bars, valuation = {}, {}
    for k, sym in enumerate(symbols):
        drift = 0.0002 * (k + 1)
        close = 20.0 * (1 + drift) ** np.arange(n_sessions) * \
            np.exp(np.cumsum(rng.normal(0, 0.008, n_sessions)))
        high = close * 1.01
        low = close * 0.99
        open_ = close * (1 + rng.normal(0, 0.002, n_sessions))
        open_ = np.clip(open_, low, high)
        bars[sym] = pd.DataFrame({
            "date": dates, "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.integers(1e6, 5e6, n_sessions).astype(float),
            "adj_factor": 1.0,
            "open_adj": open_, "high_adj": high, "low_adj": low, "close_adj": close,
        })
        val_index = dates[::2]  # ~2-calendar-day sampling like the provider
        valuation[sym] = pd.DataFrame({
            "date": val_index,
            "市盈率(TTM)": 10.0 + k * 3 + rng.normal(0, 0.5, len(val_index)),
            "市净率": 1.0 + 0.5 * k + rng.normal(0, 0.05, len(val_index)),
            "总市值": 100.0 + 50 * k + rng.normal(0, 2, len(val_index)),
        }).set_index("date")  # date-indexed, matching _load_xs_frames output
    return {"bars": bars, "valuation": valuation}


def test_factor_rows_coverage_gate_and_values():
    frames = _make_frames()
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames["bars"].values()]).unique()))
    rebalance = [sessions[i] for i in range(MIN_HISTORY_SESSIONS + 5,
                                            len(sessions) - REBALANCE_SESSIONS,
                                            REBALANCE_SESSIONS)]
    rows = _factor_rows(frames, sessions, rebalance)
    assert len(rows) > 0
    factor_cols = ["EP", "BP", "size", "mom_1m", "mom_12m_1m", "lowvol",
                   "reversal_5", "turnover"]
    assert rows["fwd_ret"].notna().all()
    assert rows[factor_cols].notna().all().all()
    # EP = 1/PE, BP = 1/PB exactly for one known (date, symbol)
    r = rows.iloc[0]
    val = frames["valuation"][r["symbol"]]
    window = val.loc[: r["date"] + pd.Timedelta(days=5)].tail(1)
    assert r["EP"] == pytest.approx(1.0 / float(window["市盈率(TTM)"].iloc[0]), abs=1e-9)
    assert r["BP"] == pytest.approx(1.0 / float(window["市净率"].iloc[0]), abs=1e-9)


def test_factor_rows_excludes_short_history_symbols():
    frames = _make_frames()
    frames["bars"]["GAMMA"] = frames["bars"]["GAMMA"].iloc[:100]  # below gate
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames["bars"].values()]).unique()))
    rebalance = [sessions[i] for i in range(MIN_HISTORY_SESSIONS + 5,
                                            len(sessions) - REBALANCE_SESSIONS,
                                            REBALANCE_SESSIONS)]
    rows = _factor_rows(frames, sessions, rebalance)
    assert "GAMMA" not in set(rows["symbol"])


def test_as_of_valuation_uses_latest_within_gap_window():
    frames = _make_frames()
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames["bars"].values()]).unique()))
    rebalance = [sessions[MIN_HISTORY_SESSIONS + 1]]
    rows = _factor_rows(frames, sessions, rebalance)
    assert len(rows) == 3
    r = rows.iloc[0]
    val = frames["valuation"][r["symbol"]]
    asof = val.loc[: r["date"] + pd.Timedelta(days=5)].tail(1)
    assert (r["date"] - asof.index[0]).days <= 5


TECH_FEATURES = ["ret_1", "ret_20", "close_over_ma20", "rsi14", "vol_20"]


def test_technical_factor_rows_values_and_gate():
    frames = _make_frames()
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames["bars"].values()]).unique()))
    rebalance = [sessions[i] for i in range(MIN_HISTORY_TECHNICAL + 5,
                                            len(sessions) - REBALANCE_SESSIONS,
                                            REBALANCE_SESSIONS)]
    rows = _technical_factor_rows(frames, sessions, rebalance, TECH_FEATURES)
    assert len(rows) > 0
    assert rows["fwd_ret"].notna().all()
    assert rows[TECH_FEATURES].notna().all().all()
    # feature value matches the shared library for one known (date, symbol)
    from quant_agent.features.indicators import compute_features
    r = rows.iloc[0]
    df = frames["bars"][r["symbol"]]
    ft = compute_features(df, ["rsi14"]).set_index(df["date"])
    assert r["rsi14"] == pytest.approx(float(ft.loc[r["date"], "rsi14"]), abs=1e-9)


def test_technical_factor_rows_excludes_warmup_and_short_history():
    frames = _make_frames()
    frames["bars"]["GAMMA"] = frames["bars"]["GAMMA"].iloc[:50]  # below 61
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames["bars"].values()]).unique()))
    rebalance = [sessions[i] for i in range(MIN_HISTORY_TECHNICAL + 5,
                                            len(sessions) - REBALANCE_SESSIONS,
                                            REBALANCE_SESSIONS)]
    rows = _technical_factor_rows(frames, sessions, rebalance, TECH_FEATURES)
    assert "GAMMA" not in set(rows["symbol"])
