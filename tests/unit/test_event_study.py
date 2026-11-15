"""H6 forward validation pure functions (NO network)."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.research.event_study import (classify_materiality,
                                              daily_aggregate, event_outcomes,
                                              nw_diff_t)


def make_bar_frame(n=80, seed=1):
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({
        "date": dates, "open_adj": open_, "close_adj": close,
        "amount": np.full(n, 1_000_000.0) * (1 + rng.normal(0, 0.05, n)),
    }, index=None)


def test_classify_materiality_tiers():
    # Functional Chinese test inputs: the classifier matches Chinese
    # headlines by contract; inputs below are test data.
    assert classify_materiality("公司公告：股东增持1.2%股份") == "high"
    assert classify_materiality("股东减持计划公告") == "high"
    assert classify_materiality("交通运输行业资金流出榜：多股净流出居前") == "low"
    assert classify_materiality("机构近期密集调研该公司") == "medium"
    assert classify_materiality("今日天气") == "low"


def test_boilerplate_beats_medium():
    # a fund-flow ranking headline mentioning institutions stays LOW
    assert classify_materiality("资金流出榜：机构净卖出这些股") == "low"


def test_event_outcomes_no_lookahead_and_values():
    df = make_bar_frame(80)
    sessions = pd.DatetimeIndex(df["date"])
    spike_date = sessions[50]
    df.loc[df["date"] == spike_date, "amount"] = 8_000_000.0
    anom = detect_like(sessions, df, spike_date)
    events = event_outcomes({"S": df}, {"S": anom}, sessions)
    assert len(events) == 1
    ev = events.iloc[0]
    close = df.set_index("date")["close_adj"]
    t, t1 = spike_date, sessions[51]
    assert ev["reaction_return"] == pytest.approx(close.loc[t1] / close.loc[t] - 1.0)
    assert ev["persistence"] == pytest.approx(
        df.set_index("date")["amount"].loc[t1]
        / df.set_index("date")["amount"].rolling(30).median().loc[t])
    assert 0 < ev["persistence"] < 3  # next-day amount back to normal levels


def detect_like(sessions, df, spike_date):
    """Minimal anomaly frame for the spike session (mirrors the detector)."""
    return pd.DataFrame([{"date": spike_date,
                          "amount": float(df.loc[df["date"] == spike_date,
                                                 "amount"].iloc[0]),
                          "z": 5.0, "rate": 7.0, "rule": "robust_z"}])


def test_event_outcomes_drops_tail_events_without_next_session():
    df = make_bar_frame(80)
    sessions = pd.DatetimeIndex(df["date"])
    last = sessions[-1]
    anom = detect_like(sessions, df, last)
    events = event_outcomes({"S": df}, {"S": anom}, sessions)
    assert events.empty  # no next session -> outcome undefined, dropped


def test_daily_aggregate_and_nw_diff():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2025-01-01", periods=40)
    a = pd.DataFrame({"date": np.repeat(dates[:30], 3),
                      "v": rng.normal(0.01, 0.01, 90)})
    b = pd.DataFrame({"date": np.repeat(dates[:30], 3),
                      "v": rng.normal(-0.005, 0.01, 90)})
    t = nw_diff_t(None, "v", a, b)
    assert np.isfinite(t)
