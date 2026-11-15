"""H5 monitor pure functions (NO network): anomaly rules, cluster collapse,
correlation windows, dual-requirement flagging, JSON contract."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.research.info_monitor import (collapse_clusters,
                                               correlate_stock_news,
                                               detect_amount_anomalies,
                                               emit_report,
                                               _headline_materiality)


def make_series(n=60, spike_at=50, spike_mult=8.0, base=1_000_000.0):
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(7)
    vals = base * (1 + rng.normal(0, 0.05, n))
    if spike_at is not None and spike_at < n:
        vals[spike_at] = base * spike_mult
    return pd.Series(vals, index=dates)


def test_robust_z_rule_fires_on_single_spike():
    anom = detect_amount_anomalies(make_series(spike_mult=10.0))
    assert len(anom) >= 1
    hit = anom[anom["rule"].str.contains("robust_z")]
    assert len(hit) >= 1
    assert hit["z"].max() >= 3.0


def test_rate_rule_fires_on_day_over_day_jump():
    dates = pd.bdate_range("2025-01-01", periods=40)
    vals = [1_000_000.0] * 39 + [5_000_000.0]  # 400% d-o-d jump
    anom = detect_amount_anomalies(pd.Series(vals, index=dates))
    assert anom.iloc[-1]["rule"].find("rate") >= 0
    assert anom.iloc[-1]["rate"] == pytest.approx(4.0)


def test_min_baseline_guard_never_flags_early_rows():
    s = make_series(n=12, spike_at=3, spike_mult=20.0)  # spike before baseline
    assert detect_amount_anomalies(s).empty


def test_insufficient_history_returns_empty():
    s = make_series(n=5)
    assert detect_amount_anomalies(s).empty


def test_collapse_clusters_merges_consecutive_sessions():
    dates = pd.bdate_range("2025-01-01", periods=40)
    vals = [1_000_000.0] * 30 + [5_000_000.0, 6_000_000.0, 5_500_000.0] + [1_000_000.0] * 7
    anom = detect_amount_anomalies(pd.Series(vals, index=dates))
    collapsed = collapse_clusters(anom, dates)
    assert len(collapsed) == 1  # 3-day burst -> one event
    assert collapsed.iloc[0]["date"] == dates[30]


def test_correlate_stock_news_daily_window():
    anom_date = pd.Timestamp("2025-03-05")
    news = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-03-05 09:00", "2025-03-08 09:00"]),
        "headline": ["same-day item", "too-late item"],
        "content": ["", ""], "source": ["x", "x"], "url": ["", ""],
    })
    matched = correlate_stock_news(anom_date, news, next_session=None)
    assert len(matched) == 1
    assert matched.iloc[0]["headline"] == "same-day item"


def test_emit_report_enforces_exact_schema():
    report = emit_report(
        ["step1", "step2"],
        [{"timestamp": "2025-03-05", "deal_amount": "100 CNY",
          "rate_of_change": "+400%/d", "relevant_news": "headline",
          "rationale": "per steps 1-2"}])
    assert list(report) == ["reasoning_steps", "critical_information_points"]
    assert set(report["critical_information_points"][0]) == {
        "timestamp", "deal_amount", "rate_of_change", "relevant_news", "rationale"}


def test_emit_report_rejects_schema_violation():
    with pytest.raises(AssertionError):
        emit_report(["s"], [{"timestamp": "x", "extra": 1}])


def test_cctv_materiality_keyword_classes():
    # Functional Chinese test inputs: the materiality matcher operates on
    # Chinese-language news text by contract; inputs below are test data.
    hits = _headline_materiality("央行宣布降低存款准备金率并引导利率下行")
    assert "monetary" in hits
    assert _headline_materiality("天气预报：明日有雨") == []
