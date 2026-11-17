"""W2 regression fixtures (comprehensive plan 2026-11-17, findings N1-N4).

Each test encodes the observed defect BEFORE the fix (red), then pins the
corrected behavior (green):
- N1: risk penalties must be able to remove every holding -> cash.
- N2: a first-period loss must appear in maximum drawdown.
- N3: completed-trade statistics must be quantity- and fee-aware (FIFO).
- N4: a missing mark must fall back to the last VALIDATED mark (with age),
  never silently to entry price; stale/unavailable marks block new BUYs.
"""
import numpy as np
import pandas as pd
import pytest

from quant_agent.ledger.account import Account
from quant_agent.portfolio.optimizer import PortfolioConfig, target_weights
from quant_agent.research.metrics import completed_trade_stats, summarize


def cfg(**kw):
    d = dict(edge_threshold=0.002, lambda_vol=0.0, max_names=5,
             max_weight_per_name=0.20, max_gross_exposure=0.80)
    d.update(kw)
    return PortfolioConfig(**d)


def test_n1_negative_penalized_edge_means_cash():
    """N1: mu=0.001, vol=0.1, lambda=1 -> penalized edge -0.099 for both
    names; the optimizer must hold cash, not allocate the caps."""
    mu = pd.Series({"A": 0.001, "B": 0.001})
    vol = pd.Series({"A": 0.1, "B": 0.1})
    res = target_weights(mu, vol, cfg(lambda_vol=1.0))
    assert res.weights.sum() == pytest.approx(0.0)
    assert res.diagnostics["cash_weight"] == pytest.approx(1.0)


def test_n1_penalty_can_remove_one_name_but_keep_another():
    mu = pd.Series({"GOOD": 0.05, "BAD": -0.02})
    vol = pd.Series({"GOOD": 0.10, "BAD": 0.10})
    # lambda=1: GOOD penalized 0.05-0.10=-0.05 (out), BAD -0.12 (out)
    res = target_weights(mu, vol, cfg(lambda_vol=1.0, edge_threshold=-0.04))
    # both penalized edges below the (negative) threshold -> cash
    assert res.weights.sum() == pytest.approx(0.0)
    # lower lambda keeps GOOD in: penalized 0.05-0.03=0.02 > -0.04, while BAD
    # penalized -0.05 < -0.04 stays out
    res2 = target_weights(mu, vol, cfg(lambda_vol=0.3, edge_threshold=-0.04))
    assert res2.weights["GOOD"] > 0
    assert res2.weights["BAD"] == pytest.approx(0.0)


def test_n2_first_period_loss_visible_in_drawdown():
    """N2: summarize([-0.1, 0.0]) must report -10% maximum drawdown, not 0."""
    out = summarize(pd.Series([-0.1, 0.0]))
    assert out["max_drawdown"] == pytest.approx(-0.10)
    out2 = summarize(pd.Series([0.0, -0.1]))
    assert out2["max_drawdown"] == pytest.approx(-0.10)  # same either way


def test_n2_drawdown_unchanged_when_loss_is_later():
    out = summarize(pd.Series([0.1, -0.2, 0.0]))
    # peak 1.1 -> trough 0.88 -> -20%
    assert out["max_drawdown"] == pytest.approx(-0.20)


def _fills_frame(rows):
    return pd.DataFrame(rows, columns=["session", "symbol", "side", "qty",
                                       "price", "fees"])


def test_n3_partial_sale_net_of_allocated_fees_is_a_loss():
    """Buy 100 @10 (fee 5), sell 1 @11 (fee 5): the completed 1-share trade
    loses money after allocated fees; it is a partial exit, not a 10% win."""
    fills = _fills_frame([
        ("2026-01-05", "A", "BUY", 100.0, 10.0, 5.0),
        ("2026-01-06", "A", "SELL", 1.0, 11.0, 5.0),
    ])
    out = completed_trade_stats(fills)
    # buy unit cost 10.05; sell proceeds per share 11 - 5 = 6 -> pnl -4.05
    assert out["realized_pnl"] == pytest.approx(-4.05)
    assert out["n_partial_exits"] == 1
    assert out["n_completed_positions"] == 0  # 99 shares remain open
    assert out["trade_win_rate"] == pytest.approx(0.0)  # the exit was a loss


def test_n3_full_round_trip_win_with_fees():
    fills = _fills_frame([
        ("2026-01-05", "A", "BUY", 100.0, 10.0, 5.0),
        ("2026-01-06", "A", "SELL", 100.0, 11.0, 5.0),
    ])
    out = completed_trade_stats(fills)
    # pnl = 100*11 - 5 - (100*10 + 5) = 1095 - 1005 = 90
    assert out["realized_pnl"] == pytest.approx(90.0)
    assert out["n_completed_positions"] == 1
    assert out["trade_win_rate"] == pytest.approx(1.0)


def test_n3_fifo_across_multiple_buys():
    fills = _fills_frame([
        ("2026-01-05", "A", "BUY", 100.0, 10.0, 0.0),
        ("2026-01-06", "A", "BUY", 100.0, 12.0, 0.0),
        ("2026-01-07", "A", "SELL", 150.0, 11.0, 0.0),
    ])
    out = completed_trade_stats(fills)
    # FIFO: 100 @10 then 50 @12; pnl = 150*11 - (100*10 + 50*12) = 1650-1600 = 50
    assert out["realized_pnl"] == pytest.approx(50.0)
    # the sell fully consumed the first lot and part of the second: a partial
    # exit at the symbol level, leaving 50 shares open
    assert out["n_partial_exits"] == 1
    assert out["n_completed_positions"] == 0  # 50 shares @12 still open
    assert out["open_qty"] == pytest.approx(50.0)


def test_n4_missing_mark_uses_last_validated_mark():
    """N4: a suspension (no bar) must value the position at the last
    validated mark (12), not revert to entry price (10)."""
    acct = Account(1_000.0, currency="CNY", t_plus_1=False)
    acct.apply_fill(symbol="A", side="BUY", qty=100.0, price=10.0, fees=0.0,
                    session="2026-01-05", order_id="o1")
    acct.apply_mark("A", 12.0, session="2026-01-06")
    # session 2026-01-07: no bar for A
    eq = acct.equity({}, session="2026-01-07")
    assert eq == pytest.approx(0.0 + 100.0 * 12.0)  # cash 0, valued at last mark
    assert acct.stale_marks("2026-01-07", max_age_sessions=0) == ["A"]


def test_n4_mark_refresh_clears_staleness():
    acct = Account(1_000.0, currency="CNY", t_plus_1=False)
    acct.apply_fill(symbol="A", side="BUY", qty=100.0, price=10.0, fees=0.0,
                    session="2026-01-05", order_id="o1")
    acct.apply_mark("A", 12.0, session="2026-01-06")
    assert acct.stale_marks("2026-01-08", max_age_sessions=1) == ["A"]
    acct.apply_mark("A", 12.5, session="2026-01-08")
    assert acct.stale_marks("2026-01-08", max_age_sessions=1) == []
    eq = acct.equity({}, session="2026-01-09")  # still no bar on the 9th
    assert eq == pytest.approx(100.0 * 12.5)
