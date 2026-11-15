"""Portfolio optimizer: feasibility BY CONSTRUCTION — the legacy optimizer
solved a constrained problem and then rescaled it into violations
(handover 6.4). Every case here asserts the constraints survive."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.portfolio.optimizer import PortfolioConfig, target_weights
from quant_agent.risk.checks import RiskLimits, check_order_intents, check_weights


def cfg(**kw):
    d = dict(edge_threshold=0.002, lambda_vol=0.0, max_names=5,
             max_weight_per_name=0.20, max_gross_exposure=0.80)
    d.update(kw)
    return PortfolioConfig(**d)


def limits(**kw):
    d = dict(max_weight_per_name=0.20, max_gross_exposure=0.80, cash_reserve=0.05,
             max_turnover=0.50, min_order_value=1000.0,
             max_open_orders_per_symbol=1, allowed_sides=["BUY", "SELL"],
             stop_loss_pct=0.08, trail_stop_pct=0.10)
    d.update(kw)
    return RiskLimits(**d)


def test_single_candidate_gets_name_cap_not_80pct():
    # legacy defect: one valid candidate returned 80% allocation
    mu = pd.Series({"A": 0.05})
    vol = pd.Series({"A": 0.02})
    res = target_weights(mu, vol, cfg())
    assert res.weights["A"] == pytest.approx(0.20)
    assert res.weights.sum() == pytest.approx(0.20)
    assert check_weights(res.weights, limits()).approved


def test_two_candidates_capped_individually():
    mu = pd.Series({"A": 0.05, "B": 0.04})
    vol = pd.Series({"A": 0.02, "B": 0.02})
    res = target_weights(mu, vol, cfg())
    assert (res.weights <= 0.20 + 1e-12).all()
    assert res.weights.sum() <= 0.80 + 1e-12
    assert check_weights(res.weights, limits()).approved


def test_four_candidates_burn_budget_no_excess():
    mu = pd.Series({"A": 0.05, "B": 0.04, "C": 0.03, "D": 0.02})
    vol = pd.Series({"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02})
    res = target_weights(mu, vol, cfg())
    assert res.weights.sum() == pytest.approx(0.80)  # budget fully but exactly used
    assert (res.weights <= 0.20 + 1e-12).all()
    assert check_weights(res.weights, limits()).approved


def test_zero_alpha_means_cash():
    mu = pd.Series({"A": 0.001, "B": -0.01})  # both below edge threshold
    vol = pd.Series({"A": 0.02, "B": 0.02})
    res = target_weights(mu, vol, cfg())
    assert res.weights.abs().sum() == pytest.approx(0.0)
    assert check_weights(res.weights, limits()).approved


def test_tight_risk_cap_respected():
    mu = pd.Series({"A": 0.10, "B": 0.09, "C": 0.08})
    vol = pd.Series({"A": 0.02, "B": 0.02, "C": 0.02})
    tight = cfg(max_weight_per_name=0.05, max_gross_exposure=0.10)
    res = target_weights(mu, vol, tight)
    assert (res.weights <= 0.05 + 1e-12).all()
    assert res.weights.sum() <= 0.10 + 1e-12
    assert check_weights(res.weights, limits(max_weight_per_name=0.05,
                                             max_gross_exposure=0.10)).approved


def test_ranking_prefers_edge_per_unit_risk():
    mu = pd.Series({"HI": 0.05, "LO": 0.049})
    vol = pd.Series({"HI": 0.01, "LO": 0.10})
    res = target_weights(mu, vol, cfg(max_names=1))
    assert res.diagnostics["selected"] == ["HI"]


def test_invalid_inputs_rejected():
    mu = pd.Series({"A": 0.05})
    with pytest.raises(ValueError):
        target_weights(mu, pd.Series({"A": 0.02, "B": 0.02}), cfg())
    with pytest.raises(ValueError):
        target_weights(mu, pd.Series({"A": 0.0}), cfg())
    with pytest.raises(ValueError):
        target_weights(mu, pd.Series({"A": float("nan")}), cfg())
    with pytest.raises(ValueError):
        cfg(max_names=0)


def test_check_weights_catches_violations():
    w = pd.Series({"A": 0.9})
    rep = check_weights(w, limits())
    assert not rep.approved
    assert any("gross" in v for v in rep.violations)
    assert any("single-name" in v for v in rep.violations)
    neg = pd.Series({"A": -0.1})
    assert not check_weights(neg, limits()).approved


def test_check_order_intents_rejections():
    lim = limits()
    good = {"symbol": "A", "side": "BUY", "qty": 100, "ref_price": 20.0}
    approved, rejected = check_order_intents(
        [good], limits=lim, equity=1_000_000, cash=50_000,
        sellable_qty={}, stale_symbols=set(), eligible_symbols={"A"},
        open_orders_count={})
    assert len(approved) == 1 and not rejected

    cases = [
        ({"symbol": "A", "side": "SHORT", "qty": 10, "ref_price": 10.0}, "side"),
        ({"symbol": "A", "side": "BUY", "qty": float("nan"), "ref_price": 10.0}, "nonfinite"),
        ({"symbol": "A", "side": "BUY", "qty": 0, "ref_price": 10.0}, "nonfinite/nonpositive"),
        ({"symbol": "A", "side": "BUY", "qty": 10, "ref_price": 10.0}, "below minimum"),  # value 100 < 1000
        ({"symbol": "A", "side": "BUY", "qty": 10000, "ref_price": 10.0}, "exceeds available cash"),
        ({"symbol": "A", "side": "BUY", "qty": 500, "ref_price": 10.0}, "not in eligible"),  # value 5000 passes min
        ({"symbol": "A", "side": "SELL", "qty": 50, "ref_price": 10.0}, "sellable"),
    ]
    approved, rejected = check_order_intents(
        [c[0] for c in cases], limits=lim, equity=1_000_000, cash=5_000,
        sellable_qty={"A": 10.0}, stale_symbols=set(),
        eligible_symbols={"B"},  # non-empty set that excludes A
        open_orders_count={})
    assert len(approved) == 0 and len(rejected) == len(cases)
    for (intent, needle), rej in zip(cases, rejected):
        assert any(needle in r for r in rej["rejected_reasons"]), (needle, rej)


def test_pending_buy_value_freezes_cash():
    """F-004: open BUY notional is frozen — new buys check against cash NET
    of the freeze, so pending orders cannot double-commit the same cash."""
    lim = limits()
    buy = {"symbol": "A", "side": "BUY", "qty": 50000, "ref_price": 10.0}  # 500k
    approved, rejected = check_order_intents(
        [buy], limits=lim, equity=1_000_000, cash=1_000_000,
        pending_buy_value=800_000, sellable_qty={}, stale_symbols=set(),
        eligible_symbols={"A"}, open_orders_count={})
    assert approved == []
    assert any("frozen pending buys" in r for r in rejected[0]["rejected_reasons"])
    # within net cash -> approved
    approved, rejected = check_order_intents(
        [buy], limits=lim, equity=1_000_000, cash=1_000_000,
        pending_buy_value=100_000, sellable_qty={}, stale_symbols=set(),
        eligible_symbols={"A"}, open_orders_count={})
    assert len(approved) == 1 and not rejected


def test_stale_data_blocks_buys_but_allows_derisking_sells():
    """Handover 8.6: a data incident blocks NEW risk, never traps existing
    exposure — SELL on a stale symbol must stay possible."""
    lim = limits()
    buy = {"symbol": "A", "side": "BUY", "qty": 100, "ref_price": 10.0}
    sell = {"symbol": "A", "side": "SELL", "qty": 50, "ref_price": 10.0}
    approved, rejected = check_order_intents(
        [buy, sell], limits=lim, equity=1_000_000, cash=50_000,
        sellable_qty={"A": 100.0}, stale_symbols={"A"},
        eligible_symbols={"A"}, open_orders_count={})
    assert len(approved) == 1 and approved[0]["side"] == "SELL"
    assert len(rejected) == 1 and rejected[0]["side"] == "BUY"
    assert any("stale" in r for r in rejected[0]["rejected_reasons"])
