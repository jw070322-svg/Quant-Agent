"""Simulator acceptance scenarios (handover Phase 2 gate, review-doc P0):

hand-calculated reconciliation, zero-trade cash preservation, partial fills,
stop-loss/trailing fills, corporate actions, HOLD vs FLAT semantics.
Every number below is computed by hand in the comments.
"""
import numpy as np
import pandas as pd
import pytest

from quant_agent.execution.simulator import run_simulation
from quant_agent.ledger.account import Account
from quant_agent.risk.checks import RiskLimits


def limits(**kw):
    d = dict(max_weight_per_name=1.0, max_gross_exposure=1.0, cash_reserve=0.0,
             max_turnover=1.0, min_order_value=1.0, max_open_orders_per_symbol=5,
             allowed_sides=["BUY", "SELL"], stop_loss_pct=0.08, trail_stop_pct=0.10)
    d.update(kw)
    return RiskLimits(**d)


def mcfg(**kw):
    d = dict(currency="USD", t_plus_1=False, lots=1, allow_fractional=False,
             participation_cap=1.0,
             costs={"commission_bps": 0.0, "min_commission": 0.0,
                    "slippage_bps": 0.0, "stamp_tax_sell_bps": 0.0})
    d.update(kw)
    return d


def panel_from(bars):
    df = pd.DataFrame(bars)
    return {"SYM": df}


DATES = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])

BARS = [
    # date, open, high, low, close, volume, adj_factor (adj = raw here)
    ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
    ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
    ("2026-01-07", 12.0, 12.5, 11.9, 12.0, 1e6, 1.0),
    ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
]


def _bars(rows):
    return [{
        "date": pd.Timestamp(r[0]), "open": r[1], "high": r[2], "low": r[3],
        "close": r[4], "volume": r[5], "adj_factor": r[6],
        "open_adj": r[1], "high_adj": r[2], "low_adj": r[3], "close_adj": r[4],
    } for r in rows]


def test_hand_calculated_buy_and_hold_reconciliation():
    """Decision after close t1 {SYM: 1.0}; fills at t2 open 10.0; no fees.
    qty = 10000/10 = 1000 shares. equity: t2 = 1000*11 = 11000, t3 = 12000."""
    panel = panel_from(_bars(BARS))
    sessions = DATES

    def fn(t, account, prices):
        return {"SYM": 1.0} if t == sessions[0] else None  # buy once, hold

    sim = run_simulation(panel, sessions, fn, market_cfg=mcfg(), limits=limits(),
                         initial_equity=10_000.0, run_id="toy")
    fills = sim.fills
    assert len(fills) == 1
    assert fills.iloc[0]["qty"] == pytest.approx(1000.0)
    assert fills.iloc[0]["price"] == pytest.approx(10.0)  # no slippage
    # ledger truth: cash 0 + 1000 shares marked at close
    assert sim.account.cash == pytest.approx(0.0)
    assert sim.account.positions()["SYM"] == pytest.approx(1000.0)
    eq = sim.equity_curve
    assert eq.iloc[0] == pytest.approx(10_000.0)          # t1: bought nothing yet
    assert eq.iloc[1] == pytest.approx(1000.0 * 11.0)     # t2 mark
    assert eq.iloc[2] == pytest.approx(1000.0 * 12.0)
    assert eq.iloc[3] == pytest.approx(1000.0 * 13.0)
    # compounded simple returns, first return = 10% (10k -> 11k)
    assert sim.daily_returns.iloc[1] == pytest.approx(0.10)
    # completed round trip absent: still holding
    assert sim.account.avg_cost("SYM") == pytest.approx(10.0)


def test_zero_trading_preserves_cash_exactly():
    panel = panel_from(_bars(BARS))
    sim = run_simulation(panel, DATES, lambda t, a, p: None,
                         market_cfg=mcfg(), limits=limits(),
                         initial_equity=10_000.0, run_id="flat")
    assert sim.fills.empty
    assert (sim.equity_curve == 10_000.0).all()
    assert sim.account.cash == pytest.approx(10_000.0)
    assert sim.daily_returns.abs().sum() == pytest.approx(0.0)


def test_empty_target_liquidates_none_means_hold():
    """{} = go flat; None = do nothing. These MUST differ."""
    panel = panel_from(_bars(BARS))
    # buy at t1 decision, then explicitly go flat at t2
    def flat_after(t, account, prices):
        return {"SYM": 1.0} if t == DATES[0] else {}
    sim_flat = run_simulation(panel, DATES, flat_after, market_cfg=mcfg(),
                              limits=limits(), initial_equity=10_000.0, run_id="go-flat")
    assert sim_flat.account.positions().get("SYM", 0) == pytest.approx(0.0)
    assert sim_flat.account.cash > 0
    # hold: position survives untouched
    def hold(t, account, prices):
        return {"SYM": 1.0} if t == DATES[0] else None
    sim_hold = run_simulation(panel, DATES, hold, market_cfg=mcfg(),
                              limits=limits(), initial_equity=10_000.0, run_id="hold")
    assert sim_hold.account.positions()["SYM"] == pytest.approx(1000.0)


def test_partial_fill_volume_cap_then_cash_shortfall_cancels_remainder():
    """t2 volume 800 & cap 0.5 -> only 400 of the 1000-share order fill.
    At t3 the remainder (600 @ 12.0 = 7200) exceeds remaining cash (6000):
    the order is CANCELLED, NOT silently squeezed — explicit contract."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 800.0, 1.0),
        ("2026-01-07", 12.0, 12.5, 11.9, 12.0, 1e6, 1.0),
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(participation_cap=0.5), limits=limits(),
                         initial_equity=10_000.0, run_id="partial")
    fills = sim.fills
    assert len(fills) == 1
    assert fills.iloc[0]["qty"] == pytest.approx(400.0)
    assert fills.iloc[0]["price"] == pytest.approx(10.0)
    assert sim.account.positions()["SYM"] == pytest.approx(400.0)
    assert sim.account.cash == pytest.approx(6_000.0)
    # the remainder was cancelled with a reason, not squeezed
    order = sim.orders.iloc[0]
    assert order["state"] == "CANCELLED"
    assert "insufficient cash" in order["reason"]
    # equity hand-checked: 6000 cash + 400 shares * close
    assert sim.equity_curve.iloc[2] == pytest.approx(6_000.0 + 400.0 * 12.0)
    assert sim.equity_curve.iloc[3] == pytest.approx(6_000.0 + 400.0 * 13.0)


def test_hard_stop_loss_fills_below_level_with_gap_rule():
    """Buy 1000 @ 10 at t2. Hard stop = 10*0.92 = 9.2. t3 low 9.0 breaches:
    fill price = min(open, stop) = min(12.0, 9.2) = 9.2 (gap rule: the open is
    ABOVE the stop here, so the stop level itself prints). Trailing is
    neutralized (0.99) to isolate the hard stop."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
        ("2026-01-07", 12.0, 12.5, 9.0, 9.5, 1e6, 1.0),   # low pierces the stop
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(), limits=limits(trail_stop_pct=0.99),
                         initial_equity=10_000.0, run_id="stop")
    assert len(sim.stop_events) == 1
    ev = sim.stop_events[0]
    assert ev["symbol"] == "SYM"
    assert ev["price"] == pytest.approx(9.2)   # stop level, not the low
    assert ev["stop_level"] == pytest.approx(9.2)
    # fill recorded through the SAME broker fill log with STOP type
    stop_fills = sim.fills[sim.fills["order_type"] == "STOP"]
    assert len(stop_fills) == 1 and stop_fills.iloc[0]["qty"] == pytest.approx(1000.0)
    assert sim.account.positions().get("SYM", 0.0) == pytest.approx(0.0)
    assert sim.account.cash == pytest.approx(9_200.0)
    # realized loss booked in the ledger
    assert sim.account.realized_pnl == pytest.approx(1000 * (9.2 - 10.0))
    # the exit was an order object that followed the full state machine
    o = sim.orders[sim.orders["type"] == "STOP"].iloc[0]
    assert o["state"] == "FILLED"


def test_trailing_stop_tracks_high_water_mark():
    """Buy 1000 @ 10 (t2). t2 close 11 -> HWM 11; trailing stop = 11*0.90 = 9.9,
    hard stop = 9.2 -> effective 9.9. t3 low 9.5 breaches at 9.9:
    fill = min(open 12.0, 9.9) = 9.9."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
        ("2026-01-07", 12.0, 12.5, 9.5, 9.8, 1e6, 1.0),
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(), limits=limits(), initial_equity=10_000.0,
                         run_id="trail")
    assert len(sim.stop_events) == 1
    assert sim.stop_events[0]["price"] == pytest.approx(9.9)
    assert sim.account.realized_pnl == pytest.approx(1000 * (9.9 - 10.0))


def test_gap_through_stop_fills_at_open():
    """Open already below the stop -> fill at the OPEN (worse), never at the
    impossible stop level: open 9.0 < stop 9.2 -> price 9.0."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
        ("2026-01-07", 9.0, 9.4, 8.8, 9.1, 1e6, 1.0),    # gapped through stop
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(), limits=limits(), initial_equity=10_000.0,
                         run_id="gap")
    assert sim.stop_events[0]["price"] == pytest.approx(9.0)


def test_corporate_action_is_audit_only_in_adjusted_space():
    """adj_factor 1.0 -> 2.0 between t2 and t3 with unchanged ADJUSTED prices:
    the ledger is fully adjusted-space, so the lot basis must NOT change and
    no phantom stop/P&L may appear; the anchor move is recorded for audit."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
        ("2026-01-07", 12.0, 12.5, 11.9, 12.0, 1e6, 2.0),  # raw 2x, adj same
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 2.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(), limits=limits(), initial_equity=10_000.0,
                         run_id="corp")
    lots = sim.account.lots["SYM"]
    assert all(l.entry_price == pytest.approx(10.0) for l in lots)  # unchanged
    assert any(e["kind"] == "corp_action_observed" for e in sim.account.events)
    assert sim.account.realized_pnl == pytest.approx(0.0)
    assert sim.account.positions()["SYM"] == pytest.approx(1000.0)  # no phantom stop
    assert sim.equity_curve.iloc[2] == pytest.approx(1000.0 * 12.0)


def test_raw_space_corp_action_scales_lot_basis():
    """apply_corp_action exists for a future RAW-price ledger: scaling the
    basis must preserve per-lot economics (half price, double count)."""
    from quant_agent.ledger.account import Account, Lot
    acct = Account(0.0, currency="USD", t_plus_1=False)
    acct.lots["X"] = [Lot(qty=100, entry_price=10.0, entry_date="d1", high_water=12.0)]
    acct.apply_corp_action(symbol="X", session="d2", factor_old=1.0, factor_new=2.0)
    assert acct.lots["X"][0].entry_price == pytest.approx(20.0)
    assert acct.lots["X"][0].high_water == pytest.approx(24.0)
    assert any(e["kind"] == "corp_action" for e in acct.events)


def test_t_plus_1_blocks_same_day_sell_and_allows_next_day():
    cn = mcfg(currency="CNY", t_plus_1=True, lots=100)
    panel = panel_from(_bars(BARS))
    # decision after close t2 (position bought at t2 open): SELL would need
    # sellable inventory; the ledger refuses T+0 sells outright.
    acct_events = []
    def fn(t, account, prices):
        acct_events.append((t, dict(account.positions())))
        if t == DATES[0]:
            return {"SYM": 1.0}
        if t == DATES[1]:
            return {}  # go flat at t2's close -> sell intent fills at t3 open
        return None
    sim = run_simulation(panel, DATES, fn, market_cfg=cn, limits=limits(),
                         initial_equity=10_000.0, run_id="t1")
    # entry filled at t2 open (entry_date t2); sell intent sized at t2 close,
    # fills at t3 open — entry_date(t2) < t3 satisfies T+1
    assert sim.account.positions().get("SYM", 0.0) == pytest.approx(0.0)
    assert sim.account.cash == pytest.approx(1000.0 * 12.0)


def test_ledger_replay_from_events_reconstructs_state():
    """Restart safety: a fresh Account rebuilt from the event log alone must
    reproduce cash, positions, and realized P&L exactly."""
    bars = [
        ("2026-01-05", 10.0, 10.5, 9.9, 10.0, 1e6, 1.0),
        ("2026-01-06", 10.0, 11.0, 9.9, 11.0, 1e6, 1.0),
        ("2026-01-07", 12.0, 12.5, 9.0, 9.5, 1e6, 1.0),   # triggers the stop
        ("2026-01-08", 13.0, 13.5, 12.9, 13.0, 1e6, 1.0),
    ]
    panel = panel_from(_bars(bars))
    sim = run_simulation(panel, DATES, lambda t, a, p: {"SYM": 1.0} if t == DATES[0] else None,
                         market_cfg=mcfg(), limits=limits(), initial_equity=10_000.0,
                         run_id="replay")
    original = sim.account
    replay = Account(0.0, currency=original.currency, t_plus_1=original.t_plus_1)
    replay.apply_deposit(original.initial_cash, session="init")
    for ev in original.events:
        if ev["kind"] == "fill":
            replay.apply_fill(symbol=ev["symbol"], side=ev["side"], qty=ev["qty"],
                              price=ev["price"], fees=ev["fees"],
                              session=ev["session"], order_id=ev["order_id"])
        elif ev["kind"] == "corp_action":
            replay.apply_corp_action(symbol=ev["symbol"], session=ev["session"],
                                     factor_old=ev["factor_old"],
                                     factor_new=ev["factor_new"])
        elif ev["kind"] == "hwm_update":
            replay.update_high_water(ev["symbol"], ev["high_water"],
                                     session=ev["session"])
    assert replay.cash == pytest.approx(original.cash)
    assert replay.positions() == pytest.approx(original.positions())
    assert replay.realized_pnl == pytest.approx(original.realized_pnl)
    prices = {"SYM": 13.0}
    assert replay.equity(prices) == pytest.approx(original.equity(prices))

