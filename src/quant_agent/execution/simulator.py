"""Event-driven daily simulator with a continuous cash/holdings ledger.

Execution timeline per session t (all prices adjusted):
  1. Fill pending orders at t's OPEN (slippage, participation cap, lots,
     cash re-check, T+1 sellability re-check).
  2. Intraday stop evaluation against t's bar: if low <= stop level (derived
     from entry cost and the trailing high-water mark as of t-1), a SELL fill
     prints at min(open, stop) — a gap through the stop fills at the open.
     A stop breach is an exit intent; the fill is the only truth.
  3. Mark to market at CLOSE; update per-lot high-water marks.
  4. After close: decision_fn(t) -> None means HOLD (no orders, positions
     unchanged); {} means go flat; a dict means rebalance to these target
     weights. Deltas -> order intents; independent risk checks; surviving
     intents become next-session orders.

Fixes handover 5.1/5.2/6.3: positions persist across sessions and windows,
returns are compounded from simple mark-to-market equity, and exits actually
exist. Missing bars (suspension) leave orders pending until expiry — never
fabricated fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ledger.account import Account
from ..portfolio.optimizer import PortfolioConfig
from ..risk.checks import RiskLimits, check_order_intents
from .broker import PaperBroker
from .orders import FILLED, RISK_APPROVED, SUBMITTED, Order, idempotency_key


@dataclass
class SimResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    fills: pd.DataFrame
    orders: pd.DataFrame
    rejected_intents: list[dict]
    turnover: pd.Series
    costs: pd.Series
    exposure: pd.Series
    stop_events: list[dict]
    account: Account = field(repr=False)

    def to_dicts(self) -> dict:
        return {
            "equity_curve": self.equity_curve,
            "daily_returns": self.daily_returns,
            "fills": self.fills,
            "orders": self.orders,
        }


def weights_to_intents(target: dict[str, float], *, equity: float, account: Account,
                       prices: dict[str, float], limits: RiskLimits,
                       session: str) -> list[dict]:
    """Convert target weights into order intents; scales DOWN to respect the
    turnover cap (scaling down preserves feasibility; scaling up does not)."""
    symbols = sorted(set(target) | set(account.positions()))
    deltas: dict[str, float] = {}
    for sym in symbols:
        w = float(target.get(sym, 0.0))
        px = prices.get(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            continue  # no tradable mark: cannot size an intent this session
        cur_qty = account.positions().get(sym, 0.0)
        delta_value = w * equity - cur_qty * px
        if abs(delta_value) >= limits.min_order_value:
            deltas[sym] = delta_value

    gross_trade = sum(abs(v) for v in deltas.values())
    cap_value = limits.max_turnover * equity
    scale = min(1.0, cap_value / gross_trade) if gross_trade > cap_value else 1.0

    intents = []
    for sym, dv in deltas.items():
        side = "BUY" if dv > 0 else "SELL"
        qty = abs(dv) * scale / prices[sym]
        if qty <= 0:
            continue
        intents.append({
            "symbol": sym, "side": side, "qty": qty, "ref_price": prices[sym],
            "idempotency_key": idempotency_key("sim", session, sym, side, qty,
                                               abs(dv) * scale),
        })
    return intents


def run_simulation(panel_bars: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex,
                   decision_fn, *, market_cfg: dict, limits: RiskLimits,
                   initial_equity: float, run_id: str,
                   bars_cache: dict | None = None) -> SimResult:
    t_plus_1 = bool(market_cfg["t_plus_1"])
    account = Account(initial_equity, currency=market_cfg["currency"], t_plus_1=t_plus_1)
    broker = PaperBroker(market_cfg, run_id=run_id)

    # bars_cache: optional precomputed {symbol: {date: row-dict}} to avoid
    # repeated DatetimeIndex .loc lookups across many simulations. Semantics
    # are identical to building it here (tests assert equivalence).
    bars_indexed = bars_cache if bars_cache is not None else \
        {sym: df.set_index("date").to_dict("index") for sym, df in panel_bars.items()}
    last_factor: dict[str, float] = {}
    equity_rows, turnover_rows, cost_rows, expo_rows = [], [], [], []
    rejected_all: list[dict] = []
    stop_events: list[dict] = []
    costs_acc = 0.0

    for t in sessions:
        t_str = str(t.date())
        bar_t = {sym: rows[t] for sym, rows in bars_indexed.items() if t in rows}
        prices = {sym: float(row["close_adj"]) for sym, row in bar_t.items()}

        # 0) adjustment anchor changes: AUDIT ONLY. The ledger is fully in
        # adjusted space and adjusted prices are continuous across corporate
        # actions by construction, so lot bases must NOT be rescaled — doing
        # so fabricates cost basis and misfires stops (found by integration
        # tests on the CN factor-jump scenario).
        for sym, row in bar_t.items():
            f_new = float(row["adj_factor"])
            if sym in last_factor and not np.isclose(f_new, last_factor[sym]):
                account.record_observation("corp_action_observed", session=t_str,
                                           symbol=sym, factor_old=last_factor[sym],
                                           factor_new=f_new)
            last_factor[sym] = f_new

        # 1) fills at the open
        fillable = {sym: row for sym, row in bar_t.items()}
        day_fills = broker.fill_on_session(t, fillable, account)
        costs_acc += sum(f["fees"] for f in day_fills)

        # 2) intraday stops
        for sym, lots in list(account.lots.items()):
            row = bar_t.get(sym)
            if row is None:
                continue
            avg_cost = account.avg_cost(sym)
            if avg_cost is None:
                continue
            hwm = max(l.high_water for l in lots)
            stop = max(avg_cost * (1.0 - limits.stop_loss_pct),
                       hwm * (1.0 - limits.trail_stop_pct))
            if float(row["low_adj"]) <= stop:
                px = min(float(row["open_adj"]), stop) * (1.0 - market_cfg["costs"]["slippage_bps"] / 1e4)
                sellable = account.sellable_qty(sym, t)
                qty = min(sum(l.qty for l in lots), sellable)
                if qty > 0:
                    fees = max(qty * px * market_cfg["costs"]["commission_bps"] / 1e4,
                               market_cfg["costs"]["min_commission"]) + \
                        qty * px * market_cfg["costs"]["stamp_tax_sell_bps"] / 1e4
                    stop_id = idempotency_key(run_id, t_str, sym, "SELL", qty,
                                              qty * px, kind="stop")
                    account.apply_fill(symbol=sym, side="SELL", qty=qty, price=px,
                                       fees=fees, session=t_str, order_id=stop_id)
                    costs_acc += fees
                    broker._seen_keys.add(stop_id)
                    stop_events.append({"session": t_str, "symbol": sym, "qty": qty,
                                        "price": px, "stop_level": stop,
                                        "kind": "stop_loss_or_trail"})
                    o = Order(order_id=stop_id, idempotency_key=stop_id, symbol=sym,
                              side="SELL", qty=qty, order_type="STOP", ref_price=stop,
                              session_created=t_str, stop_price=stop)
                    # stops follow the SAME state machine and fill log as
                    # regular orders (review doc P0: stops flow through the
                    # same matching and fill log)
                    o.transition(RISK_APPROVED)
                    o.transition(SUBMITTED)
                    o.filled_qty = qty
                    o.avg_fill_price = px
                    o.fees_paid = fees
                    o.transition(FILLED)
                    broker.orders[stop_id] = o
                    broker.fills.append({
                        "session": t_str, "order_id": stop_id, "symbol": sym,
                        "side": "SELL", "qty": qty, "price": px, "fees": fees,
                        "order_type": "STOP", "stop_price": stop})

        # 3) mark to market
        for sym in list(account.lots):
            if sym in prices:
                account.update_high_water(sym, prices[sym], session=t_str)
        equity = account.equity(prices)
        invested = sum(account.positions().get(s, 0.0) * prices[s]
                       for s in account.positions() if s in prices)
        equity_rows.append((t, equity))
        cost_rows.append((t, costs_acc))
        expo_rows.append((t, invested / equity if equity > 0 else 0.0))

        # 4) after-close decisions -> next-session orders.
        # decision_fn contract: None = HOLD, change nothing (never liquidate);
        # {} = go flat; dict = rebalance to these target weights.
        target = decision_fn(t, account, prices)
        traded = 0.0
        if target is not None:
            intents = weights_to_intents(target, equity=equity, account=account,
                                         prices=prices, limits=limits, session=t_str)
            idx = sessions.get_indexer([t])[0]
            next_t = sessions[idx + 1] if idx + 1 < len(sessions) else t
            sellable_next = {sym: account.sellable_qty(sym, next_t)
                             for sym in account.positions()}
            open_count: dict[str, int] = {}
            pending_buy_value = 0.0
            for o in broker.pending_orders():
                open_count[o.symbol] = open_count.get(o.symbol, 0) + 1
                if o.side == "BUY":
                    # F-004: frozen funds — pending BUY notional cannot be
                    # double-committed by new intents in the same session
                    pending_buy_value += o.open_qty * o.ref_price
            approved, rejected = check_order_intents(
                intents, limits=limits, equity=equity, cash=account.cash,
                pending_buy_value=pending_buy_value,
                sellable_qty=sellable_next, stale_symbols=set(),
                eligible_symbols=set(bars_indexed),
                open_orders_count=open_count,
            )
            rejected_all.extend(rejected)
            broker.submit(approved, session=t_str)
            traded = sum(abs(i["qty"] * i["ref_price"]) for i in approved)
        turnover_rows.append((t, traded / equity if equity > 0 else 0.0))

    eq = pd.Series(dict(equity_rows)).sort_index()
    rets = eq.pct_change()
    if len(rets) > 0:
        rets.iloc[0] = eq.iloc[0] / initial_equity - 1.0
    orders_df = pd.DataFrame([{
        "order_id": o.order_id, "idempotency_key": o.idempotency_key,
        "symbol": o.symbol, "side": o.side, "qty": o.qty, "type": o.order_type,
        "created": o.session_created, "state": o.state, "filled_qty": o.filled_qty,
        "avg_fill_price": o.avg_fill_price, "fees": o.fees_paid, "reason": o.reason,
    } for o in broker.orders.values()])
    return SimResult(
        equity_curve=eq,
        daily_returns=rets,
        fills=pd.DataFrame(broker.fills),
        orders=orders_df,
        rejected_intents=rejected_all,
        turnover=pd.Series(dict(turnover_rows)).sort_index(),
        costs=pd.Series(dict(cost_rows)).sort_index(),
        exposure=pd.Series(dict(expo_rows)).sort_index(),
        stop_events=stop_events,
        account=account,
    )
