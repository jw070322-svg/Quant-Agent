"""Paper broker: fills at next-session open with explicit frictions.

N5 capacity assumption (disclosed): the participation cap multiplies the
COMPLETED session's volume — an ex-post, full-day capacity assumption. It is
not an observable opening-auction liquidity limit, and a stop threshold does
not guarantee fills at that price. These are modeling limitations documented
in ARCHITECTURE §6, to be resolved with event-level data (deferred per
comprehensive plan §4.5), not quantified P&L corrections.

Conventions (documented, tested):
- Fill price = next session's adjusted open, worsened by slippage.
- Fill qty is capped by participation (fraction of that bar's volume);
  the remainder stays open until filled or expired.
- BUY quantities are rounded to the market lot and re-checked against cash
  AFTER rounding (a rounded order that violates cash is rejected, not
  silently squeezed).
- SELL quantities are re-checked against T+1-aware sellable inventory.
- Duplicate idempotency keys are dropped (retry-safe submission).
"""
from __future__ import annotations

import pandas as pd

from ..ledger.account import Account
from .orders import (CANCELLED, EXPIRED, FILLED, PARTIALLY_FILLED,
                     RISK_APPROVED, SUBMITTED, Order, idempotency_key)


class PaperBroker:
    def __init__(self, market_cfg: dict, *, run_id: str, max_order_age_sessions: int = 5):
        self.costs = market_cfg["costs"]
        self.lots = int(market_cfg["lots"])
        self.allow_fractional = bool(market_cfg.get("allow_fractional", False))
        self.participation_cap = float(market_cfg["participation_cap"])
        self.run_id = run_id
        self.max_order_age_sessions = max_order_age_sessions
        self.orders: dict[str, Order] = {}
        self._seen_keys: set[str] = set()
        self._seq = 0
        self.fills: list[dict] = []

    # ------------------------------------------------------------- lifecycle
    def submit(self, intents: list[dict], session: str) -> list[Order]:
        """intents: dicts with symbol, side, qty, ref_price. Idempotent."""
        out = []
        for it in intents:
            key = it.get("idempotency_key") or idempotency_key(
                self.run_id, session, it["symbol"], it["side"],
                float(it["qty"]), float(it["ref_price"]) * float(it["qty"]),
                it.get("kind", "target"))
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._seq += 1
            order = Order(
                order_id=f"{self.run_id}-o{self._seq:06d}",
                idempotency_key=key,
                symbol=it["symbol"], side=it["side"], qty=float(it["qty"]),
                order_type=it.get("order_type", "MARKET"),
                ref_price=float(it["ref_price"]),
                session_created=session,
                stop_price=it.get("stop_price"),
            )
            # risk approval already happened upstream (risk.checks); record it
            # in the order's history before submission
            order.transition(RISK_APPROVED)
            order.transition(SUBMITTED)
            self.orders[order.order_id] = order
            out.append(order)
        return out

    def approve(self, orders: list[Order]) -> None:
        for o in orders:
            if o.state == "PROPOSED":
                o.transition("RISK_APPROVED")
                o.transition(SUBMITTED)

    def pending_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]

    # ------------------------------------------------------------------ fills
    def fill_on_session(self, session: str, bars: dict[str, pd.Series],
                        account: Account) -> list[dict]:
        """Attempt to fill every open order against this session's bars."""
        day_fills: list[dict] = []
        for order in self.pending_orders():
            created = pd.Timestamp(order.session_created)
            if created >= session:
                continue  # an order cannot fill on its creation session
            bar = bars.get(order.symbol)
            if bar is None:
                order.age_sessions += 1
                if order.age_sessions > self.max_order_age_sessions:
                    order.transition(EXPIRED, reason="no fillable bar within max age")
                continue
            qty_target = order.open_qty
            if qty_target <= 1e-9:
                continue
            max_qty = float(self.participation_cap * bar["volume"]) if bar["volume"] > 0 else 0.0
            qty = min(qty_target, max_qty)
            if qty <= 1e-9:
                order.age_sessions += 1
                if order.age_sessions > self.max_order_age_sessions:
                    order.transition(EXPIRED, reason="volume-capped beyond max age")
                continue

            if order.side == "BUY":
                qty = self._round_to_lot(qty, side="BUY")
                px = float(bar["open_adj"]) * (1.0 + self.costs["slippage_bps"] / 1e4)
                fees = self._fees(qty * px, sell=False)
                if qty <= 0 or qty * px + fees > account.cash:
                    # rounded order violates cash -> cancel the remainder
                    # (order was already submitted; CANCELLED, not REJECTED)
                    order.transition(CANCELLED, reason="insufficient cash after lot rounding")
                    continue
            else:
                sellable = account.sellable_qty(order.symbol, session)
                qty = min(qty, sellable)
                qty = self._round_to_lot(qty, side="SELL")
                if qty <= 0:
                    order.transition(CANCELLED, reason="no sellable inventory at fill time")
                    continue
                px = float(bar["open_adj"]) * (1.0 - self.costs["slippage_bps"] / 1e4)
                fees = self._fees(qty * px, sell=True)

            account.apply_fill(symbol=order.symbol, side=order.side, qty=qty,
                               price=px, fees=fees, session=str(session.date()),
                               order_id=order.order_id)
            order.filled_qty += qty
            order.avg_fill_price = px
            order.fees_paid += fees
            order.transition(FILLED if order.open_qty <= 1e-9 else PARTIALLY_FILLED)
            fill = {"session": str(session.date()), "order_id": order.order_id,
                    "symbol": order.symbol, "side": order.side, "qty": qty,
                    "price": px, "fees": fees,
                    "order_type": order.order_type, "stop_price": order.stop_price}
            self.fills.append(fill)
            day_fills.append(fill)
        return day_fills

    # -------------------------------------------------------------- internals
    def _round_to_lot(self, qty: float, *, side: str) -> float:
        if self.allow_fractional:
            return qty
        n = int(qty // self.lots)
        if n <= 0:
            return 0.0
        return float(n * self.lots) if side == "BUY" else float(n * self.lots)

    def _fees(self, value: float, *, sell: bool) -> float:
        fees = max(value * self.costs["commission_bps"] / 1e4, self.costs["min_commission"])
        if sell and self.costs["stamp_tax_sell_bps"]:
            fees += value * self.costs["stamp_tax_sell_bps"] / 1e4
        return fees


class RawPaperBroker(PaperBroker):
    """Raw opening fills with prior-session capacity estimates, not auction guarantees."""

    def fill_on_session(self, session, bars, account):
        day_fills = []
        # Sells release cash before buys. Stable order within each side.
        for order in sorted(self.pending_orders(), key=lambda o: (o.side != 'SELL', o.order_id)):
            if pd.Timestamp(order.session_created) >= session:
                continue
            order.age_sessions += 1
            if order.age_sessions > self.max_order_age_sessions:
                order.transition(EXPIRED, reason='execution window expired')
                continue
            bar = bars.get(order.symbol)
            side = order.side
            if bar is None or not bool(bar['buyable' if side == 'BUY' else 'sellable']):
                order.reason = 'missing quote or trading restriction'
                continue
            lot = int(bar['buy_lot' if side == 'BUY' else 'sell_lot'])
            qty = min(order.open_qty, max(0., float(bar['capacity_shares'])))
            if side == 'SELL':
                qty = min(qty, account.sellable_qty(order.symbol, session))
            qty = float(int(qty // lot)*lot)
            if qty <= 0:
                order.reason = 'capacity, lot size, or locked inventory'
                continue
            px = float(bar['open'])*(1 + (1 if side == 'BUY' else -1)*self.costs['slippage_bps']/10000)
            fees = self._fees(qty*px, sell=side == 'SELL')
            if side == 'BUY' and qty*px+fees > account.cash:
                order.reason = 'insufficient available cash'
                continue
            account.apply_fill(symbol=order.symbol, side=side, qty=qty, price=px, fees=fees,
                               session=str(session.date()), order_id=order.order_id)
            old_qty = order.filled_qty
            order.filled_qty += qty
            order.avg_fill_price = ((order.avg_fill_price or 0.)*old_qty+qty*px)/order.filled_qty
            order.fees_paid += fees
            order.transition(FILLED if order.open_qty <= 1e-9 else PARTIALLY_FILLED)
            fill = {'session': str(session.date()), 'order_id': order.order_id,
                    'symbol': order.symbol, 'side': side, 'qty': qty, 'price': px, 'fees': fees}
            self.fills.append(fill)
            day_fills.append(fill)
        return day_fills
