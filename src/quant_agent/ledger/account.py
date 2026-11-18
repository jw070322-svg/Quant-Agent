"""Event-sourced cash/lots ledger — the authoritative account state.

Fixes handover 6.3: holdings are NOT reset to zero each run; the highest
price since entry lives in the ledger (not process memory); T+1 sellability
and corporate actions are first-class events. State is rebuildable from the
event log alone (restart reconciliation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Lot:
    qty: float
    entry_price: float          # adjusted, split-consistent
    entry_date: str             # session date of acquisition
    high_water: float           # max adjusted close since entry

    def apply_corp_action(self, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("corp action ratio must be positive")
        self.entry_price *= ratio
        self.high_water *= ratio


class Account:
    """Cash + FIFO lots. All prices are ADJUSTED session prices."""

    def __init__(self, initial_cash: float, *, currency: str, t_plus_1: bool):
        if initial_cash < 0:
            raise ValueError("initial cash must be nonnegative")
        self.currency = currency
        self.t_plus_1 = t_plus_1
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.lots: dict[str, list[Lot]] = {}
        self.realized_pnl = 0.0
        self.events: list[dict] = []
        # N4: last VALIDATED mark per symbol, with its session. Equity falls
        # back to this (never to entry price) when no bar exists; symbols
        # whose mark is stale are reported by stale_marks() and must not
        # receive new risk.
        self.last_marks: dict[str, dict] = {}

    def apply_mark(self, symbol: str, mark: float, *, session: str) -> None:
        """Record a validated mark (e.g., an adjusted close that exists for
        this symbol on this session)."""
        if not np.isfinite(mark) or mark <= 0:
            raise ValueError(f"invalid mark for {symbol}: {mark}")
        self.last_marks[symbol] = {"mark": float(mark), "session": str(session)}
        self.update_high_water(symbol, float(mark), session=session)
        self._record("mark", session, symbol=symbol, mark=float(mark))

    def stale_marks(self, session, *, max_age_sessions: int) -> list[str]:
        """Held symbols whose valuation is materially incomplete: no mark at
        all, or a last mark older than max_age_sessions (calendar-day gap at
        daily resolution). New BUYs on these symbols must be blocked; SELLs
        (de-risking) remain permitted."""
        session_ts = pd.Timestamp(session)
        out: set[str] = set()
        for sym in self.lots:
            m = self.last_marks.get(sym)
            if m is None:
                out.add(sym)
                continue
            age_days = (session_ts - pd.Timestamp(m["session"])).days
            if age_days > max_age_sessions:
                out.add(sym)
        return sorted(out)

    # ---------------------------------------------------------------- events
    def apply_deposit(self, amount: float, session: str) -> None:
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.cash += amount
        self._record("deposit", session, amount=amount)

    def apply_fill(self, *, symbol: str, side: str, qty: float, price: float,
                   fees: float, session: str, order_id: str) -> None:
        if side not in ("BUY", "SELL"):
            raise ValueError(f"invalid fill side: {side}")
        if qty <= 0 or price <= 0:
            raise ValueError(f"invalid fill qty/price: {qty} @ {price}")
        if side == "SELL":
            sellable = self.sellable_qty(symbol, session)
            if qty > sellable + 1e-9:
                raise ValueError(
                    f"SELL fill for {symbol} on {session} exceeds sellable qty "
                    f"({qty} > {sellable}); T+1/holdings violated")
        if side == "BUY" and qty * price + fees > self.cash + 1e-6:
            raise ValueError(f"BUY fill for {symbol} exceeds cash")
        if side == "BUY":
            self.cash -= qty * price + fees
            self.lots.setdefault(symbol, []).append(
                Lot(qty=qty, entry_price=price, entry_date=session, high_water=price))
        else:
            self.cash += qty * price - fees
            remaining = qty
            while remaining > 1e-9:
                lot = self.lots[symbol][0]
                take = min(lot.qty, remaining)
                self.realized_pnl += take * (price - lot.entry_price)
                lot.qty -= take
                remaining -= take
                if lot.qty <= 1e-9:
                    self.lots[symbol].pop(0)
            if not self.lots.get(symbol):
                self.lots.pop(symbol, None)
        self._record("fill", session, symbol=symbol, side=side, qty=qty,
                     price=price, fees=fees, order_id=order_id)

    def apply_corp_action(self, *, symbol: str, session: str, factor_old: float,
                          factor_new: float) -> None:
        """RAW-space lot basis adjustment. WARNING: this ledger stores
        ADJUSTED prices throughout; adjusted series are continuous across
        corporate actions by construction, so the simulator must NOT call
        this. It exists only for a future raw-price ledger."""
        if symbol not in self.lots:
            return
        if factor_old <= 0 or factor_new <= 0:
            raise ValueError("adjustment factors must be positive")
        ratio = factor_new / factor_old
        for lot in self.lots[symbol]:
            lot.apply_corp_action(ratio)
        self._record("corp_action", session, symbol=symbol, ratio=ratio,
                     factor_old=factor_old, factor_new=factor_new)

    def record_observation(self, kind: str, session: str, **fields) -> None:
        """Append an audit event WITHOUT state changes (e.g. an adjustment
        anchor move observed in the data — a no-op for an adjusted-space
        ledger, but provenance worth keeping)."""
        self._record(kind, session, **fields)

    def update_high_water(self, symbol: str, close: float, session: str = "") -> None:
        """Track highest adjusted close since entry per lot; record an event
        whenever the mark rises (review doc: high-water must reach the log)."""
        lots = self.lots.get(symbol)
        if not lots:
            return
        prev = max(l.high_water for l in lots)
        if close > prev:
            for lot in lots:
                lot.high_water = max(lot.high_water, close)
            self._record("hwm_update", session, symbol=symbol, high_water=close)

    # ----------------------------------------------------------------- state
    def positions(self) -> dict[str, float]:
        return {sym: sum(l.qty for l in lots) for sym, lots in self.lots.items()}

    def avg_cost(self, symbol: str) -> float | None:
        lots = self.lots.get(symbol)
        if not lots:
            return None
        total_q = sum(l.qty for l in lots)
        return sum(l.qty * l.entry_price for l in lots) / total_q

    def sellable_qty(self, symbol: str, session) -> float:
        # normalize ANY session representation to an ISO date string BEFORE
        # comparing with lot entry dates: a Timestamp stringifies with a time
        # suffix, and "2022-09-19" < "2022-09-19 00:00:00" silently made
        # same-day lots sellable under T+1 (found by the CN research run).
        session = str(pd.Timestamp(session).date())
        lots = self.lots.get(symbol, [])
        if not self.t_plus_1:
            return sum(l.qty for l in lots)
        return sum(l.qty for l in lots if l.entry_date < session)

    def equity(self, prices: dict[str, float], session: str | None = None) -> float:
        """Mark-to-market equity. Fallback chain per held symbol (N4):
        today's price → last VALIDATED mark (with its age recorded via
        stale_marks) → entry price as the final resort. Symbols valued below
        today's bar are reported by stale_marks() and blocked from new risk."""
        total = self.cash
        for sym, lots in self.lots.items():
            qty = sum(l.qty for l in lots)
            px = prices.get(sym)
            if px is None:
                m = self.last_marks.get(sym)
                px = m["mark"] if m else lots[0].entry_price
            total += qty * px
        return float(total)

    def _record(self, kind: str, session: str, **fields) -> None:
        self.events.append({"kind": kind, "session": session, **fields})


class RawAccount(Account):
    """Raw shares/currency, with dividend receivables included in NAV."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receivables = []

    def action(self, event):
        symbol, date = event['symbol'], str(pd.Timestamp(event['date']).date())
        value, kind = float(event['value']), event['kind']
        if not np.isfinite(value) or value < 0 or (kind == 'split' and value == 0):
            raise ValueError('invalid corporate action')
        qty = self.positions().get(symbol, 0.)
        if kind == 'split':
            for lot in self.lots.get(symbol, []):
                lot.qty *= value
                lot.entry_price /= value
                lot.high_water /= value
            if symbol in self.last_marks:
                self.last_marks[symbol]['mark'] /= value
        elif kind == 'dividend':
            self.receivables.append({'amount': qty*value, 'pay_date': str(pd.Timestamp(event['pay_date']).date())})
        elif kind == 'delisting':
            self.cash += qty*value
            self.lots.pop(symbol, None)
        else:
            raise ValueError(f'unsupported action {kind}')
        self._record('raw_action', date, symbol=symbol, action=kind, value=value, quantity=qty)

    def settle(self, session):
        day = str(pd.Timestamp(session).date())
        due = [r for r in self.receivables if r['pay_date'] <= day]
        for item in due:
            self.cash += item['amount']
            self._record('dividend_payment', day, amount=item['amount'])
        self.receivables = [r for r in self.receivables if r['pay_date'] > day]

    def equity(self, prices, session=None):
        return super().equity(prices, session) + sum(r['amount'] for r in self.receivables)
