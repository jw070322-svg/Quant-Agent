"""Order lifecycle: durable ids, explicit state machine, idempotency keys.

Handover 8.7: proposed -> risk-approved -> submitted -> partially
filled/filled, or rejected/cancelled/expired. A threshold breach is an exit
INTENT; fills happen only through the broker.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

PROPOSED = "PROPOSED"
RISK_APPROVED = "RISK_APPROVED"
SUBMITTED = "SUBMITTED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
REJECTED = "REJECTED"
CANCELLED = "CANCELLED"
EXPIRED = "EXPIRED"

VALID_TRANSITIONS: dict[str, set[str]] = {
    PROPOSED: {RISK_APPROVED, REJECTED, CANCELLED},
    RISK_APPROVED: {SUBMITTED, CANCELLED},
    SUBMITTED: {PARTIALLY_FILLED, FILLED, CANCELLED, EXPIRED},
    PARTIALLY_FILLED: {PARTIALLY_FILLED, FILLED, CANCELLED, EXPIRED},
    FILLED: set(),
    REJECTED: set(),
    CANCELLED: set(),
    EXPIRED: set(),
}


class OrderStateError(RuntimeError):
    pass


@dataclass
class Order:
    order_id: str
    idempotency_key: str
    symbol: str
    side: str                     # BUY | SELL
    qty: float
    order_type: str               # MARKET | STOP
    ref_price: float
    session_created: str
    stop_price: float | None = None
    state: str = PROPOSED
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fees_paid: float = 0.0
    age_sessions: int = 0
    reason: str | None = None
    history: list[dict] = field(default_factory=list)

    def transition(self, new_state: str, *, reason: str | None = None) -> None:
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            raise OrderStateError(
                f"illegal transition {self.state} -> {new_state} for order {self.order_id}")
        self.history.append({"from": self.state, "to": new_state, "reason": reason})
        self.state = new_state
        if reason:
            self.reason = reason

    @property
    def open_qty(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_open(self) -> bool:
        return self.state in (RISK_APPROVED, SUBMITTED, PARTIALLY_FILLED)


def idempotency_key(run_id: str, session: str, symbol: str, side: str,
                    qty: float, ref_value: float, kind: str = "target") -> str:
    raw = f"{run_id}|{session}|{symbol}|{side}|{qty:.0f}|{ref_value:.2f}|{kind}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
