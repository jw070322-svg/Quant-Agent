"""Order state machine and idempotent submission."""
import pytest

from quant_agent.execution.broker import PaperBroker
from quant_agent.execution.orders import (CANCELLED, EXPIRED, FILLED,
                                          Order, OrderStateError,
                                          PROPOSED, REJECTED, SUBMITTED,
                                          idempotency_key)


def test_illegal_transitions_raise():
    o = Order(order_id="x", idempotency_key="k", symbol="A", side="BUY",
              qty=100, order_type="MARKET", ref_price=10.0, session_created="2026-01-05")
    with pytest.raises(OrderStateError):
        o.transition(SUBMITTED)          # PROPOSED -> SUBMITTED is illegal
    with pytest.raises(OrderStateError):
        o.transition(FILLED)
    o.transition("RISK_APPROVED")
    o.transition(SUBMITTED)
    with pytest.raises(OrderStateError):
        o.transition(REJECTED)           # SUBMITTED -> REJECTED is illegal
    o.transition(FILLED)
    with pytest.raises(OrderStateError):
        o.transition(CANCELLED)          # terminal state


def test_terminal_states_have_no_outgoing():
    for terminal in (FILLED, REJECTED, CANCELLED, EXPIRED):
        o = Order(order_id="x", idempotency_key="k", symbol="A", side="BUY",
                  qty=1, order_type="MARKET", ref_price=1.0, session_created="2026-01-05")
        o._state = terminal  # direct placement for the guard check
        with pytest.raises(OrderStateError):
            o.transition(FILLED)


def test_idempotency_key_deterministic_and_distinct():
    a = idempotency_key("run", "2026-01-05", "A", "BUY", 100, 1000.0)
    b = idempotency_key("run", "2026-01-05", "A", "BUY", 100, 1000.0)
    c = idempotency_key("run", "2026-01-06", "A", "BUY", 100, 1000.0)
    assert a == b
    assert a != c


def test_broker_dedupes_duplicate_submissions():
    mcfg = {"costs": {"commission_bps": 0, "min_commission": 0.0,
                      "slippage_bps": 0, "stamp_tax_sell_bps": 0},
            "lots": 1, "allow_fractional": False, "participation_cap": 1.0}
    broker = PaperBroker(mcfg, run_id="r")
    intent = {"symbol": "A", "side": "BUY", "qty": 100, "ref_price": 10.0,
              "idempotency_key": "same-key"}
    first = broker.submit([dict(intent)], session="2026-01-05")
    second = broker.submit([dict(intent)], session="2026-01-06")  # retry/replay
    assert len(first) == 1
    assert len(second) == 0  # no duplicate economic order
    assert len(broker.orders) == 1
