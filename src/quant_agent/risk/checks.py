"""Independent risk checks. Pure functions, no side effects, no fallbacks.

The risk module re-validates BOTH solved weights and rounded order intents.
A rejected check blocks the new risk; it never auto-rescales (handover 8.6:
"Return feasible solver weights unchanged ... validate constraints again
after rounding").
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RiskLimits:
    max_weight_per_name: float
    max_gross_exposure: float
    cash_reserve: float
    max_turnover: float
    min_order_value: float
    max_open_orders_per_symbol: int
    allowed_sides: list[str]
    stop_loss_pct: float
    trail_stop_pct: float


@dataclass
class CheckReport:
    approved: bool
    violations: list[str] = field(default_factory=list)


def check_weights(weights: pd.Series, limits: RiskLimits) -> CheckReport:
    w = weights.astype(float)
    violations = []
    gross = float(w.sum())
    if gross > limits.max_gross_exposure + 1e-9:
        violations.append(f"gross exposure {gross:.4f} > cap {limits.max_gross_exposure:.4f}")
    over = w[w > limits.max_weight_per_name + 1e-9]
    for name, wi in over.items():
        violations.append(f"single-name weight {name}={wi:.4f} > cap {limits.max_weight_per_name:.4f}")
    if (w < -1e-9).any():
        violations.append("negative weights present (long-only policy)")
    if gross > 1.0 - limits.cash_reserve + 1e-9:
        violations.append(
            f"cash reserve breached: exposure {gross:.4f} > {1 - limits.cash_reserve:.4f}")
    return CheckReport(not violations, violations)


def check_order_intents(intents: list[dict], *, limits: RiskLimits,
                        equity: float, cash: float,
                        sellable_qty: dict[str, float],
                        stale_symbols: set[str],
                        eligible_symbols: set[str],
                        open_orders_count: dict[str, int],
                        pending_buy_value: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Validate per-order intents. Returns (approved, rejected_with_reasons).

    F-004: `pending_buy_value` is the notional of open BUY orders — frozen
    cash that new buys CANNOT double-commit. An empty `eligible_symbols` set
    disables the eligibility check."""
    approved, rejected = [], []
    for it in intents:
        sym = it["symbol"]
        side = it["side"]
        qty = float(it["qty"])
        price = float(it.get("ref_price", 0.0))
        value = abs(qty * price)
        reasons: list[str] = []
        if side not in limits.allowed_sides:
            reasons.append(f"side {side} not allowed")
        if not (math.isfinite(qty) and math.isfinite(price)) or qty <= 0 or price <= 0:
            reasons.append(f"nonfinite/nonpositive qty/price ({qty} @ {price})")
        if sym in stale_symbols and side == "BUY":
            # stale data blocks NEW risk only: reducing exposure (SELL) must
            # stay possible during data incidents (handover 8.6 — a failure
            # blocks new risk, never traps existing exposure)
            reasons.append("stale data: last bar older than policy limit")
        if eligible_symbols and sym not in eligible_symbols:
            reasons.append("symbol not in eligible universe")
        if side == "SELL" and qty > sellable_qty.get(sym, 0.0) + 1e-9:
            reasons.append(
                f"sell qty {qty:.0f} exceeds sellable {sellable_qty.get(sym, 0.0):.0f} (T+1/holdings)")
        if side == "BUY":
            if value < limits.min_order_value:
                reasons.append(f"order value {value:.0f} below minimum {limits.min_order_value:.0f}")
            available = cash - pending_buy_value
            if value > available + 1e-6:
                reasons.append(
                    f"order value {value:.0f} exceeds available cash {available:.0f} "
                    f"(cash {cash:.0f} minus frozen pending buys {pending_buy_value:.0f})")
        if open_orders_count.get(sym, 0) >= limits.max_open_orders_per_symbol:
            reasons.append("open-order limit reached for symbol")
        if reasons:
            rejected.append({**it, "rejected_reasons": reasons})
        else:
            approved.append(it)
    return approved, rejected
