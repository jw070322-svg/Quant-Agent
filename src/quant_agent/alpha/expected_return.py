"""From event probability to expected return (handover 4.4, 8.4).

P(label > 0) is NOT an expected return and its complement is NOT a short
signal. We combine the calibrated-ish event probability with CONDITIONAL
payoff estimates estimated on the same training fold and shrunk toward the
global mean:

    E[r | x] = p * E[r | event, sym] + (1-p) * E[r | complement, sym]

Everything here is estimated on training data only, never on the OOS window
being reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PayoffModel:
    """Per-symbol conditional payoff estimates with shrinkage."""
    e_event: dict[str, float]
    e_complement: dict[str, float]
    global_event: float
    global_complement: float
    shrink: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, symbol_col: str = "symbol",
            label_col: str = "label", shrink: float = 0.5) -> "PayoffModel":
        if not 0.0 <= shrink <= 1.0:
            raise ValueError("shrink must be in [0, 1]")
        y = train[label_col].to_numpy(dtype=float)
        event = y > 0
        g_ev = float(y[event].mean()) if event.any() else 0.0
        g_co = float(y[~event].mean()) if (~event).any() else 0.0
        e_ev: dict[str, float] = {}
        e_co: dict[str, float] = {}
        for sym, grp in train.groupby(symbol_col):
            v = grp[label_col].to_numpy(dtype=float)
            ev = v > 0
            s_ev = float(v[ev].mean()) if ev.sum() >= 10 else g_ev
            s_co = float(v[~ev].mean()) if (~ev).sum() >= 10 else g_co
            e_ev[sym] = shrink * s_ev + (1 - shrink) * g_ev
            e_co[sym] = shrink * s_co + (1 - shrink) * g_co
        return cls(e_ev, e_co, g_ev, g_co, shrink)

    def expected_return(self, symbol: str, p_event: float) -> float:
        e_ev = self.e_event.get(symbol, self.global_event)
        e_co = self.e_complement.get(symbol, self.global_complement)
        p = float(np.clip(p_event, 0.0, 1.0))
        return p * e_ev + (1.0 - p) * e_co


def round_trip_cost_bps(market_cfg: dict) -> float:
    """Conservative full round-trip cost in bps of notional (entry+exit)."""
    c = market_cfg["costs"]
    return (c["commission_bps"] + c["slippage_bps"]) * 2.0 + c["stamp_tax_sell_bps"]


def net_edge(mu_gross: float, cost_bps: float) -> float:
    return mu_gross - cost_bps / 1e4
