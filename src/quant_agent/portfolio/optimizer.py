"""Long-only target weights. Feasible by construction; NEVER rescaled.

Fixes handover 6.4: the legacy optimizer solved a constrained problem and
then divided by total weight and forced 80% exposure, breaking every
constraint it had just "respected". Here:

- allocation is a capped proportional water-filling solved exactly on the
  capped simplex (no solver, no post-hoc normalization);
- cash is a first-class output: no candidates or weak edges -> all cash;
- the returned weights satisfy every limit by construction, and the risk
  module re-validates them independently (defense in depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    edge_threshold: float          # min net expected edge to hold a name
    lambda_vol: float              # risk penalty weight on volatility in the score
    max_names: int
    max_weight_per_name: float
    max_gross_exposure: float

    def __post_init__(self):
        if self.max_names < 1:
            raise ValueError("max_names must be >= 1")
        if not 0.0 < self.max_weight_per_name <= 1.0:
            raise ValueError("max_weight_per_name must be in (0, 1]")
        if not 0.0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")


@dataclass
class AllocationResult:
    weights: pd.Series                       # name -> target weight, sum <= gross cap
    diagnostics: dict = field(default_factory=dict)


def _capped_proportional(base: pd.Series, cap: float, budget: float) -> pd.Series:
    """Largest s with sum(min(cap, s*b_i)) <= budget, exact on the capped
    simplex. Names with a nonpositive base receive zero weight (they were
    selected for review but the risk penalty removed their allocation)."""
    if base.empty or base.sum() <= 0 or budget <= 0 or cap <= 0:
        return pd.Series(0.0, index=base.index)
    b = base[base > 0].sort_values(ascending=False)
    remaining = budget
    weights = {}
    for name, bi in b.items():
        # level s at which this and all remaining uncapped names hit cap
        s_full = remaining / b.loc[name:].sum()
        w = min(cap, s_full * bi)
        weights[name] = w
        remaining -= w
        if remaining <= 1e-12:
            break
    out = pd.Series(weights).reindex(base.index).fillna(0.0)
    # numeric guard: never exceed budget/cap by float noise
    excess = out.sum() - budget
    if excess > 1e-12:
        out = out * (budget / out.sum())
    return out.clip(lower=0.0, upper=cap)


def target_weights(mu_net: pd.Series, vol: pd.Series, cfg: PortfolioConfig) -> AllocationResult:
    """mu_net: expected return NET of estimated round-trip costs per name
    (latest estimate per symbol). vol: per-name volatility estimate."""
    if not mu_net.index.equals(vol.index):
        raise ValueError("mu and vol must share the same index")
    mu = pd.to_numeric(mu_net, errors="coerce").fillna(0.0)
    v = pd.to_numeric(vol, errors="coerce")
    if (v <= 0).any() or not np.isfinite(v).all():
        raise ValueError("vol must be finite and positive for every candidate")

    # N1 fix: the RISK-ADJUSTED edge (net of the volatility penalty) must clear
    # the threshold — a raw edge that the penalty eliminates is not eligible,
    # and if every penalized edge fails, the correct portfolio is cash.
    penalized = mu - cfg.lambda_vol * v
    eligible = penalized[penalized > cfg.edge_threshold].index
    if len(eligible) == 0:
        return AllocationResult(
            pd.Series(0.0, index=mu.index),
            {"reason": "no candidate above risk-adjusted edge threshold",
             "selected": [], "gross_weight": 0.0, "cash_weight": 1.0})
    # rank by edge per unit risk; tie-break deterministically by mu then name
    score = (mu[eligible] / v[eligible]).sort_values(ascending=False)
    selected = list(score.index[: cfg.max_names])
    base = penalized[selected].clip(lower=0.0)
    if base.sum() <= 0:
        return AllocationResult(pd.Series(0.0, index=mu.index),
                                {"reason": "risk penalty zeroed all edges", "selected": selected})
    w = _capped_proportional(base, cfg.max_weight_per_name, cfg.max_gross_exposure)
    w = w.reindex(mu.index).fillna(0.0)
    return AllocationResult(w, {
        "selected": selected,
        "scores": {s: float(score[s]) for s in selected},
        "gross_weight": float(w.sum()),
        "cash_weight": float(1.0 - w.sum()),
    })
