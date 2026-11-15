"""Factor-based FALLBACK strategy and failure-mode state machine.

STATUS: DESIGN DELIVERY, UNVALIDATED (review doc `CURRENT_STATUS_FACTOR_FALLBACK_CN.md`,
section II.3 — that review document is preserved in Chinese as immutable
evidence). The rule below is a research specification to be independently
accepted (cost/slippage/whipsaw stress tests) BEFORE it may ever be activated
in paper or live mode. Nothing in this module is wired into execution.

Working definition (no authoritative public definition of the so-called
"long-short factor tactics" exists; see review doc section I.3): interpretable
factors form long/cash tilts, managed under portfolio and execution
constraints. This v1 rule is the LONG/CASH version — selling existing
holdings is NOT shorting; a true long-short version would need borrow,
margin, and eligibility checks.

Rule scores are RULE SCORES, never probabilities (they must not be presented
as P(up) — that was a legacy defect class).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..util import now_utc_iso, read_json, write_json

# Conservative exposure budget for the fallback book — deliberately BELOW the
# primary policy caps (risk/default.yaml); a fallback runs smaller, not bigger.
FALLBACK_DEFAULT_BUDGET = 0.40


def fallback_rule_weights(factor_rows: dict[str, dict[str, float]], *,
                          budget: float = FALLBACK_DEFAULT_BUDGET,
                          max_weight_per_name: float = 0.10) -> tuple[dict[str, float], dict]:
    """Long/cash trend + volatility-constrained rule.

    factor_rows: symbol -> {ret_20, close_over_ma60, vol_20} (all causal,
    dimensionless features from the shared feature module).

    Eligibility: ret_20 > 0 AND close_over_ma60 > 0 AND 0 < vol_20 < inf.
    Ranking score: ret_20 / vol_20 (deflated trend score; NOT a probability).
    Allocation: inverse-volatility weights scaled to `budget`, each name
    capped; feasibility by construction, remainder stays cash.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    if not 0.0 < max_weight_per_name <= budget:
        raise ValueError("max_weight_per_name must be in (0, budget]")
    eligible: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for sym, f in factor_rows.items():
        ret20 = f.get("ret_20")
        over_ma60 = f.get("close_over_ma60")
        vol = f.get("vol_20")
        if ret20 is None or over_ma60 is None or vol is None \
                or not all(math.isfinite(float(x)) for x in (ret20, over_ma60, vol)):
            skipped[sym] = "nonfinite factors"
            continue
        vol = float(vol)
        if vol <= 0:
            skipped[sym] = "nonpositive volatility"
            continue
        if float(ret20) <= 0 or float(over_ma60) <= 0:
            skipped[sym] = "trend condition not met"
            continue
        eligible[sym] = float(ret20) / vol  # ranking score only
    weights: dict[str, float] = {}
    if eligible:
        inv_vol = {s: 1.0 / float(factor_rows[s]["vol_20"]) for s in eligible}
        total = sum(inv_vol.values())
        for s in sorted(eligible, key=lambda x: (-eligible[x], x)):
            weights[s] = min(max_weight_per_name, budget * inv_vol[s] / total)
    diagnostics = {
        "budget": budget,
        "rank_scores": dict(sorted(eligible.items(), key=lambda kv: -kv[1])),
        "skipped": skipped,
        "gross_weight": sum(weights.values()),
        "cash_weight": 1.0 - sum(weights.values()),
        "score_semantics": "rule score (ret20/vol20); NOT a probability",
        "validated": False,
    }
    return weights, diagnostics


class FallbackMode(str, Enum):
    PRIMARY = "PRIMARY"
    DEGRADED = "DEGRADED"
    FALLBACK = "FALLBACK"
    RECOVERY = "RECOVERY"
    HALT = "HALT"


# Legal transitions with their preconditions. Key design points from the
# review doc: model-quality degradation alone may route to FALLBACK (and only
# when the fallback strategy has been independently accepted); infrastructure
# faults go to HALT (no new risk, manage existing positions, do NOT silently
# switch strategies); recovery requires shadow validation and a cooldown.
LEGAL_TRANSITIONS: dict[FallbackMode, dict[str, FallbackMode]] = {
    FallbackMode.PRIMARY: {
        "MODEL_DEGRADED": FallbackMode.DEGRADED,
        "INFRA_FAULT": FallbackMode.HALT,
        "OPERATOR_HALT": FallbackMode.HALT,
    },
    FallbackMode.DEGRADED: {
        "PROMOTE_FALLBACK": FallbackMode.FALLBACK,   # requires fallback_accepted
        "MODEL_RECOVERED": FallbackMode.RECOVERY,    # requires shadow_ok + cooldown
        "INFRA_FAULT": FallbackMode.HALT,
        "OPERATOR_HALT": FallbackMode.HALT,
    },
    FallbackMode.FALLBACK: {
        "MODEL_RECOVERED": FallbackMode.RECOVERY,    # requires shadow_ok + cooldown
        "INFRA_FAULT": FallbackMode.HALT,
        "OPERATOR_HALT": FallbackMode.HALT,
    },
    FallbackMode.RECOVERY: {
        # restricted observation state: TWO gated exits to PRIMARY, both
        # requiring the cooldown and (for the operator path) explicit evidence
        "SHADOW_OK": FallbackMode.PRIMARY,           # shadow validation recorded
        "OPERATOR_RECOVERY_APPROVED": FallbackMode.PRIMARY,  # explicit human approval
        "MODEL_DEGRADED": FallbackMode.DEGRADED,
        "INFRA_FAULT": FallbackMode.HALT,
        "OPERATOR_HALT": FallbackMode.HALT,
    },
    FallbackMode.HALT: {
        # automatic step lands in the RESTRICTED observation state only;
        # returning to PRIMARY requires the explicit operator event below
        "INFRA_RECOVERED": FallbackMode.RECOVERY,
    },
}


class IllegalTransitionError(RuntimeError):
    pass


class StateCorruptionError(RuntimeError):
    """Persisted fallback state is unreadable/corrupt: fail conservatively —
    the caller must halt new risk; never auto-reset to PRIMARY."""


@dataclass
class FallbackPolicy:
    """Failure-mode router with a persisted, append-only decision history."""
    state_path: str
    fallback_accepted: bool = False     # set ONLY after independent acceptance
    cooldown_sessions: int = 10
    state: FallbackMode = FallbackMode.PRIMARY
    sessions_since_recovery: int = 0
    last_session: str = ""
    history: list[dict] = field(default_factory=list)

    def load(self) -> "FallbackPolicy":
        try:
            d = read_json(self.state_path)
        except FileNotFoundError:
            return self
        except ValueError as exc:  # invalid JSON (JSONDecodeError subclasses it)
            raise StateCorruptionError(
                f"fallback state file '{self.state_path}' is corrupt: {exc}") from exc
        try:
            mode = FallbackMode(d["mode"])
            sessions = int(d.get("sessions_since_recovery", 0))
            history = d.get("history", [])
            if sessions < 0 or not isinstance(history, list):
                raise ValueError("negative counter or non-list history")
        except (KeyError, ValueError, TypeError) as exc:
            raise StateCorruptionError(
                f"fallback state file '{self.state_path}' failed validation: {exc}") from exc
        # assign only after full validation (conservative failure, no partial state)
        self.state = mode
        self.sessions_since_recovery = sessions
        self.history = history
        self.last_session = str(d.get("last_session", ""))
        return self

    def persist(self) -> None:
        write_json(self.state_path, {
            "mode": self.state.value,
            "sessions_since_recovery": self.sessions_since_recovery,
            "last_session": self.last_session,
            "fallback_accepted": self.fallback_accepted,
            "history": self.history,
        })

    def handle(self, event: str, *, reason: str = "", shadow_ok: bool = False,
               evidence: str = "", session: str = "",
               actor: str = "unknown") -> FallbackMode:
        """Process a lifecycle event; raises on illegal transitions. OPERATOR_*
        events REQUIRE non-empty evidence (no fabricated approvals)."""
        candidates = LEGAL_TRANSITIONS[self.state]
        if event not in candidates:
            raise IllegalTransitionError(
                f"event '{event}' not legal in mode {self.state.value}; "
                f"legal: {sorted(candidates)}")
        if event.startswith("OPERATOR_") and not str(evidence).strip():
            raise IllegalTransitionError(
                f"event '{event}' requires non-empty evidence (who/what/where) — "
                "refusing to record an unverified operator action")
        target = candidates[event]
        if target is FallbackMode.FALLBACK and not self.fallback_accepted:
            raise IllegalTransitionError(
                "refusing FALLBACK: the fallback rule has not passed independent "
                "acceptance (stress tests + pre-registered thresholds)")
        if event == "MODEL_RECOVERED":
            if not shadow_ok:
                raise IllegalTransitionError(
                    "refusing recovery: shadow-run validation not recorded")
            if self.sessions_since_recovery < self.cooldown_sessions:
                raise IllegalTransitionError(
                    f"refusing recovery: cooldown {self.sessions_since_recovery}/"
                    f"{self.cooldown_sessions} sessions not elapsed")
        if target is FallbackMode.PRIMARY:
            # every exit to PRIMARY (SHADOW_OK / OPERATOR_RECOVERY_APPROVED)
            # is gated by the unique-session cooldown
            if self.sessions_since_recovery < self.cooldown_sessions:
                raise IllegalTransitionError(
                    f"refusing return to PRIMARY: cooldown "
                    f"{self.sessions_since_recovery}/{self.cooldown_sessions} sessions not elapsed")
        entering_recovery = target is FallbackMode.RECOVERY
        self.state = target
        if entering_recovery:
            # the restricted-observation period restarts on entry: the
            # cooldown to PRIMARY counts sessions IN the observation state
            self.sessions_since_recovery = 0
        self.history.append({
            "ts_utc": now_utc_iso(), "session": session, "event": event,
            "from": self.history[-1]["to"] if self.history else FallbackMode.PRIMARY.value,
            "to": target.value, "reason": reason, "evidence": str(evidence),
            "actor": actor,
        })
        self.persist()
        return target

    def on_session(self, *, primary_healthy: bool = True, infra_healthy: bool,
                   session: str = "") -> FallbackMode:
        """Session hook. HALT + healthy infra moves to the RESTRICTED
        observation state only — returning to PRIMARY requires the explicit
        OPERATOR_RECOVERY_APPROVED event with evidence (audit must never
        fabricate an operator confirmation). Model degradation events come
        from MONITORING on pre-registered metrics — never from a short run of
        losses (review doc section II.3.1)."""
        if not infra_healthy:
            return self.handle("INFRA_FAULT", reason="infrastructure health failed",
                               session=session)
        if self.state is FallbackMode.HALT:
            return self.handle("INFRA_RECOVERED",
                               reason="infra healthy again; restricted observation, "
                                      "PRIMARY requires OPERATOR_RECOVERY_APPROVED",
                               session=session)
        # cooldown advances once per UNIQUE session
        if not session or session != self.last_session:
            self.sessions_since_recovery += 1
            if session:
                self.last_session = session
            self.persist()
        return self.state
