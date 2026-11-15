"""Fallback rule and failure-mode state machine (design delivery, unvalidated —
tests assert the SPEC, not profitability)."""
import pytest

from quant_agent.alpha.factor_fallback import (FALLBACK_DEFAULT_BUDGET,
                                               FallbackMode, FallbackPolicy,
                                               IllegalTransitionError,
                                               fallback_rule_weights)


def test_rule_picks_trending_names_and_respects_budget():
    rows = {
        "TREND": {"ret_20": 0.05, "close_over_ma60": 0.03, "vol_20": 0.02},
        "FLAT": {"ret_20": 0.001, "close_over_ma60": -0.002, "vol_20": 0.02},  # below MA60
        "DOWN": {"ret_20": -0.03, "close_over_ma60": -0.01, "vol_20": 0.02},
    }
    w, diag = fallback_rule_weights(rows, budget=0.40, max_weight_per_name=0.10)
    assert set(w) == {"TREND"}
    assert w["TREND"] <= 0.10
    assert w["TREND"] <= 0.40
    assert diag["cash_weight"] >= 0.60
    assert diag["score_semantics"].startswith("rule score")
    assert "NOT a probability" in diag["score_semantics"]


def test_rule_skips_nonfinite_and_zero_vol():
    rows = {
        "BAD": {"ret_20": float("nan"), "close_over_ma60": 0.01, "vol_20": 0.02},
        "ZEROVOL": {"ret_20": 0.05, "close_over_ma60": 0.01, "vol_20": 0.0},
        "GOOD": {"ret_20": 0.05, "close_over_ma60": 0.01, "vol_20": 0.02},
    }
    w, diag = fallback_rule_weights(rows)
    assert "BAD" in diag["skipped"] and "ZEROVOL" in diag["skipped"]
    assert set(w) == {"GOOD"}


def test_rule_inverse_vol_weighting_prefers_lower_vol():
    rows = {
        "LOWVOL": {"ret_20": 0.05, "close_over_ma60": 0.02, "vol_20": 0.01},
        "HIGHVOL": {"ret_20": 0.05, "close_over_ma60": 0.02, "vol_20": 0.04},
    }
    w, _ = fallback_rule_weights(rows, budget=0.40, max_weight_per_name=0.20)
    assert w["LOWVOL"] > w["HIGHVOL"]
    assert w["LOWVOL"] <= 0.20 and sum(w.values()) <= 0.40


def test_invalid_budget_and_caps_rejected():
    rows = {"A": {"ret_20": 0.05, "close_over_ma60": 0.01, "vol_20": 0.02}}
    with pytest.raises(ValueError):
        fallback_rule_weights(rows, budget=0.0)
    with pytest.raises(ValueError):
        fallback_rule_weights(rows, budget=1.5)
    with pytest.raises(ValueError):
        fallback_rule_weights(rows, budget=0.4, max_weight_per_name=0.5)


def test_default_budget_is_conservative():
    assert FALLBACK_DEFAULT_BUDGET <= 0.5


def test_state_machine_legal_and_illegal_paths(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"), fallback_accepted=True)
    assert pol.state is FallbackMode.PRIMARY
    with pytest.raises(IllegalTransitionError):
        pol.handle("PROMOTE_FALLBACK")  # PRIMARY -> FALLBACK directly is illegal
    assert pol.handle("MODEL_DEGRADED") is FallbackMode.DEGRADED
    assert pol.handle("PROMOTE_FALLBACK") is FallbackMode.FALLBACK
    # infra fault -> HALT from any mode
    assert pol.handle("INFRA_FAULT") is FallbackMode.HALT
    # repeated fault event while already HALTed: illegal event changes nothing
    with pytest.raises(IllegalTransitionError):
        pol.handle("INFRA_FAULT")
    assert pol.state is FallbackMode.HALT


def test_halt_recovery_requires_operator_approval(tmp_path):
    """F-001 regression: infra recovery must NOT auto-return to PRIMARY, and
    the audit log must not fabricate an operator confirmation."""
    pol = FallbackPolicy(str(tmp_path / "state.json"), fallback_accepted=True)
    pol.handle("INFRA_FAULT")
    # automatic step lands in the RESTRICTED observation state, never PRIMARY
    assert pol.on_session(primary_healthy=True, infra_healthy=True) is FallbackMode.RECOVERY
    # explicit approval requires evidence
    pol.sessions_since_recovery = pol.cooldown_sessions
    with pytest.raises(IllegalTransitionError):
        pol.handle("OPERATOR_RECOVERY_APPROVED", evidence="")
    assert pol.state is FallbackMode.RECOVERY
    # cooldown not elapsed -> approval still refused
    pol.sessions_since_recovery = 0
    with pytest.raises(IllegalTransitionError):
        pol.handle("OPERATOR_RECOVERY_APPROVED", evidence="ticket-42")
    pol.sessions_since_recovery = pol.cooldown_sessions
    assert pol.handle("OPERATOR_RECOVERY_APPROVED", evidence="ticket-42") is FallbackMode.PRIMARY
    # audit trail records the real transition with the evidence, no fabricated approval
    last = pol.history[-1]
    assert last["event"] == "OPERATOR_RECOVERY_APPROVED"
    assert last["from"] == "RECOVERY" and last["to"] == "PRIMARY"
    assert last["evidence"] == "ticket-42"


def test_fallback_requires_independent_acceptance(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"), fallback_accepted=False)
    pol.handle("MODEL_DEGRADED")
    with pytest.raises(IllegalTransitionError):
        pol.handle("PROMOTE_FALLBACK")
    assert pol.state is FallbackMode.DEGRADED  # blocked, not moved


def test_recovery_entry_requires_shadow_and_cooldown(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"), fallback_accepted=True)
    pol.handle("MODEL_DEGRADED")
    with pytest.raises(IllegalTransitionError):
        pol.handle("MODEL_RECOVERED", shadow_ok=False)
    with pytest.raises(IllegalTransitionError):
        pol.handle("MODEL_RECOVERED", shadow_ok=True)  # cooldown 0/10
    pol.sessions_since_recovery = pol.cooldown_sessions
    assert pol.handle("MODEL_RECOVERED", shadow_ok=True) is FallbackMode.RECOVERY


def test_recovery_via_shadow_ok_still_gated_by_cooldown(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"), fallback_accepted=True)
    pol.handle("MODEL_DEGRADED")
    pol.sessions_since_recovery = pol.cooldown_sessions
    assert pol.handle("MODEL_RECOVERED", shadow_ok=True) is FallbackMode.RECOVERY
    with pytest.raises(IllegalTransitionError):
        pol.handle("SHADOW_OK")  # cooldown gate applies to the ->PRIMARY exit too
    pol.sessions_since_recovery = pol.cooldown_sessions
    assert pol.handle("SHADOW_OK") is FallbackMode.PRIMARY


def test_operator_halt_requires_evidence(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"))
    with pytest.raises(IllegalTransitionError):
        pol.handle("OPERATOR_HALT", evidence="")
    assert pol.state is FallbackMode.PRIMARY


def test_cooldown_counts_unique_sessions_only(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"))
    pol.on_session(primary_healthy=True, infra_healthy=True, session="2026-01-05")
    pol.on_session(primary_healthy=True, infra_healthy=True, session="2026-01-05")  # duplicate call
    assert pol.sessions_since_recovery == 1
    pol.on_session(primary_healthy=True, infra_healthy=True, session="2026-01-08")
    assert pol.sessions_since_recovery == 2


def test_corrupt_state_file_fails_conservatively(tmp_path):
    path = str(tmp_path / "state.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"mode": "GARBAGE", "history": "not-a-list"')
    pol = FallbackPolicy(path)
    with pytest.raises(Exception) as exc:
        pol.load()
    # conservative: corrupted persistence never silently resets to PRIMARY
    assert not isinstance(exc.value, FileNotFoundError)
    assert pol.state is FallbackMode.PRIMARY  # untouched in-memory default; caller must halt


def test_state_persists_and_reloads(tmp_path):
    path = str(tmp_path / "state.json")
    pol = FallbackPolicy(path, fallback_accepted=True)
    pol.handle("MODEL_DEGRADED", reason="brier breach", session="2026-01-05")
    pol2 = FallbackPolicy(path).load()
    assert pol2.state is FallbackMode.DEGRADED
    assert pol2.history[-1]["reason"] == "brier breach"


def test_session_hook_halts_on_infra_fault(tmp_path):
    pol = FallbackPolicy(str(tmp_path / "state.json"))
    assert pol.on_session(primary_healthy=True, infra_healthy=False) is FallbackMode.HALT
