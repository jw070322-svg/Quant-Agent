"""Registry approval gate (F-003): legal status set, legal transitions,
actor recorded in append-only history."""
import pytest

from quant_agent.registry.store import (LEGAL_STATUSES,
                                         LEGAL_STATUS_TRANSITIONS,
                                         save_bundle, set_status)
from quant_agent.util import write_json


def _bundle(tmp_path, monkeypatch, status="candidate"):
    import quant_agent.registry.store as store_mod
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "config.json", {"run_id": "gate-test", "strategy": "s",
                                         "market": "us"})
    monkeypatch.setattr(store_mod, "MODELS_DIR", models)
    monkeypatch.setattr(store_mod, "RUNS_DIR", tmp_path)
    save_bundle(run_dir, status=status)
    return models


def test_legal_status_sets_are_defined():
    assert LEGAL_STATUSES == {"candidate", "approved", "revoked", "expired"}
    # terminal states have no outgoing transitions except defined ones
    assert LEGAL_STATUS_TRANSITIONS["revoked"] == set()


def test_arbitrary_status_string_rejected(tmp_path, monkeypatch):
    _bundle(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="illegal status"):
        set_status("gate-test", "super-approved-by-boss")


def test_illegal_transition_rejected(tmp_path, monkeypatch):
    _bundle(tmp_path, monkeypatch, status="approved")
    # approved -> candidate would silently resurrect a revoked candidate
    with pytest.raises(ValueError, match="illegal transition"):
        set_status("gate-test", "candidate")
    set_status("gate-test", "revoked", actor="risk-owner")  # legal transition
    with pytest.raises(ValueError, match="illegal transition"):
        set_status("gate-test", "approved")  # revoked is terminal


def test_approval_records_actor_and_evidence(tmp_path, monkeypatch):
    _bundle(tmp_path, monkeypatch)
    set_status("gate-test", "approved", note="meets DECISIONS 2.1",
               actor="risk-owner", evidence="metrics.json sha abc123")
    from quant_agent.registry.store import get_status
    reg = get_status("gate-test")
    assert reg["status"] == "approved"
    last = reg["history"][-1]
    assert last["actor"] == "risk-owner"
    assert last["evidence"] == "metrics.json sha abc123"
    assert last["from"] == "candidate" and last["to"] == "approved"
