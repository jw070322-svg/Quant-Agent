"""registry -> paper-decide mechanism test (handover §6 P0 integration + F-002 gates).

Injects a synthetic bundle into a TEMP registry (never the real one) and
verifies: unapproved bundles are refused; decisions use the FROZEN model
artifact (no refit) with SHA-256 verified; dataset drift is refused; holdings
feed equity and generate SELL intents; stale data blocks BUYs while allowing
de-risking SELLs; pending orders freeze cash; the decision record lands on
disk with provenance.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import quant_agent.registry.store as store_mod
import quant_agent.research.decide as decide_mod
from quant_agent.alpha.expected_return import PayoffModel
from quant_agent.config import load_strategy_config
from quant_agent.research.models import make_model
from quant_agent.util import ensure_dir, sha256_file, write_json

SYMS = ("ALPHA", "BETA")
FEATURES = ["ret_1", "ret_20", "vol_20"]  # short warm-up for tiny test frames


def make_frames(n=120, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    frames = {}
    for sym in SYMS:
        steps = rng.normal(0.0003, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
        open_ = np.clip(close * (1 + rng.normal(0, 0.002, n)), low, high)
        vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
        frames[sym] = pd.DataFrame({
            "date": dates, "open": open_, "high": high, "low": low,
            "close": close, "volume": vol, "adj_factor": 1.0,
            "open_adj": open_, "high_adj": high, "low_adj": low, "close_adj": close,
        })
    return frames


def _fit_artifact(frames):
    """Registration-time artifact fit (same contract as research.runner).
    Deterministic: a linearly separable logistic model so p_up splits on the
    first feature's sign, plus a directly-constructed payoff model."""
    rng = np.random.default_rng(11)
    # train at the SAME scale as real feature values (~0.01) so the fitted
    # model discriminates on realistic inference inputs
    X = rng.normal(0, 0.02, size=(200, len(FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    model = make_model("logistic", {"C": 10.0, "max_iter": 200}, 42)
    model.fit(X, y)
    payoff = PayoffModel(
        e_event={s: 0.01 for s in SYMS},
        e_complement={s: -0.005 for s in SYMS},
        global_event=0.01, global_complement=-0.005, shrink=0.5)
    return {"model": model, "payoff": payoff, "feature_names": FEATURES,
            "kind": "logistic", "trained_through": "2025-06-30"}


def setup_env(tmp_path, monkeypatch, *, status, manifest_date_end,
              drift=False, with_model=True):
    """Temp registry + temp project root + stubbed market data + artifact."""
    models = tmp_path / "models"
    proj = tmp_path / "proj"
    ensure_dir(models)
    ensure_dir(proj / "datasets")

    artifact = _fit_artifact(make_frames())
    model_path = models / "mech-test" / "model.pkl"
    ensure_dir(model_path.parent)
    if with_model:
        model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    model_sha = sha256_file(model_path) if with_model else None

    scfg = load_strategy_config("us_research_v1")
    scfg["features"] = FEATURES
    # deterministic gate tests: every predicted symbol gets a target weight,
    # so intent generation does not depend on the artifact's edge filtering
    scfg["portfolio"]["edge_threshold"] = -1.0
    cfg = {
        "run_id": "mech-test", "market": "us",
        "strategy_config": scfg,
        "config_fingerprint": "mech-test",
        "dataset_source_sha256": {s: f"hash-{s}" for s in SYMS},
        "created_utc": "mech-test",
        "model_sha256": model_sha,
        "model_kind": "gbm",
    }
    write_json(models / "mech-test" / "config.json", cfg)
    write_json(models / "mech-test" / "REGISTRY.json",
               {"bundle_id": "mech-test", "status": status,
                "model_sha256": model_sha, "history": []})

    manifest_syms = {}
    for s in SYMS:
        entry = {"usable": True, "date_end": manifest_date_end,
                 "source_sha256": f"hash-{s}"}
        if drift:
            entry["source_sha256"] = f"drifted-{s}"
        manifest_syms[s] = entry
    write_json(proj / "datasets" / "manifest_us.json",
               {"market": "us", "symbols": manifest_syms})

    monkeypatch.setattr(decide_mod, "MODELS_DIR", models)
    monkeypatch.setattr(store_mod, "MODELS_DIR", models)
    monkeypatch.setattr(decide_mod, "PROJECT_ROOT", proj)
    monkeypatch.setattr(decide_mod, "load_normalized",
                        lambda market, symbols=None: make_frames())
    from quant_agent.config import load_market_config

    real_us = load_market_config("us")

    def fake_market_config(market):
        cfg = dict(real_us)
        cfg["universe"] = list(SYMS)  # synthetic universe for the stub data
        return cfg

    monkeypatch.setattr(decide_mod, "load_market_config", fake_market_config)
    return proj


def test_unapproved_bundle_refused(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch, status="candidate", manifest_date_end="2099-01-01")
    with pytest.raises(PermissionError, match="not approved"):
        decide_mod.paper_decide("mech-test")


def test_approved_bundle_uses_frozen_artifact(tmp_path, monkeypatch):
    proj = setup_env(tmp_path, monkeypatch, status="approved",
                     manifest_date_end="2099-01-01")
    record = decide_mod.paper_decide("mech-test", initial_equity=1_000_000.0)
    assert record["bundle_id"] == "mech-test"
    src = record["model_source"]
    assert src["refit_at_decision"] is False
    assert src["sha256"] and len(src["sha256"]) == 64
    assert "PAPER INTENTS ONLY" in record["mode"]
    assert sum(record["target_weights"].values()) <= 0.80 + 1e-9
    assert max(record["target_weights"].values()) <= 0.20 + 1e-9
    assert record["inputs"]["equity_estimate"] == pytest.approx(1_000_000.0)
    assert record["approved_intents"], "fresh data + cash should approve buys"
    assert all(i["side"] == "BUY" for i in record["approved_intents"])
    files = list((proj / "artifacts" / "decisions").glob("decision_mech-test_*.json"))
    assert len(files) == 1


def test_tampered_model_artifact_refused(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch, status="approved", manifest_date_end="2099-01-01")
    # overwrite the artifact with different bytes -> hash mismatch
    model_path = decide_mod.MODELS_DIR / "mech-test" / "model.pkl"
    model_path.write_bytes(b"tampered")
    with pytest.raises(PermissionError, match="hash mismatch"):
        decide_mod.paper_decide("mech-test")


def test_missing_model_artifact_refused(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch, status="approved", manifest_date_end="2099-01-01",
              with_model=False)
    with pytest.raises(PermissionError, match="no frozen model artifact"):
        decide_mod.paper_decide("mech-test")


def test_dataset_drift_refused(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch, status="approved", manifest_date_end="2099-01-01",
              drift=True)
    with pytest.raises(PermissionError, match="dataset drift"):
        decide_mod.paper_decide("mech-test")


def test_holdings_drive_equity_and_sell_intents(tmp_path, monkeypatch):
    setup_env(tmp_path, monkeypatch, status="approved", manifest_date_end="2099-01-01")
    record = decide_mod.paper_decide("mech-test", initial_equity=1_000_000.0,
                                     cash=200_000.0,
                                     holdings={"ALPHA": 5000.0})  # ~500k position
    assert record["inputs"]["holdings"] == {"ALPHA": 5000.0}
    assert record["inputs"]["equity_estimate"] > 600_000.0
    sides = [i["side"] for i in record["approved_intents"]]
    assert "SELL" in sides, "a ~31% position must be trimmed toward the 20% cap"


def test_stale_data_blocks_buys_but_allows_derisking_sells(tmp_path, monkeypatch):
    proj = setup_env(tmp_path, monkeypatch, status="approved",
                     manifest_date_end="2026-01-01")  # far in the past -> stale
    record = decide_mod.paper_decide("mech-test", initial_equity=1_000_000.0,
                                     cash=200_000.0, holdings={"ALPHA": 5000.0})
    assert record["inputs"]["stale_symbols_blocked"] == ["ALPHA", "BETA"]
    approved_sides = [i["side"] for i in record["approved_intents"]]
    assert all(s == "SELL" for s in approved_sides), \
        "stale incident: no new risk (BUY), de-risking (SELL) still allowed"
    assert any("stale" in r for rej in record["rejected_intents"]
               for r in rej["rejected_reasons"])


def test_pending_orders_freeze_cash(tmp_path, monkeypatch):
    """F-004 decision-side: pending BUY notional reduces cash for new buys."""
    setup_env(tmp_path, monkeypatch, status="approved", manifest_date_end="2099-01-01")
    pending = [{"symbol": "ALPHA", "side": "BUY", "qty": 40000.0, "ref_price": 100.0}]
    record = decide_mod.paper_decide("mech-test", initial_equity=1_000_000.0,
                                     cash=1_000_000.0,
                                     pending_orders=pending)  # 4M > cash! frozen
    assert record["inputs"]["pending_buy_value_frozen"] == pytest.approx(4_000_000.0)
    buys = [i for i in record["approved_intents"] if i["side"] == "BUY"]
    assert buys == [], "cash (1M) net of frozen (4M) must reject all buys"
    assert any("cash" in r for rej in record["rejected_intents"]
               for r in rej["rejected_reasons"])


def test_real_registry_refuses_mech_test_bundle():
    """The mechanism bundle must never leak into the REAL registry: the real
    store has no 'mech-test' entry, so get_status must fail."""
    from quant_agent.registry.store import get_status
    with pytest.raises(FileNotFoundError):
        get_status("mech-test")
