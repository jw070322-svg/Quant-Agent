"""Immutable model/policy bundle registry.

A bundle is a run's frozen evidence: config fingerprints, folds, metrics,
report. Promotion (candidate -> approved) is explicit and audited; the
research/decision pipeline refuses unapproved bundles for paper decisions.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import PROJECT_ROOT
from ..util import now_utc_iso, read_json, write_json

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
BUNDLE_FILES = ("config.json", "metrics.json", "folds.json", "report.md",
                "health.json", "oos_predictions.csv", "equity_curve.csv",
                "model.pkl")

# F-003 gate: approval writes are validated — no arbitrary status strings,
# no illegal transitions (e.g. approved -> candidate resurrection), and every
# write records its actor and evidence in the append-only history.
LEGAL_STATUSES = {"candidate", "approved", "revoked", "expired"}
LEGAL_STATUS_TRANSITIONS = {
    "candidate": {"approved", "revoked", "expired"},
    "approved": {"revoked", "expired"},
    "revoked": set(),    # terminal: re-approval requires a NEW bundle
    "expired": {"revoked"},
}


def _validate_status_transition(current: str, target: str) -> None:
    if target not in LEGAL_STATUSES:
        raise ValueError(
            f"illegal status '{target}'; legal statuses: {sorted(LEGAL_STATUSES)}")
    if target != current and target not in LEGAL_STATUS_TRANSITIONS.get(current, set()):
        raise ValueError(
            f"illegal transition {current} -> {target}; legal from '{current}': "
            f"{sorted(LEGAL_STATUS_TRANSITIONS.get(current, set()))}")


def new_run_dir(strategy: str, market: str, run_id: str) -> Path:
    d = RUNS_DIR / run_id
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


def save_bundle(run_dir: str | Path, *, status: str = "candidate",
                note: str = "", overwrite: bool = False,
                actor: str = "research-runner") -> Path:
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "config.json")
    _validate_status_transition("candidate", status)  # a new bundle starts at candidate
    bundle_id = meta["run_id"]
    dst = MODELS_DIR / bundle_id
    reg_path = dst / "REGISTRY.json"
    if reg_path.exists() and not overwrite:
        raise FileExistsError(
            f"bundle '{bundle_id}' already registered; refusing silent overwrite "
            "(immutability). Pass overwrite=True only with an explicit audit note.")
    dst.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)
    history = []
    if reg_path.exists():
        history = read_json(reg_path).get("history", [])
    history.append({"ts_utc": now_utc_iso(), "status": status, "note": note,
                    "actor": actor, "evidence": "", "from": "candidate",
                    "to": status})
    write_json(reg_path, {
        "bundle_id": bundle_id,
        "strategy": meta.get("strategy"),
        "market": meta.get("market"),
        "config_fingerprint": meta.get("config_fingerprint"),
        "model_sha256": meta.get("model_sha256"),
        "status": status,
        "history": history,
    })
    return dst


def set_status(bundle_id: str, status: str, note: str = "",
               actor: str = "unknown", evidence: str = "") -> None:
    reg_path = MODELS_DIR / bundle_id / "REGISTRY.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"unknown bundle: {bundle_id}")
    reg = read_json(reg_path)
    prev = reg.get("status", "candidate")
    _validate_status_transition(prev, status)
    reg["status"] = status
    reg["history"].append({
        "ts_utc": now_utc_iso(), "status": status, "note": note,
        "actor": actor, "evidence": str(evidence), "from": prev, "to": status,
    })
    write_json(reg_path, reg)


def get_status(bundle_id: str) -> dict:
    reg_path = MODELS_DIR / bundle_id / "REGISTRY.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"unknown bundle: {bundle_id}")
    return read_json(reg_path)


def list_bundles() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in sorted(MODELS_DIR.iterdir()):
        reg = d / "REGISTRY.json"
        if reg.exists():
            out.append(read_json(reg))
    return out
