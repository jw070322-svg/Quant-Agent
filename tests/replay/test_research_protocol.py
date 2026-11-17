"""Research protocol invariants on synthetic panel data (review-doc P0:
time leakage / unique-OOS / activation schedule), plus a real-data audit reconciliation test."""
import numpy as np
import pandas as pd
import pytest


from quant_agent.features.pipeline import build_panel
from quant_agent.labels.forward import LabelSpec, add_labels, training_eligible
from quant_agent.research.splits import build_folds
SYMS = ("ALPHA", "BETA", "GAMMA")


def make_frames(n=420, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
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


def build_labeled_panel(frames):
    panel = build_panel(frames, ["ret_1", "ret_5", "ret_20", "close_over_ma20", "vol_20"])
    for sym, df in panel.items():
        df["symbol"] = sym
    spec = LabelSpec()
    return {sym: add_labels(df, spec) for sym, df in panel.items()}


def test_every_fitting_label_matured_before_fit_cutoff():
    frames = make_frames()
    panel = build_labeled_panel(frames)
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames.values()]).unique()))
    folds = build_folds(sessions, initial_train_fraction=0.5, test_sessions=40,
                        step_sessions=40, policy_tune_fraction=0.5)
    for fold in folds:
        for sym, df in panel.items():
            elig = df.loc[training_eligible(df, fit_cutoff=fold.fit_cutoff)]
            assert len(elig) > 0
            # the decisive anti-leakage property: no fitting label depends on
            # any price after the fit cutoff
            assert elig["label_maturity_date"].max() <= fold.fit_cutoff


def test_oos_windows_disjoint_and_contiguous():
    frames = make_frames()
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames.values()]).unique()))
    folds = build_folds(sessions, initial_train_fraction=0.5, test_sessions=40,
                        step_sessions=40, policy_tune_fraction=0.5)
    spans = [(f.predict_start, f.predict_end) for f in folds]
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 > e1, "OOS windows overlap -> double counting"
    # contiguity: each window starts the session after the previous ends
    idx = sessions.get_indexer([s for s, _ in spans] + [e for _, e in spans])
    k = len(spans)
    assert (idx[1:k] == idx[k:2 * k - 1] + 1).all()


def test_report_window_policy_selected_strictly_before():
    """Mirror the runner's activation schedule: a fold's REPORT window must
    use that fold's threshold, and its TUNE window must use the PREVIOUS
    fold's threshold (or the pre-registered default) — never its own."""
    frames = make_frames()
    sessions = pd.DatetimeIndex(sorted(
        pd.concat([df["date"] for df in frames.values()]).unique()))
    folds = build_folds(sessions, initial_train_fraction=0.5, test_sessions=40,
                        step_sessions=40, policy_tune_fraction=0.5)
    chosen = {i: 0.40 + 0.05 * i for i in range(len(folds))}
    DEFAULT = 0.55
    fold_for_date = {}
    for fold in folds:
        for t in sessions[(sessions >= fold.predict_start) & (sessions <= fold.predict_end)]:
            fold_for_date[t] = fold

    def active_threshold(t):
        fold = fold_for_date.get(t)
        if fold is None:
            return DEFAULT
        if t <= fold.tune_end:
            return chosen.get(fold.fold_id - 1, DEFAULT)
        return chosen.get(fold.fold_id, DEFAULT)

    for i, fold in enumerate(folds):
        for t in sessions[(sessions >= fold.tune_start) & (sessions <= fold.tune_end)]:
            expected = chosen.get(i - 1, DEFAULT)
            assert active_threshold(t) == expected, "tune window used its own choice"
        for t in sessions[(sessions >= fold.report_start) & (sessions <= fold.report_end)]:
            assert active_threshold(t) == chosen[i]


def test_real_data_quarantine_matches_audit_evidence():
    """The ingestion quarantine must reproduce the audit's three OHLC
    defects exactly on the LEGACY files (handover audit_evidence.json).
    Provider-updated data has zero quarantines; this pins the reconciliation
    to the original bundled snapshots."""
    from pathlib import Path
    from quant_agent.config import PROJECT_ROOT
    from quant_agent.data.ingest import normalize_symbol

    amzn, meta = normalize_symbol(PROJECT_ROOT / "data" / "us" / "AMZN.csv", "AMZN")
    googl, meta_g = normalize_symbol(PROJECT_ROOT / "data" / "us" / "GOOGL.csv", "GOOGL")
    amzn_dates = {q["date"] for q in meta["quarantined"]}
    googl_dates = {q["date"] for q in meta_g["quarantined"]}
    assert amzn_dates == {"2026-02-04", "2026-02-12"}
    assert googl_dates == {"2026-02-02"}
    assert all(q["reasons"] == ["ohlc_inconsistent"]
               for q in meta["quarantined"] + meta_g["quarantined"])
    # provider data for the same symbols must carry NO hard quarantines
    fresh = normalize_symbol(Path("datasets/updates/us/AMZN.csv"), "AMZN")[1]
    assert fresh["hard_quarantine_count"] == 0


def test_complete_run_gate_blocks_unusable_universe(tmp_path, monkeypatch):
    """A manifest marking a symbol unusable must block research (no
    complete-run claim over incomplete data).

    W0 evidence isolation: runs against a TEMPORARY project root with a copied
    manifest — the suite must never write repository datasets or manifests
    (the previous version called the real ingest_market twice, regenerating
    datasets and manifest hashes as a side effect)."""
    import json as _json

    import quant_agent.research.runner as runner_mod
    from quant_agent.config import PROJECT_ROOT as REAL_ROOT
    from quant_agent.util import write_json as _write_json

    tmp_proj = tmp_path / "proj"
    (tmp_proj / "datasets").mkdir(parents=True)
    real_manifest = _json.loads(
        (REAL_ROOT / "datasets" / "manifest_us.json").read_text(encoding="utf-8"))
    real_manifest["symbols"]["MSFT"]["usable"] = False
    real_manifest["symbols"]["MSFT"]["reason"] = "test-injected unusable"
    _write_json(tmp_proj / "datasets" / "manifest_us.json", real_manifest)

    monkeypatch.setattr(runner_mod, "PROJECT_ROOT", tmp_proj)
    with pytest.raises(RuntimeError, match="complete-run gate"):
        runner_mod.run_research("us_research_v1")
    # the real repository manifest was never touched
    live = (REAL_ROOT / "datasets" / "manifest_us.json").read_text(encoding="utf-8")
    assert "test-injected unusable" not in live
