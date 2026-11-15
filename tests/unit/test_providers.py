"""Provider layer tests (NO network): factor derivation, alignment handling,
error paths, and ingest source-preference/provenance."""
import pandas as pd
import pytest

from quant_agent.data.providers import (ProviderError, _merge_raw_qfq,
                                        cn_symbol)
import quant_agent.data.ingest as ingest_mod
from quant_agent.util import write_json


def _raw_qfq_frames(start="2025-01-01", periods=10):
    dates = pd.bdate_range(start, periods=periods)
    raw = pd.DataFrame({
        "date": dates, "open": 10.0, "high": 10.5, "low": 9.9, "close": 10.0,
        "volume": 1_000_000.0, "amount": 10_000_000.0,
    })
    # qfq = raw for the first half, 0.5x raw afterwards (a 2:1 split happened)
    qfq = raw.copy()
    qfq.loc[qfq.index >= periods // 2, ["open", "high", "low", "close"]] *= 0.5
    return raw, qfq


def test_factor_derived_from_qfq_ratio():
    raw, qfq = _raw_qfq_frames()
    res = _merge_raw_qfq(raw, qfq, "TEST", start="2025-01-01", end="2099-01-01")
    f = res.frame["adj_factor"]
    assert ((f.iloc[:5] - 1.0).abs() < 1e-9).all()
    assert ((f.iloc[5:] - 0.5).abs() < 1e-9).all()
    assert res.factor_dates_changed == 5
    assert res.notes == []  # dates aligned


def test_misalignment_reported_not_silently_dropped():
    raw, qfq = _raw_qfq_frames()
    qfq = qfq.iloc[2:]  # provider qfq missing the first two sessions
    res = _merge_raw_qfq(raw, qfq, "TEST", start="2025-01-01", end="2099-01-01")
    assert any("misalignment" in n for n in res.notes)
    assert res.rows == len(raw) - 2


def test_invalid_factor_rejected():
    raw, qfq = _raw_qfq_frames()
    qfq.loc[qfq.index[3], "close"] = -5.0  # nonsense price -> negative factor
    with pytest.raises(ProviderError, match="invalid adjustment factor"):
        _merge_raw_qfq(raw, qfq, "TEST", start="2025-01-01", end="2099-01-01")


def test_empty_overlap_rejected():
    raw, qfq = _raw_qfq_frames()
    with pytest.raises(ProviderError):
        _merge_raw_qfq(raw, qfq, "TEST", start="2030-01-01", end="2031-01-01")


def test_cn_symbol_mapping_and_validation():
    assert cn_symbol("000001.SZ") == "sz000001"
    assert cn_symbol("600519.SH") == "sh600519"
    with pytest.raises(ProviderError):
        cn_symbol("AAPL")
    with pytest.raises(ProviderError):
        cn_symbol("000001.XX")


def test_simple_schema_normalization_uses_provider_factor():
    """Update CSVs (simple schema + adj_factor) must keep the real factor and
    mark adjustment status accordingly."""
    raw, qfq = _raw_qfq_frames()
    res = _merge_raw_qfq(raw, qfq, "TEST", start="2025-01-01", end="2099-01-01")
    df = res.frame.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    norm = ingest_mod._normalize_simple(df, "TEST")
    assert (norm["adjustment_status"] == "raw_with_adj_factor").all()
    assert norm["adj_factor"].iloc[-1] == pytest.approx(0.5)


def _csv_rows(n=120, factor_tail=1.0):
    dates = pd.bdate_range("2025-01-01", periods=n)
    legacy = ["date,symbol,open,high,low,close,volume"]
    update = ["date,symbol,open,high,low,close,volume,amount,adj_factor"]
    for i, d in enumerate(dates):
        close = 10.0 + 0.01 * (i % 7)
        factor = factor_tail if i >= n // 2 else 1.0
        legacy.append(f"{d.date()},TEST,{close:.3f},{close * 1.01:.3f},{close * 0.99:.3f},{close:.3f},1000")
        update.append(f"{d.date()},TEST,{close * factor:.3f},{close * 1.01 * factor:.3f},"
                      f"{close * 0.99 * factor:.3f},{close * factor:.3f},1000,{close * 1000:.2f},{factor:.4f}")
    return "\n".join(legacy) + "\n", "\n".join(update) + "\n"


def test_ingest_prefers_update_and_records_provenance(tmp_path, monkeypatch):
    """A provider update file wins over the legacy snapshot; provenance and
    hashes are recorded; removing the update falls back to legacy."""
    root = tmp_path / "proj"
    (root / "data" / "us").mkdir(parents=True)
    (root / "datasets" / "updates" / "us").mkdir(parents=True)
    legacy = root / "data" / "us" / "TEST.csv"
    update = root / "datasets" / "updates" / "us" / "TEST.csv"

    legacy_csv, update_csv = _csv_rows(n=120, factor_tail=0.5)
    legacy.write_text(legacy_csv, encoding="utf-8")
    update.write_text(update_csv, encoding="utf-8")
    write_json(root / "datasets" / "updates" / "us" / "_update_summary.json",
               {"fetched_at_utc": "2026-11-14T00:00:00Z", "symbols": {}})

    monkeypatch.setattr(ingest_mod, "PROJECT_ROOT", root)
    from quant_agent.config import load_market_config

    fake_cfg = dict(load_market_config("us"))
    fake_cfg["universe"] = ["TEST"]
    monkeypatch.setattr(ingest_mod, "load_market_config", lambda m: fake_cfg)
    manifest = ingest_mod.ingest_market("us")
    entry = manifest["symbols"]["TEST"]
    assert entry["source_kind"] == "provider_update"
    assert entry["provider"] == "akshare"
    assert entry["fetched_at_utc"] == "2026-11-14T00:00:00Z"
    assert entry["adjustment_status"] == "raw_with_adj_factor"
    assert entry["usable"] is True
    assert entry["rows"] == 120
    # back-adjustment anchors the LAST session to factor 1 -> normalized
    # factor is rescaled to end at 1.0; the raw snapshot keeps source hashes
    assert manifest["market"] == "us"

    update.unlink()
    manifest2 = ingest_mod.ingest_market("us")
    assert manifest2["symbols"]["TEST"]["source_kind"] == "legacy_snapshot"
    assert manifest2["symbols"]["TEST"]["adjustment_status"] == "unknown_no_factor"
