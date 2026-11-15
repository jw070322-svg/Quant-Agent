"""Ingestion pipeline: raw snapshots, schema normalization, validation, manifest.

Fixes the handover defects:
- One date schema enforced at ingestion (ISO session dates, ascending, unique).
- OHLC-inconsistent / nonfinite rows are quarantined with reasons, not clipped.
- Adjustment factors are applied BEFORE any downstream indicator (single
  shared pipeline), and adjustment status is recorded per symbol.
- Volume units are resolved by an amount/volume cross-check instead of an
  assumption, and recorded in the manifest.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_market_config
from ..util import ensure_dir, now_utc_iso, sha256_file, write_json
from .providers import (PROVIDER_CN_SOURCE, PROVIDER_NAME, PROVIDER_US_SOURCE,
                        update_summary)
from .schema import (ADJ_STATUS_FACTOR, ADJ_STATUS_UNKNOWN, CANONICAL_COLS,
                     validate_bars)

SCHEMA_SIMPLE = "simple"    # date,symbol,open,high,low,close,volume
SCHEMA_TUSHARE = "tushare"  # ts_code,trade_date,open,...,[adj_factor]

NORMALIZED_COLS = ["date", "open", "high", "low", "close", "volume",
                   "adj_factor", "open_adj", "high_adj", "low_adj", "close_adj"]


def detect_schema(columns: list[str]) -> str:
    cols = {c.strip().strip('"').lower() for c in columns}
    if "trade_date" in cols and "ts_code" in cols:
        return SCHEMA_TUSHARE
    if "date" in cols and "close" in cols:
        return SCHEMA_SIMPLE
    raise ValueError(f"unrecognized CSV schema with columns: {sorted(cols)}")


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    return df


def _normalize_simple(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    need = {"date", "open", "high", "low", "close", "volume"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"simple schema missing columns: {sorted(missing)}")
    df = raw[list(need)].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["symbol"] = symbol
    if "adj_factor" in raw.columns:
        # provider updates carry a real adjustment-factor series
        df["adj_factor"] = pd.to_numeric(raw["adj_factor"], errors="coerce")
        df["adjustment_status"] = ADJ_STATUS_FACTOR
    else:
        df["adj_factor"] = 1.0
        df["adjustment_status"] = ADJ_STATUS_UNKNOWN
    df["amount"] = pd.to_numeric(raw["amount"], errors="coerce") if "amount" in raw.columns else np.nan
    df["volume_raw"] = df["volume"]
    df = df.drop(columns=["volume"])
    return df


def _normalize_tushare(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    need = {"trade_date", "open", "high", "low", "close", "vol"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"tushare schema missing columns: {sorted(missing)}")
    df = raw[list(need)].copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.rename(columns={"vol": "volume_raw"})
    df["symbol"] = symbol
    if "adj_factor" in raw.columns:
        df["adj_factor"] = pd.to_numeric(raw["adj_factor"], errors="coerce")
        df["adjustment_status"] = ADJ_STATUS_FACTOR
    else:
        df["adj_factor"] = 1.0
        df["adjustment_status"] = ADJ_STATUS_UNKNOWN
    df["amount"] = pd.to_numeric(raw["amount"], errors="coerce") if "amount" in raw.columns else np.nan
    return df


def _resolve_volume_unit(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Determine raw volume unit and amount currency scale from data.

    r = median(amount / (volume_raw * close)) identifies the pair
    (amount_unit, shares_per_raw_volume_unit):
      r ≈ 1    -> amount in currency,  volume_raw in shares
      r ≈ 0.1  -> amount in thousands, volume_raw in lots (x100 shares)
      r ≈ 0.001-> amount in thousands, volume_raw in shares
    The conclusion is recorded, never assumed (handover 5.4: volume units).
    """
    df = df.copy()
    if "amount" not in df.columns or df["amount"].notna().sum() < 10:
        df["volume"] = df["volume_raw"]
        return df, "shares_assumed_no_amount_check", "unverified"
    close, amount, vol = df["close"], df["amount"], df["volume_raw"]
    ok = np.isfinite(amount) & np.isfinite(vol) & (vol > 0) & (close > 0)
    if ok.sum() < 10:
        df["volume"] = vol
        return df, "shares_assumed_insufficient_amount", "unverified"
    r = float((amount[ok] / (vol[ok] * close[ok])).median())
    if abs(r - 1.0) <= 0.25:
        df["volume"] = vol
        return df, "shares", "currency"
    if abs(r * 10.0 - 1.0) <= 0.25:
        df["volume"] = vol * 100.0
        return df, "lots_x100_converted", "thousands"
    if abs(r * 1000.0 - 1.0) <= 0.25:
        df["volume"] = vol
        return df, "shares", "thousands"
    df["volume"] = vol
    return df, "unresolved", "unresolved"


def normalize_symbol(path: Path, symbol: str) -> tuple[pd.DataFrame, dict]:
    raw = read_raw(path)
    schema = detect_schema(list(raw.columns))
    if schema == SCHEMA_SIMPLE:
        df = _normalize_simple(raw, symbol)
    else:
        df = _normalize_tushare(raw, symbol)
    df, vol_unit, amount_unit = _resolve_volume_unit(df)

    clean, quarantine, review_flags = validate_bars(df[["date", "symbol", "open", "high", "low",
                                                        "close", "volume", "adj_factor",
                                                        "adjustment_status", "amount"]])
    # back-adjusted prices: latest session's adjusted price equals raw price
    factor = clean["adj_factor"].astype(float)
    norm = factor / factor.iloc[-1]
    clean["open_adj"] = clean["open"] * norm
    clean["high_adj"] = clean["high"] * norm
    clean["low_adj"] = clean["low"] * norm
    clean["close_adj"] = clean["close"] * norm
    clean = clean.drop(columns=["amount"], errors="ignore").sort_values("date").reset_index(drop=True)

    meta = {
        "schema": schema,
        "volume_unit": vol_unit,
        "amount_unit": amount_unit,
        "adjustment_status": str(clean["adjustment_status"].iloc[0]),
        "rows": int(len(clean)),
        "date_start": str(clean["date"].iloc[0].date()),
        "date_end": str(clean["date"].iloc[-1].date()),
        "quarantined": quarantine,
        "hard_quarantine_count": len(quarantine),
        "review_flags": review_flags,
        "review_flag_count": len(review_flags),
    }
    return clean[NORMALIZED_COLS], meta


def ingest_market(market: str, *, as_of: str | None = None) -> dict:
    """Snapshot + normalize + validate all universe symbols. Writes the
    manifest to datasets/manifest_{market}.json and returns it.

    Source preference: a provider update file under datasets/updates/{market}/
    wins over the legacy snapshot; provenance is recorded per symbol.
    """
    cfg = load_market_config(market)
    source_dir = PROJECT_ROOT / cfg["source_dir"]
    update_dir = PROJECT_ROOT / "datasets" / "updates" / market
    raw_dir = ensure_dir(PROJECT_ROOT / "datasets" / "raw" / market)
    norm_dir = ensure_dir(PROJECT_ROOT / "datasets" / "normalized" / market)

    manifest: dict = {
        "market": market,
        "generated_for_universe": list(cfg["universe"]),
        "symbols": {},
        "project_data_inventory": [],
    }

    for symbol in cfg["universe"]:
        update_src = update_dir / f"{symbol}.csv"
        legacy_src = source_dir / f"{symbol}.csv"
        if update_src.exists():
            src, source_kind = update_src, "provider_update"
        elif legacy_src.exists():
            src, source_kind = legacy_src, "legacy_snapshot"
        else:
            manifest["symbols"][symbol] = {
                "symbol": symbol, "usable": False,
                "reason": f"no source file ({update_src} or {legacy_src})"}
            continue
        snap = raw_dir / f"{symbol}.csv"
        # snapshot immutability: if the chosen source changed since the last
        # snapshot, archive the previous snapshot instead of clobbering it
        if snap.exists() and sha256_file(snap) != sha256_file(src):
            archive = ensure_dir(raw_dir / "_archive")
            stamp = now_utc_iso().replace(":", "").replace("-", "")
            snap.rename(archive / f"{symbol}.{stamp}.csv")
        shutil.copy2(src, snap)
        entry: dict = {"symbol": symbol, "usable": False, "source_kind": source_kind,
                       "source_path": str(src.relative_to(PROJECT_ROOT))}
        if source_kind == "provider_update":
            entry["provider"] = PROVIDER_NAME
            entry["provider_cn_source"] = PROVIDER_CN_SOURCE
            entry["provider_us_source"] = PROVIDER_US_SOURCE
            upd = update_summary(market, root=PROJECT_ROOT)
            entry["fetched_at_utc"] = upd["fetched_at_utc"] if upd else None
        else:
            entry["source_path"] = str(src.relative_to(PROJECT_ROOT))
        entry["source_sha256"] = sha256_file(snap)
        try:
            clean, meta = normalize_symbol(snap, symbol)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            entry["reason"] = f"normalization failed: {exc}"
            manifest["symbols"][symbol] = entry
            continue

        out = norm_dir / f"{symbol}.csv"
        clean.to_csv(out, index=False)
        entry.update(meta)
        entry["normalized_sha256"] = sha256_file(out)
        entry["usable"] = len(clean) >= 100
        if not entry["usable"]:
            entry["reason"] = f"too few clean rows ({len(clean)})"
        if as_of:
            # staleness honors the configured tolerance: a bar from yesterday
            # is fresh; string comparison alone would call it stale
            age_days = (pd.Timestamp(as_of) - pd.Timestamp(entry["date_end"])).days
            entry["age_days_at_ingest"] = int(age_days)
            entry["stale_at_ingest"] = bool(age_days > int(cfg["stale_session_days"]))
        manifest["symbols"][symbol] = entry

    # Phase 0 evidence preservation: hash every CSV currently under data/
    for p in sorted((PROJECT_ROOT / "data").rglob("*.csv")):
        manifest["project_data_inventory"].append({
            "path": str(p.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })

    write_json(PROJECT_ROOT / "datasets" / f"manifest_{market}.json", manifest)
    return manifest


def load_normalized(market: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load normalized frames for usable symbols only. Raises on missing
    normalized data so callers cannot mistake absence for a complete run."""
    from ..util import read_json
    manifest = read_json(PROJECT_ROOT / "datasets" / f"manifest_{market}.json")
    if manifest.get("market") != market:
        raise RuntimeError(f"manifest is for market '{manifest.get('market')}', not '{market}'")
    cfg = load_market_config(market)
    symbols = symbols or cfg["universe"]
    norm_dir = PROJECT_ROOT / "datasets" / "normalized" / market
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        entry = manifest["symbols"].get(symbol)
        if not entry or not entry.get("usable"):
            raise RuntimeError(
                f"symbol '{symbol}' has no usable normalized data for market "
                f"'{market}' — run `qa ingest --market {market}` and resolve "
                "quarantines first (complete-run gate)."
            )
        df = pd.read_csv(norm_dir / f"{symbol}.csv", parse_dates=["date"])
        frames[symbol] = df
    return frames
