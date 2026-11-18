"""CN canonical snapshot assembler (comprehensive plan 2026-11-17, W6).

Assembles the factor-bundle canonical snapshot from token-free akshare
sources, following the contract in `handover/FACTOR_BUNDLE_IMPLEMENTATION.md`
and enforced by `data/factor_snapshot.py::read_snapshot`:

- bars: RAW OHLC + volume + amount (Sina `stock_zh_a_daily`, adjust="") plus
  `tri`/`tri_open` computed from the qfq series — the qfq series is a
  split/dividend-consistent wealth index anchored to the latest bar (the
  documented TRI convention; reinvestment at ex-date close is assumed).
- float: daily `outstanding_share` (Sina) → float_shares;
  market_cap = close × outstanding_share; available_at = observation date
  (`availability_is_publication_date: false` convention).
- financials: annual rows joined from the three Sina statements, with
  available_at = latest of the three 公告日期 (announcement dates) — real
  publication-based availability. Current-vintage restatement disclosed.
  equity_issued = increase in 实收资本(或股本) between consecutive annual
  periods (par ≈ 1 CNY/share) — documented recipe mapping.
- actions: implemented 分红/送转 events from the Eastmoney fhps batch feed,
  per-10-share ratios ÷ 10; dividend applied before split on the same ex-date
  (both declared per pre-event share).
- universe: current CSI 300 + CSI 500 constituents (SURVIVORSHIP BIAS —
  historical membership is unavailable token-free; disclosed), listing dates
  and industry from Eastmoney/Sina, industry fallback "UNKNOWN".
- costs: the declared (UNVERIFIED) schedule from configs/markets/cn.yaml.

The assembler is resumable (per-symbol/period fetch caches under
datasets/factor_snapshots/_fetch_cache/) and writes to a NEW output directory
only. Everything regenerable — datasets/factor_snapshots/ is gitignored.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_market_config
from ..data.providers import (ProviderError, _retry, fetch_cn_daily_pair,
                              fetch_cn_universe)
from ..data.providers_fundamental import (fetch_cn_actions_batch,
                                          fetch_cn_financial_statements,
                                          fetch_cn_listing_info)
from ..util import now_utc_iso, read_json, sha256_file, write_json

BALANCE_FIELDS = {
    "book_equity": ["归属于母公司股东权益合计", "归属于母公司所有者权益合计"],
    "assets": ["资产总计"],
    "current_assets": ["流动资产合计", "流动资产"],
    "current_liabilities": ["流动负债合计", "流动负债"],
    "long_debt": ["非流动负债合计", "非流动负债"],
    "share_capital": ["实收资本(或股本)", "实收资本(或股本)净额"],
}
INCOME_FIELDS = {
    "net_income": ["归属于母公司所有者的净利润", "归属于母公司股东的净利润", "净利润"],
    "revenue": ["营业总收入", "营业收入"],
    "cogs": ["营业成本"],
}
CASHFLOW_FIELDS = {
    "operating_cashflow": ["经营活动产生的现金流量净额", "经营活动产生的现金流量"],
}
EVENT_PRIORITY = {"dividend": 0, "split": 1}  # dividend before split on the
# same ex-date: both ratios are declared per pre-event share


# ------------------------------------------------------------------ fetchers
def _fetch_cached(path: Path, fetch, *, refresh: bool = False):
    """Resumable fetch: skip when the cache file exists unless refreshed."""
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_bars_cached(symbols: list[str], *, start: str, end: str | None,
                      cache_dir: Path, refresh: bool = False) -> dict[str, dict]:
    """Raw + qfq daily bars per symbol (Sina), resumable. Values:
    {"raw": DataFrame(date, open, high, low, close, volume, amount,
    outstanding_share, ...), "qfq": DataFrame(date, open, close, ...)}."""
    out = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        raw_path = cache_dir / f"{symbol}_raw.csv"
        qfq_path = cache_dir / f"{symbol}_qfq.csv"
        if not (raw_path.exists() and qfq_path.exists()) or refresh:
            raw, qfq = fetch_cn_daily_pair(symbol, start=start, end=end)
            raw.to_csv(raw_path, index=False)
            qfq.to_csv(qfq_path, index=False)
        out[symbol] = {"raw": pd.read_csv(raw_path, parse_dates=["date"]),
                       "qfq": pd.read_csv(qfq_path, parse_dates=["date"])}
    return out


# ------------------------------------------------------------- table builders
def build_bars_table(raw_by_symbol: dict[str, pd.DataFrame],
                     qfq_by_symbol: dict[str, pd.DataFrame], *,
                     buy_lot: int, sell_lot: int) -> tuple[pd.DataFrame, dict]:
    """Canonical bars table from raw prices + qfq wealth index.

    Rows failing validation (OHLC inconsistency, nonpositive/nonfinite
    prices, invalid dates) are DROPPED and the drop counts disclosed —
    `read_snapshot` would otherwise soft-error the whole table."""
    rows, dropped = [], {}
    for sym, raw in raw_by_symbol.items():
        qfq_src = qfq_by_symbol[sym]
        # defensive: accept both date-column and date-index frames
        qfq = (qfq_src.set_index("date") if "date" in qfq_src.columns
               else qfq_src)
        for _, r in raw.iterrows():
            d = pd.Timestamp(r["date"])
            o, h, l, c = (float(r[k]) for k in ("open", "high", "low", "close"))
            v, amt = float(r["volume"]), float(r["amount"])
            if d not in qfq.index:
                dropped[sym] = dropped.get(sym, 0) + 1
                continue
            tri = float(qfq.at[d, "close"])
            tri_open = float(qfq.at[d, "open"])
            if not all(np.isfinite(x) and x > 0 for x in (o, h, l, c, tri, tri_open, v, amt)):
                dropped[sym] = dropped.get(sym, 0) + 1
                continue
            if h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9:
                dropped[sym] = dropped.get(sym, 0) + 1
                continue
            rows.append({"symbol": sym, "date": d, "open": o, "high": h,
                         "low": l, "close": c, "volume": v, "amount": amt,
                         "tri": tri, "tri_open": tri_open,
                         "buyable": 1, "sellable": 1,
                         "buy_lot": float(buy_lot), "sell_lot": float(sell_lot)})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    return df, {"dropped_rows_per_symbol": dropped,
                "dropped_total": sum(dropped.values())}


def build_universe_table(listing_info: dict[str, dict],
                         first_bar: dict[str, pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for sym in sorted(listing_info):
        info = listing_info[sym]
        listing = info.get("listing_date")
        if listing is None or pd.isna(listing):
            listing = first_bar.get(sym)  # disclosed fallback
        if listing is None:
            continue
        industry = str(info.get("industry") or "UNKNOWN")
        rows.append({"symbol": sym,
                     "effective_from": pd.Timestamp(listing),
                     "effective_to": pd.NaT,
                     "available_at": pd.Timestamp(listing),
                     "listed_at": pd.Timestamp(listing),
                     "eligible": 1,
                     "industry": industry})
    return pd.DataFrame(rows)


def build_actions_table(batch_frames: list[pd.DataFrame],
                        universe_symbols: set[str]) -> pd.DataFrame:
    """Implemented dividend/split events for universe symbols from the fhps
    batch frames. Per-10-share vendor ratios are converted to per-share
    values; dividend is sequenced BEFORE split on the same ex-date (both
    declared per pre-event share)."""
    rows = []
    combined = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame()
    for _, r in combined.iterrows():
        sym6 = str(r.get("symbol") or "")
        symbol = next((s for s in universe_symbols if s.split(".")[0] == sym6), None)
        if symbol is None:
            continue
        ex = pd.Timestamp(r.get("ex_date"))
        if pd.isna(ex):
            continue  # pending plan without an implemented ex-date
        events = []
        for col, kind, priority in (("dividend_ratio_10", "dividend", 0),
                                    ("split_ratio_10", "split", 1)):
            value = pd.to_numeric(r.get(col), errors="coerce")
            if np.isfinite(value) and value > 0:
                events.append((priority, kind, float(value) / 10.0))
        for seq, (priority, kind, value) in enumerate(sorted(events)):
            rows.append({"event_id": f"{symbol}-{ex.date()}-{seq}-{kind}",
                         "symbol": symbol, "date": ex, "sequence": seq,
                         "kind": kind, "value": value, "pay_date": ex})
    df = pd.DataFrame(rows, columns=["event_id", "symbol", "date", "sequence",
                                     "kind", "value", "pay_date"])
    return df.sort_values(["symbol", "date", "sequence"]).reset_index(drop=True)


def build_float_table(raw_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, raw in raw_by_symbol.items():
        out = raw[["date", "close", "outstanding_share"]].copy()
        out = out[pd.to_numeric(out["outstanding_share"], errors="coerce") > 0]
        rows.append(pd.DataFrame({
            "symbol": sym,
            "date": pd.to_datetime(out["date"]),
            "available_at": pd.to_datetime(out["date"]),
            "float_shares": pd.to_numeric(out["outstanding_share"], errors="coerce"),
            "market_cap": (pd.to_numeric(out["close"], errors="coerce")
                           * pd.to_numeric(out["outstanding_share"], errors="coerce")),
        }))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df.dropna().sort_values(["symbol", "date"]).reset_index(drop=True)


def build_financials_table(statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(statements.values(), ignore_index=True) if statements else pd.DataFrame()
    if df.empty:
        return df
    return df.sort_values(["symbol", "period"]).reset_index(drop=True)


def build_costs_table(mcfg: dict, first_session: pd.Timestamp) -> pd.DataFrame:
    c = mcfg["costs"]
    return pd.DataFrame([{
        "date": first_session,
        "commission_bps": float(c["commission_bps"]),
        "min_commission": float(c["min_commission"]),
        "slippage_bps": float(c["slippage_bps"]),
        "stamp_tax_sell_bps": float(c["stamp_tax_sell_bps"]),
    }])


def _normalize_to_session(ts: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """Canonical availability = first usable exchange session on/after ts."""
    ts = pd.Timestamp(ts).normalize()
    idx = sessions.searchsorted(ts)
    return sessions[idx] if idx < len(sessions) else ts


# ---------------------------------------------------------------- orchestrator
def build_cn_snapshot(*, output: str | Path, indices: list[str],
                      start: str = "2018-01-01", end: str | None = None,
                      refresh: bool = False,
                      progress_cb=None) -> dict:
    mcfg = load_market_config("cn")
    out_dir = Path(output)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"snapshot output directory not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "datasets" / "factor_snapshots" / "_fetch_cache" / "cn"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def report(msg):
        if progress_cb:
            progress_cb(msg)
        write_json(out_dir / "_build_progress.json",
                   {"ts": now_utc_iso(), "last": msg})

    # ---- universe --------------------------------------------------------
    report("fetching universe constituents")
    universe_symbols: list[str] = []
    for index in indices:
        universe_symbols.extend(fetch_cn_universe(index))
    universe_symbols = sorted(set(universe_symbols))
    report(f"universe: {len(universe_symbols)} symbols from {indices}")

    # ---- bars --------------------------------------------------------------
    report(f"fetching bars for {len(universe_symbols)} symbols (resumable)")
    raw_by_symbol, qfq_by_symbol = {}, {}
    for i, symbol in enumerate(universe_symbols):
        pair = fetch_bars_cached([symbol], start=start, end=end,
                                 cache_dir=cache_dir / "bars", refresh=refresh)
        raw_by_symbol[symbol] = pair[symbol]["raw"]
        qfq_by_symbol[symbol] = pair[symbol]["qfq"]
        if progress_cb and (i + 1) % 25 == 0:
            report(f"bars fetched: {i + 1}/{len(universe_symbols)}")
    report(f"bars fetched for {len(raw_by_symbol)} symbols")

    # ---- listing info --------------------------------------------------------
    report("fetching listing dates and industries")
    listing_info: dict[str, dict] = {}
    listing_failures: dict[str, str] = {}
    info_cache = cache_dir / "listing.json"
    if info_cache.exists() and not refresh:
        listing_info = {k: {"listing_date": pd.Timestamp(v["listing_date"])
                            if v["listing_date"] else None,
                            "industry": v["industry"]}
                        for k, v in read_json(info_cache).items()}
    else:
        for i, symbol in enumerate(universe_symbols):
            try:
                listing_info[symbol] = fetch_cn_listing_info(symbol)
            except Exception as exc:  # noqa: BLE001 - disclosed
                listing_failures[symbol] = f"{type(exc).__name__}: {str(exc)[:100]}"
            if progress_cb and (i + 1) % 50 == 0:
                report(f"listing info fetched: {i + 1}/{len(universe_symbols)}")
        write_json(info_cache, {k: {"listing_date": str(v["listing_date"]) if v["listing_date"] is not None else None,
                                    "industry": v["industry"]}
                                for k, v in listing_info.items()})
    first_bar = {sym: frames["raw"]["date"].min()
                 for sym, frames in raw_by_symbol.items()}
    report(f"listing info: {len(listing_info)} symbols, "
           f"{len(listing_failures)} failures (fallback to first bar)")

    # ---- actions ---------------------------------------------------------------
    report("fetching corporate-action batches")
    periods = []
    year = int(start[:4])
    while year <= int((end or now_utc_iso()[:10])[:4]) + 1:
        periods.append(f"{year}0630")
        periods.append(f"{year}1231")
        year += 1
    batch_frames, action_failures = [], {}
    for period in periods:
        ppath = cache_dir / "actions" / f"{period}.csv"
        try:
            batch_frames.append(_fetch_cached(
                ppath, lambda p=period: fetch_cn_actions_batch(p), refresh=refresh))
        except Exception as exc:  # noqa: BLE001 - disclosed
            action_failures[period] = f"{type(exc).__name__}: {str(exc)[:100]}"
    report(f"action batches fetched: {len(batch_frames)}/{len(periods)} periods")

    # ---- financials ---------------------------------------------------------
    report(f"fetching annual financial statements for {len(universe_symbols)} symbols (resumable)")
    statements, fin_failures = {}, {}
    for i, symbol in enumerate(universe_symbols):
        try:
            statements[symbol] = fetch_cn_financial_statements(
                symbol, cache_dir=cache_dir / "financials", refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            fin_failures[symbol] = f"{type(exc).__name__}: {str(exc)[:100]}"
        if progress_cb and (i + 1) % 25 == 0:
            report(f"financials fetched: {i + 1}/{len(universe_symbols)}")
    report(f"financials: {len(statements)} symbols, {len(fin_failures)} failures")

    # ---- build tables ----------------------------------------------------------
    sessions_df = None
    bars_df, dropped = build_bars_table(raw_by_symbol, qfq_by_symbol,
                                        buy_lot=int(mcfg["lots"]), sell_lot=1)
    sessions_df = pd.DataFrame({"date": sorted(bars_df["date"].unique())})
    universe_df = build_universe_table(listing_info, first_bar)
    actions_df = build_actions_table(batch_frames, set(universe_symbols))
    float_df = build_float_table(raw_by_symbol)
    fin_df = build_financials_table(statements)

    # normalize availability/effective/listed dates to exchange sessions
    session_idx = pd.DatetimeIndex(sessions_df["date"])
    for col, df_ in (("available_at", universe_df), ("listed_at", universe_df),
                     ("effective_from", universe_df)):
        if len(df_):
            df_[col] = df_[col].map(lambda ts: _normalize_to_session(ts, session_idx))
    if len(fin_df):
        fin_df["available_at"] = fin_df["available_at"].map(
            lambda ts: _normalize_to_session(ts, session_idx))
    if len(actions_df):
        actions_df["date"] = actions_df["date"].map(
            lambda ts: _normalize_to_session(ts, session_idx))
        actions_df["pay_date"] = actions_df["date"]
    costs_df = build_costs_table(mcfg, session_idx[0])

    # ---- manifest -----------------------------------------------------------
    tables = {"sessions": sessions_df, "bars": bars_df, "universe": universe_df,
              "actions": actions_df, "costs": costs_df, "float": float_df,
              "financials": fin_df}
    files = {}
    for name, df in tables.items():
        fpath = out_dir / f"{name}.csv"
        df.to_csv(fpath, index=False)
        # contract: files is keyed by TABLE NAME — read_snapshot looks up
        # manifest["files"][name] to find the CSV and verify its hash
        files[name] = {"path": f"{name}.csv", "sha256": sha256_file(fpath)}
    limitations = [
        "SURVIVORSHIP BIAS: universe is current index membership; historical "
        "constituents are unavailable token-free — results are exploratory "
        "with a research_only ceiling.",
        "Financial statement values are the CURRENT VINTAGE (latest "
        "restatement) with original announcement dates; as-originally-reported "
        "vintages are unavailable from the token-free feed.",
        "equity_issued is approximated as the increase in share capital at "
        "par (~1 CNY/share) between consecutive annual periods.",
        "tri/tri_open use the qfq wealth index anchored to the latest bar "
        "(dividend reinvestment at ex-date close assumed).",
        "buyable/sellable are 1 for every observed bar; suspensions appear as "
        "missing bars and are handled by the stale-mark/order-expiry machinery.",
        "No delisting events: delisting proceeds are unknown in this universe.",
        f"Bars rows dropped by pre-validation: {dropped['dropped_total']} "
        f"({len(dropped['dropped_rows_per_symbol'])} symbols affected).",
        f"Fetch failures disclosed: bars ok={len(raw_by_symbol)}/{len(universe_symbols)}, "
        f"listing failures={len(listing_failures)}, financial failures={len(fin_failures)}, "
        f"action-batch failures={len(action_failures)}.",
    ]
    manifest = {
        "market": "cn",
        "schema_version": 1,
        "price_basis": "raw",
        "quantity_unit": "shares",
        "amount_unit": "local_currency",
        "generated_utc": now_utc_iso(),
        "start": start, "end": end or now_utc_iso()[:10],
        "indices": list(indices),
        "files": files,
        "limitations": limitations,
        "evidence": {
            "survivorship_bias": True,
            "financial_vintages": "current_only_with_announcement_dates",
            "tri_convention": "qfq wealth index anchored to latest bar",
            "providers": ["sina daily (raw+qfq)", "sina financial reports",
                          "eastmoney fhps batch", "csindex constituents",
                          "eastmoney individual info"],
        },
        "failures": {"listing": listing_failures, "financials": fin_failures,
                     "actions": action_failures},
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "_build_progress.json",
               {"ts": now_utc_iso(), "done": True,
                "tables": {k: int(len(v)) for k, v in tables.items()}})
    return {"output": out_dir, "manifest": manifest,
            "table_rows": {k: int(len(v)) for k, v in tables.items()},
            "dropped_bars": dropped}

