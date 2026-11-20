"""Token-free fundamental-data providers for the CN canonical snapshot
(comprehensive plan 2026-11-17 W6; factor-bundle snapshot contract).

Sources (verified live 2026-11-17/18, akshare 1.18.54):
- Financial statements: `stock_financial_report_sina` (Sina openapi) —
  balance sheet / income / cash flow, one row per report period, with
  公告日期 (announcement date, YYYYMMDD string) and 是否审计 (audit flag).
  Values are the CURRENT VINTAGE (latest restatement): original as-reported
  vintages do not exist in this feed — disclosed in the manifest.
- Corporate actions: `stock_fhps_detail_em` (Eastmoney datacenter, direct
  connection works with the NO_PROXY bypass) — per-10-share 送转/派息 ratios,
  预案公告日, 股权登记日, 除权除息日, 方案进度.
- Listing date / industry: `stock_individual_info_em` (上市时间, 行业).

All fetchers are resumable: `fetch_*_cached` writes each per-symbol result
under a cache directory and skips existing files, so a multi-hour snapshot
build can be interrupted and resumed. Failures are disclosed per symbol and
never overwrite cached results.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .providers import ProviderError, _akshare, _retry

# Sina statement item names → canonical fields (verified against 600519 on
# 2026-11-17; the vendor renames items occasionally, so each field has an
# ordered candidate list and an unmatched field is disclosed, not guessed).
BALANCE_SHEET_MAP = {
    "book_equity": ["归属于母公司股东权益合计", "归属于母公司所有者权益合计"],
    "assets": ["资产总计"],
    "current_assets": ["流动资产合计", "流动资产"],
    "current_liabilities": ["流动负债合计", "流动负债"],
    "long_debt": ["非流动负债合计", "非流动负债"],
    "share_capital": ["实收资本(或股本)", "实收资本(或股本)净额"],
}
INCOME_MAP = {
    "net_income": ["归属于母公司所有者的净利润", "归属于母公司股东的净利润", "净利润"],
    "revenue": ["营业总收入", "营业收入"],
    "cogs": ["营业成本"],
}
CASHFLOW_MAP = {
    "operating_cashflow": ["经营活动产生的现金流量净额", "经营活动产生的现金流量"],
}

STATEMENTS = ("资产负债表", "利润表", "现金流量表")


def _statement(symbol: str, statement: str) -> pd.DataFrame:
    ak = _akshare()
    code, _, suffix = symbol.partition(".")
    stock = f"{suffix.lower()}{code}"
    df = _retry(lambda: ak.stock_financial_report_sina(stock=stock, symbol=statement),
                what=f"sina {statement} {symbol}")
    if df is None or len(df) == 0:
        raise ProviderError(f"empty {statement} for {symbol}")
    df = df.copy()
    df["公告日期"] = pd.to_datetime(df["公告日期"].astype(str), format="%Y%m%d",
                                    errors="coerce")
    df["报告日"] = pd.to_datetime(df["报告日"].astype(str), format="%Y%m%d",
                                  errors="coerce")
    return df.dropna(subset=["报告日"])


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """First present candidate column as a numeric series; None if absent.
    Unmatched fields are disclosed by the caller (contract: missing required
    fields remain blocked, never guessed)."""
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def fetch_cn_financial_statements(symbol: str, *, cache_dir: str | Path | None = None,
                                  force: bool = False) -> pd.DataFrame:
    """Annual financial rows for one symbol: one row per fiscal year with the
    canonical fields, available_at = announcement date, revision = announcement
    date string. Quarterly rows are excluded (F-Score consumes annual only).

    Resumable: the joined result is cached per symbol in `cache_dir`.
    """
    cache = Path(cache_dir) if cache_dir else None
    if cache and (cache / f"{symbol}.csv").exists() and not force:
        return pd.read_csv(cache / f"{symbol}.csv", parse_dates=["period", "available_at"])

    frames = {stmt: _statement(symbol, stmt) for stmt in STATEMENTS}
    bs, isl, cf = frames["资产负债表"], frames["利润表"], frames["现金流量表"]

    # annual rows only (F-Score consumes annual statements 300-430 days apart)
    bs = bs[bs["报告日"].dt.strftime("%m%d") == "1231"]

    merged = bs[["报告日", "公告日期"]].rename(
        columns={"报告日": "period", "公告日期": "ann_bs"}).merge(
        isl[["报告日", "公告日期"]].rename(columns={"报告日": "period", "公告日期": "ann_is"}),
        on="period", how="inner").merge(
        cf[["报告日", "公告日期"]].rename(columns={"报告日": "period", "公告日期": "ann_cf"}),
        on="period", how="inner")
    if merged.empty:
        raise ProviderError(f"no annual statements with all three reports for {symbol}")

    # available_at = the LATEST announcement among the three statements
    # (conservative: the full statement set is only usable once all are out)
    merged["available_at"] = merged[["ann_bs", "ann_is", "ann_cf"]].max(axis=1)

    out = pd.DataFrame({
        "symbol": symbol,
        "period": merged["period"],
        "available_at": merged["available_at"],
        "revision": merged["available_at"].dt.strftime("%Y%m%d"),
    })
    for field, (df, cands) in {
        "book_equity": (bs, BALANCE_SHEET_MAP["book_equity"]),
        "assets": (bs, BALANCE_SHEET_MAP["assets"]),
        "current_assets": (bs, BALANCE_SHEET_MAP["current_assets"]),
        "current_liabilities": (bs, BALANCE_SHEET_MAP["current_liabilities"]),
        "long_debt": (bs, BALANCE_SHEET_MAP["long_debt"]),
        "share_capital": (bs, BALANCE_SHEET_MAP["share_capital"]),
        "net_income": (isl, INCOME_MAP["net_income"]),
        "revenue": (isl, INCOME_MAP["revenue"]),
        "cogs": (isl, INCOME_MAP["cogs"]),
        "operating_cashflow": (cf, CASHFLOW_MAP["operating_cashflow"]),
    }.items():
        series = _pick(df, cands)
        if series is None:
            raise ProviderError(
                f"field '{field}' not found in {symbol} statements "
                f"(candidates {cands} all absent) — disclosed, never guessed")
        by_period = pd.Series(series.values, index=df["报告日"])
        out[field] = out["period"].map(by_period)

    out = out.drop(columns=[], errors="ignore")
    # equity_issued: increase in share capital (par ≈ 1 CNY/share) between
    # consecutive annual periods — monetary issuance approximation, documented
    out = out.sort_values("period").reset_index(drop=True)
    out["equity_issued"] = (out["share_capital"].diff()
                            .clip(lower=0.0).fillna(0.0))

    out = out.drop(columns=["share_capital"])
    num = out.drop(columns=["symbol", "period", "available_at", "revision"])
    bad = ~num.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    out = out[~bad].reset_index(drop=True)
    if out.empty:
        raise ProviderError(f"no complete annual rows for {symbol}")

    if cache:
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache / f"{symbol}.csv", index=False)
    return out


def fetch_cn_actions_batch(report_date: str) -> pd.DataFrame:
    """All implemented-eligible fhps rows for ONE report period across the
    whole market (Eastmoney datacenter, single call) — far more efficient
    than per-symbol calls. Returns the normalized vendor frame with
    per-10-share ratio values RAW (the snapshot builder converts to per-share).

    Columns: symbol, split_ratio_10, dividend_ratio_10, ex_date, plan_notice.
    Rows without an implemented ex-date (pending plans) are excluded here.
    """
    ak = _akshare()
    df = _retry(lambda: ak.stock_fhps_em(date=str(report_date)),
                what=f"fhps batch {report_date}")
    if df is None or len(df) == 0:
        raise ProviderError(f"empty fhps batch for {report_date}")
    out = pd.DataFrame({
        "symbol": pd.to_numeric(df["代码"], errors="coerce").map(
            lambda v: f"{int(v):06d}" if np.isfinite(v) else None),
        "split_ratio_10": pd.to_numeric(df.get("送转股份-送转总比例"), errors="coerce"),
        "dividend_ratio_10": pd.to_numeric(df.get("现金分红-现金分红比例"), errors="coerce"),
        "ex_date": pd.to_datetime(df.get("除权除息日"), errors="coerce"),
        "plan_notice": pd.to_datetime(df.get("预案公告日"), errors="coerce"),
    })
    out = out[out["symbol"].notna() & out["ex_date"].notna()]
    progress = df.get("方案进度")
    if progress is not None:
        implemented = progress.astype(str).str.contains("实施", na=False)
        out = out[implemented.reindex(out.index, fill_value=False)]
    return out.reset_index(drop=True)




def fetch_cn_listing_batch() -> dict[str, dict]:
    """Listing dates for ALL A-share symbols in two batch calls (SZSE + SSE
    exchange official lists). Returns {code6: {"listing_date": Timestamp,
    "exchange": "SZ"|"SH", "industry": str|None}}.

    SZSE includes 所属行业; SSE does not — SSE industries fall back to
    UNKNOWN (disclosed in the manifest)."""
    ak = _akshare()
    out: dict[str, dict] = {}
    try:
        sz = _retry(lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
                    what="szse listing")
        for _, r in sz.iterrows():
            code = str(r.get("A股代码") or r.get("证券代码")).zfill(6)
            date = pd.to_datetime(str(r.get("A股上市日期") or r.get("上市日期")),
                                  errors="coerce")
            if pd.isna(date):
                continue
            out[code] = {"listing_date": date, "exchange": "SZ",
                         "industry": str(r.get("所属行业") or "UNKNOWN")}
    except Exception as exc:  # noqa: BLE001 - disclosed
        out["_szse_error"] = {"listing_date": None, "exchange": "SZ",
                              "industry": f"{type(exc).__name__}: {str(exc)[:100]}"}
    try:
        sh = _retry(lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
                    what="sse listing")
        for _, r in sh.iterrows():
            code = str(r.get("证券代码")).zfill(6)
            date = pd.to_datetime(str(r.get("上市日期")), errors="coerce")
            if pd.isna(date):
                continue
            out[code] = {"listing_date": date, "exchange": "SH",
                         "industry": None}
    except Exception as exc:  # noqa: BLE001 - disclosed
        out["_sse_error"] = {"listing_date": None, "exchange": "SH",
                             "industry": f"{type(exc).__name__}: {str(exc)[:100]}"}
    return out

def fetch_cn_listing_info(symbol: str) -> dict:
    """Listing date (上市时间) and industry (行业) for one symbol — current
    snapshot from Eastmoney (direct connection, NO_PROXY bypass)."""
    ak = _akshare()
    code = symbol.split(".")[0]
    df = _retry(lambda: ak.stock_individual_info_em(symbol=code),
                what=f"info {symbol}")
    info = dict(zip(df["item"].astype(str), df["value"]))
    listing = info.get("上市时间")
    listing_ts = pd.to_datetime(str(listing), format="%Y%m%d", errors="coerce") \
        if listing else pd.NaT
    return {"listing_date": listing_ts, "industry": str(info.get("行业", "UNKNOWN"))}


def fetch_cn_financials_cached(symbols: list[str], *, cache_dir: str | Path,
                               force: bool = False) -> dict[str, pd.DataFrame]:
    """Resumable bulk financial fetch. Returns {symbol: frame} for symbols
    that succeeded; failures are disclosed via the returned failures dict."""
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for symbol in symbols:
        try:
            results[symbol] = fetch_cn_financial_statements(
                symbol, cache_dir=cache_dir, force=force)
        except Exception as exc:  # noqa: BLE001 - disclosed, skipped
            failures[symbol] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return results, failures
