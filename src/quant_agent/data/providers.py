"""Market-data providers for updating the bundled historical snapshots.

Sources found available in this environment (verified live 2026-11-14):
- CN: akshare `stock_zh_a_daily` (Sina) — free, no token; raw volume already
  in shares; qfq variant gives back-adjusted prices anchored to the latest
  session, from which a real adjustment-factor series is derived. This CLOSES
  the bundled data's "CN adjustment unknown" manifest gap.
- US: akshare `stock_us_daily` (Yahoo-backed) — free; raw + qfq variants.

Network notes (measured): a Windows SYSTEM proxy is configured and flaky for
EastMoney endpoints; Sina/Yahoo respond reliably. The provider therefore
bypasses proxies for these hosts via NO_PROXY and retries with backoff.

Provenance rules:
- Updates are written to datasets/updates/{market}/{symbol}.csv in the simple
  schema PLUS adj_factor and amount; ingest prefers an update file over the
  legacy snapshot and records source_kind/provider/fetched_at in the manifest.
- A failed fetch never overwrites anything and is reported explicitly.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

PROVIDER_NAME = "akshare"
PROVIDER_CN_SOURCE = "sina (akshare stock_zh_a_daily)"
PROVIDER_US_SOURCE = "yahoo via akshare stock_us_daily"
PROVIDER_VALUATION_SOURCE = "baidu gushitong via akshare stock_zh_valuation_baidu"
PROVIDER_UNIVERSE_SOURCE = "csi index constituents via akshare index_stock_cons_csindex"

# bypass flaky Windows system proxy for direct-tested hosts
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")


class ProviderError(RuntimeError):
    pass


def _retry(fn, *, tries: int = 4, wait_s: float = 2.0, what: str = "fetch"):
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider boundary
            last = exc
            time.sleep(wait_s * (i + 1))
    raise ProviderError(f"{what} failed after {tries} attempts: {type(last).__name__}: {last}")


def _akshare():
    warnings = []
    try:
        import warnings as _w
        _w.filterwarnings("ignore")
        import akshare as ak
        return ak
    except ImportError as exc:  # pragma: no cover
        raise ProviderError("akshare is not installed in this environment") from exc


def cn_symbol(symbol: str) -> str:
    """'000001.SZ' -> 'sz000001'; '600036.SH' -> 'sh600036'."""
    code, _, suffix = symbol.partition(".")
    suffix = suffix.upper()
    if suffix not in ("SZ", "SH"):
        raise ProviderError(f"unsupported CN suffix in '{symbol}'")
    return f"{suffix.lower()}{code}"


@dataclass
class ProviderResult:
    frame: pd.DataFrame          # canonical: date,symbol,OHLC,volume,amount,adj_factor
    rows: int
    first_date: str
    last_date: str
    factor_dates_changed: int
    notes: list[str]


def _merge_raw_qfq(raw: pd.DataFrame, qfq: pd.DataFrame, symbol: str,
                   start: str, end: str) -> ProviderResult:
    raw = raw.copy()
    qfq = qfq.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    qfq["date"] = pd.to_datetime(qfq["date"])
    raw = raw[(raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))]
    qfq = qfq[(qfq["date"] >= pd.Timestamp(start)) & (qfq["date"] <= pd.Timestamp(end))]
    notes = []
    m = raw.merge(qfq[["date", "close"]], on="date", how="inner",
                  suffixes=("", "_qfq"))
    if len(m) < len(raw):
        notes.append(f"qfq/raw date misalignment: dropped {len(raw) - len(m)} raw rows")
    if m.empty:
        raise ProviderError(f"no overlapping rows for {symbol} in [{start},{end}]")
    factor = m["close_qfq"] / m["close"]
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ProviderError(f"invalid adjustment factor derived for {symbol}")
    changed = int(((factor - 1.0).abs() > 1e-4).sum())
    out = pd.DataFrame({
        "date": m["date"],
        "symbol": symbol,
        "open": m["open"], "high": m["high"], "low": m["low"], "close": m["close"],
        "volume": m["volume"],
        "amount": m.get("amount", np.nan),
        "adj_factor": factor,
    }).sort_values("date").reset_index(drop=True)
    return ProviderResult(
        frame=out, rows=int(len(out)),
        first_date=str(out["date"].iloc[0].date()),
        last_date=str(out["date"].iloc[-1].date()),
        factor_dates_changed=changed, notes=notes,
    )


def fetch_cn_daily(symbol: str, *, start: str, end: str) -> ProviderResult:
    """CN daily bars from Sina with real adjustment factors (qfq/raw)."""
    ak = _akshare()
    sina_sym = cn_symbol(symbol)

    def _call(adjust):
        return ak.stock_zh_a_daily(symbol=sina_sym, start_date=start.replace("-", ""),
                                   end_date=end.replace("-", ""), adjust=adjust)

    raw = _retry(lambda: _call(""), what=f"CN raw {symbol}")
    qfq = _retry(lambda: _call("qfq"), what=f"CN qfq {symbol}")
    return _merge_raw_qfq(raw, qfq, symbol, start, end)


def fetch_us_daily(symbol: str, *, start: str, end: str) -> ProviderResult:
    """US daily bars from the Yahoo-backed akshare interface with factors."""
    ak = _akshare()
    raw = _retry(lambda: ak.stock_us_daily(symbol=symbol, adjust=""),
                 what=f"US raw {symbol}")
    qfq = _retry(lambda: ak.stock_us_daily(symbol=symbol, adjust="qfq"),
                 what=f"US qfq {symbol}")
    return _merge_raw_qfq(raw, qfq, symbol, start, end)


def fetch_daily(market: str, symbol: str, *, start: str, end: str) -> ProviderResult:
    if market == "cn":
        return fetch_cn_daily(symbol, start=start, end=end)
    if market == "us":
        return fetch_us_daily(symbol, start=start, end=end)
    raise ProviderError(f"no provider configured for market '{market}'")


def update_market(market: str, *, symbols: list[str] | None = None,
                  start: str = "2019-01-02",
                  end: str | None = None,
                  out_dir: str | None = None) -> dict:
    """Fetch fresh bars for the universe and write update CSVs.

    Writes datasets/updates/{market}/{symbol}.csv (simple schema + amount +
    adj_factor). Failures are reported per symbol and overwrite nothing.
    """
    from ..config import PROJECT_ROOT, load_market_config
    from ..util import now_utc_iso

    cfg = load_market_config(market)
    symbols = symbols or list(cfg["universe"])
    end = end or now_utc_iso()[:10]
    dest = out_dir or str(PROJECT_ROOT / "datasets" / "updates" / market)
    os.makedirs(dest, exist_ok=True)

    summary: dict = {"market": market, "provider": PROVIDER_NAME,
                     "cn_source": PROVIDER_CN_SOURCE, "us_source": PROVIDER_US_SOURCE,
                     "start": start, "end": end, "fetched_at_utc": now_utc_iso(),
                     "symbols": {}}
    for symbol in symbols:
        entry: dict = {"symbol": symbol}
        try:
            res = fetch_daily(market, symbol, start=start, end=end)
            path = os.path.join(dest, f"{symbol}.csv")
            res.frame.to_csv(path, index=False)
            entry.update({
                "written": path, "rows": res.rows,
                "first_date": res.first_date, "last_date": res.last_date,
                "factor_dates_changed": res.factor_dates_changed,
                "notes": res.notes,
            })
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            entry["error"] = f"{type(exc).__name__}: {exc}"
        summary["symbols"][symbol] = entry
    from ..util import write_json
    write_json(os.path.join(dest, "_update_summary.json"), summary)
    return summary


def update_summary(market: str, root=None) -> dict | None:
    """Last update summary for provenance (None if never updated)."""
    from ..config import PROJECT_ROOT
    from ..util import read_json
    base = root if root is not None else PROJECT_ROOT
    path = base / "datasets" / "updates" / market / "_update_summary.json"
    if not path.exists():
        return None
    return read_json(path)


def fetch_cn_valuation(symbol: str, indicator: str, *, period: str = "近五年") -> pd.DataFrame:
    """Per-symbol valuation history from Baidu Gushitong (date, value).

    Vendor API contract: the literals are Chinese wire-format values, not
    documentation, and must stay in Chinese for the API to work.
    `indicator` accepts "市盈率(TTM)" (PE_TTM), "市净率" (PB), "总市值"
    (total market cap, unit: 100M CNY); `period` accepts "近五年" (last five
    years) or "近一年" (last year). Sampling: 近五年 ≈ every 2 calendar days;
    近一年 ≈ daily.
    """
    ak = _akshare()
    code = symbol.split(".")[0]
    df = _retry(lambda: ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator,
                                                    period=period),
                what=f"valuation {symbol}/{indicator}")
    if df is None or len(df) == 0:
        raise ProviderError(f"empty valuation series for {symbol}/{indicator}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "indicator": indicator,
        "value": pd.to_numeric(df["value"], errors="coerce"),
    })
    if out["value"].isna().all():
        raise ProviderError(f"all-nonnumeric valuation values for {symbol}/{indicator}")
    return out.dropna()


def fetch_cn_universe(index: str = "000300") -> list[str]:
    """Current index constituents as ts_code-style symbols ('000001.SZ')."""
    ak = _akshare()
    df = _retry(lambda: ak.index_stock_cons_csindex(symbol=index),
                what=f"constituents {index}")
    suffix_map = {"深圳证券交易所": "SZ", "上海证券交易所": "SH"}
    symbols = []
    for _, row in df.iterrows():
        code = str(row["成分券代码"]).zfill(6)
        suffix = suffix_map.get(str(row["交易所"]))
        if suffix is None:
            continue
        symbols.append(f"{code}.{suffix}")
    if not symbols:
        raise ProviderError(f"no constituents parsed for index {index}")
    return sorted(set(symbols))


def update_xs_universe(index: str = "000300", *, n_symbols: int = 150, seed: int = 42,
                       start: str = "2020-06-01", end: str | None = None,
                       out_dir: str | None = None) -> dict:
    """Fetch bars + valuation for a seeded sample of index constituents.

    Writes datasets/updates/xs_cn/{bars,valuation}/... plus a summary. This
    snapshot universe carries SURVIVORSHIP BIAS (current membership) — the
    study built on it is exploratory screening only (DECISIONS.md 2.5).
    """
    from ..util import now_utc_iso

    symbols = fetch_cn_universe(index)
    rng = np.random.default_rng(seed)
    if len(symbols) > n_symbols:
        symbols = sorted(rng.choice(symbols, size=n_symbols, replace=False).tolist())
    end = end or now_utc_iso()[:10]
    base = out_dir or str(PROJECT_ROOT / "datasets" / "updates" / "xs_cn")
    bars_dir, val_dir = os.path.join(base, "bars"), os.path.join(base, "valuation")
    os.makedirs(bars_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    summary: dict = {"index": index, "requested": n_symbols, "seed": seed,
                     "start": start, "end": end, "fetched_at_utc": now_utc_iso(),
                     "provider": PROVIDER_NAME,
                     "valuation_source": PROVIDER_VALUATION_SOURCE,
                     "universe_source": PROVIDER_UNIVERSE_SOURCE,
                     "survivorship_bias": True, "symbols": {}}
    for symbol in symbols:
        entry: dict = {"symbol": symbol}
        try:
            res = fetch_cn_daily(symbol, start=start, end=end)
            res.frame.to_csv(os.path.join(bars_dir, f"{symbol}.csv"), index=False)
            entry["bars_rows"] = res.rows
            entry["bars_last"] = res.last_date
        except Exception as exc:  # noqa: BLE001
            entry["bars_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        val = {}
        for indicator in ("市盈率(TTM)", "市净率", "总市值"):
            try:
                v = fetch_cn_valuation(symbol, indicator)
                v.to_csv(os.path.join(val_dir, f"{symbol}_{indicator}.csv"), index=False)
                val[indicator] = int(len(v))
            except Exception as exc:  # noqa: BLE001
                val[indicator] = f"ERROR {type(exc).__name__}: {str(exc)[:80]}"
        entry["valuation_rows"] = val
        summary["symbols"][symbol] = entry
    from ..util import write_json
    write_json(os.path.join(base, "_xs_summary.json"), summary)
    return summary


# --------------------------------------------------------------------------
# News providers (H5 critical-information-point monitor).
# Verified live 2026-11-15: both endpoints work in this environment.
# --------------------------------------------------------------------------
PROVIDER_NEWS_SOURCE_STOCK = "eastmoney via akshare stock_news_em"
PROVIDER_NEWS_SOURCE_MACRO = "CCTV xinwenlianbo archive via akshare news_cctv"


def fetch_stock_news(symbol: str, *, max_items: int = 10) -> pd.DataFrame:
    """Recent per-stock news (eastmoney). Columns:
    timestamp (datetime), headline, content, source, url.

    NOTE: the feed returns only the MOST RECENT items (roughly the last few
    days); historical per-stock news is not available from this source.
    """
    ak = _akshare()
    code = symbol.split(".")[0]
    df = _retry(lambda: ak.stock_news_em(symbol=code), what=f"news {symbol}")
    if df is None or len(df) == 0:
        raise ProviderError(f"empty news feed for {symbol}")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["发布时间"], errors="coerce"),
        "headline": df["新闻标题"].astype(str),
        "content": df["新闻内容"].astype(str),
        "source": df["文章来源"].astype(str),
        "url": df["新闻链接"].astype(str),
    }).dropna(subset=["timestamp"]).head(max_items)
    if out.empty:
        raise ProviderError(f"no parseable news timestamps for {symbol}")
    return out


def fetch_cctv_news(date: str) -> pd.DataFrame:
    """CCTV xinwenlianbo items for one calendar date (historical archive).
    Columns: timestamp (datetime, date-only), headline, content."""
    ak = _akshare()
    yyyymmdd = pd.Timestamp(date).strftime("%Y%m%d")
    df = _retry(lambda: ak.news_cctv(date=yyyymmdd), what=f"cctv {date}")
    if df is None or len(df) == 0:
        raise ProviderError(f"empty CCTV news for {date}")
    return pd.DataFrame({
        "timestamp": pd.to_datetime(df["date"].astype(str), errors="coerce"),
        "headline": df["title"].astype(str),
        "content": df["content"].astype(str),
    }).dropna(subset=["timestamp"])
