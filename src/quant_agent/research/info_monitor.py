"""Critical information point monitor (H5, pre-registered DECISIONS 2.8).

Deterministic implementation of the system-prompt contract designed in this
session: flag a "critical information point" only when a deal-amount
statistical anomaly AND a correlated news item BOTH hold. Quantitative
anomalies without news correlation stay on the watchlist and appear only in
reasoning_steps. Daily resolution (no intraday tape available) — disclosed
in every report.

JSON output contract (exact): {"reasoning_steps": [...],
"critical_information_points": [{timestamp, deal_amount, rate_of_change,
relevant_news, rationale}]}. Reasoning precedes conclusions; every rationale
cites its reasoning step numbers.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_market_config
from ..data.calendar import market_sessions
from ..data.providers import (PROVIDER_NAME, PROVIDER_NEWS_SOURCE_MACRO,
                              PROVIDER_NEWS_SOURCE_STOCK,
                              fetch_cctv_news, fetch_stock_news)
from ..util import config_fingerprint, now_utc_iso, write_json

BASELINE_WINDOW = 30
MIN_BASELINE = 10
Z_THRESHOLD = 3.0
RATE_THRESHOLD = 2.0
CCTV_KEYWORD_CLASSES = {
    # Chinese keyword lists match Chinese-language news text by contract.
    "monetary": ["降准", "降息", "利率", "LPR", "央行", "货币", "MLF", "流动性"],
    "fiscal": ["财政", "税", "国债", "专项债", "地方债", "减税"],
    "capital_market": ["证监会", "注册制", "IPO", "上市", "退市", "资本市场",
                        "股市", "交易", "投资者"],
    "macro": ["GDP", "PMI", "CPI", "进出口", "消费", "经济", "增长", "就业"],
    "industry_policy": ["行业", "规划", "改革", "政策", "发展", "振兴", "支持"],
}
MACRO_TOP_DATES = 8
STOCK_TOP_SYMBOLS = 10


# ------------------------------------------------------------------ detection
def detect_amount_anomalies(amount: pd.Series, *, baseline: int = BASELINE_WINDOW,
                            z_threshold: float = Z_THRESHOLD,
                            rate_threshold: float = RATE_THRESHOLD,
                            min_baseline: int = MIN_BASELINE) -> pd.DataFrame:
    """Robust anomaly detection on a date-indexed deal-amount series.

    Primary rule: robust z >= z_threshold vs trailing median/MAD baseline.
    Secondary rule: |day-over-day rate| >= rate_threshold. Rows with fewer
    than min_baseline baseline observations are never flagged (insufficient
    history) — disclosed rather than fabricated.
    """
    amount = amount.dropna().astype(float)
    if len(amount) < min_baseline + 1:
        return pd.DataFrame(columns=["date", "amount", "z", "rate", "rule"])
    roll_med = amount.rolling(baseline).median()
    roll_mad = amount.rolling(baseline).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    z = (amount - roll_med) / (1.4826 * roll_mad.replace(0.0, np.nan))
    rate = amount.pct_change()
    prior_obs = pd.Series(np.arange(len(amount)), index=amount.index)

    rows = []
    for date in amount.index:
        i = prior_obs.loc[date]
        if i < min_baseline:
            continue  # insufficient baseline: disclosed, never flagged
        z_t, r_t = z.loc[date], rate.loc[date]
        rules = []
        if np.isfinite(z_t) and z_t >= z_threshold:
            rules.append("robust_z")
        if np.isfinite(r_t) and abs(r_t) >= rate_threshold:
            rules.append("rate")
        if rules:
            rows.append({"date": date, "amount": float(amount.loc[date]),
                         "z": float(z_t) if np.isfinite(z_t) else float("nan"),
                         "rate": float(r_t) if np.isfinite(r_t) else float("nan"),
                         "rule": "+".join(rules)})
    return pd.DataFrame(rows)


def collapse_clusters(anomalies: pd.DataFrame,
                      session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Collapse consecutive-session anomaly runs into one event (first date),
    so a multi-day burst is not re-flagged per session."""
    if anomalies.empty:
        return anomalies
    pos = session_dates.get_indexer(anomalies["date"])
    keep = [0]
    for k in range(1, len(anomalies)):
        if pos[k] != pos[k - 1] + 1:
            keep.append(k)
    return anomalies.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------- correlation
def _headline_materiality(headline: str, content: str = "") -> list[str]:
    text = f"{headline} {content}"
    return [cls for cls, kws in CCTV_KEYWORD_CLASSES.items()
            if any(kw in text for kw in kws)]


def correlate_stock_news(anomaly_date: pd.Timestamp, news: pd.DataFrame,
                         next_session: pd.Timestamp | None) -> pd.DataFrame:
    """Stock-level correlation at daily resolution: a news item correlates if
    published on the anomaly date or on the next session (+1, per
    pre-registration; intraday alignment unavailable)."""
    if news is None or len(news) == 0:
        return pd.DataFrame()
    dates = pd.Series(news["timestamp"].dt.date)
    target = {anomaly_date.date()}
    if next_session is not None:
        target.add(pd.Timestamp(next_session).date())
    return news[dates.isin(target)]


def correlate_cctv_news(anomaly_date: pd.Timestamp,
                        cctv: pd.DataFrame | None) -> tuple[list[dict], list[str]]:
    """Market-level correlation: CCTV items published on the anomaly date that
    hit at least one macro keyword class. Returns (matched_items, classes)."""
    if cctv is None or len(cctv) == 0:
        return [], []
    items = cctv[cctv["timestamp"].dt.date == anomaly_date.date()]
    matched, classes = [], set()
    for _, row in items.iterrows():
        hit = _headline_materiality(str(row["headline"]), str(row.get("content", "")))
        if hit:
            classes.update(hit)
            matched.append({"headline": str(row["headline"]),
                            "classes": hit,
                            "source": "CCTV xinwenlianbo"})
    return matched, sorted(classes)


# ------------------------------------------------------------------ emission
def emit_report(reasoning_steps: list[str],
                points: list[dict]) -> dict:
    """Emit the exact JSON contract: reasoning first, conclusions after."""
    for p in points:
        assert set(p) == {"timestamp", "deal_amount", "rate_of_change",
                          "relevant_news", "rationale"}, f"schema violation: {set(p)}"
    return {
        "reasoning_steps": reasoning_steps,
        "critical_information_points": points,
    }


# --------------------------------------------------------------------- runner
def run_info_monitor_study(*, stock_top: int = STOCK_TOP_SYMBOLS,
                           macro_top: int = MACRO_TOP_DATES,
                           recent_sessions: int = 30) -> Path:
    t0 = time.time()
    mcfg = load_market_config("cn")
    bars_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "bars"
    frames = {"bars": _load_bars(bars_dir)}
    bars = frames["bars"]
    if len(bars) < 30:
        raise RuntimeError("insufficient universe data for the monitor study")
    sessions = market_sessions(bars)
    fp = config_fingerprint({"n": len(bars)})
    run_id = (f"xs_info_h5-{fp}"
              f"-{now_utc_iso().replace(':', '')}")
    study_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    study_dir.mkdir(parents=True, exist_ok=True)

    reasoning: list[str] = []
    points_stock: list[dict] = []
    points_market: list[dict] = []
    calibration = {"symbols_scanned": len(bars), "anomaly_sessions": 0,
                   "stock_news_symbols_fetched": 0, "cctv_dates_fetched": 0,
                   "stock_dual_flags": 0, "market_dual_flags": 0,
                   "watchlisted_quant_only": 0}

    # ---- step 1: input validation -------------------------------------------
    total_anomaly_rows = 0
    per_symbol_anomalies: dict[str, pd.DataFrame] = {}
    for sym, df in bars.items():
        amt = df.set_index("date")["amount"].sort_index()
        anom = detect_amount_anomalies(amt)
        anom = collapse_clusters(anom, sessions)
        per_symbol_anomalies[sym] = anom
        total_anomaly_rows += len(anom)
    calibration["anomaly_sessions"] = total_anomaly_rows
    reasoning.append(
        f"1. Input validation: {len(bars)} symbols loaded from cached provider "
        f"snapshots; full-history robust anomaly scan (baseline={BASELINE_WINDOW}, "
        f"Z>={Z_THRESHOLD}, |rate|>={RATE_THRESHOLD}) produced "
        f"{total_anomaly_rows} collapsed anomaly clusters; daily resolution "
        f"disclosed (no intraday tape available).")

    # ---- step 2: market-level anomalies --------------------------------------
    agg = pd.DataFrame({sym: df.set_index("date")["amount"].sort_index()
                        for sym, df in bars.items()}).sum(axis=1).dropna()
    mkt = collapse_clusters(detect_amount_anomalies(agg), sessions)
    mkt["abs_z"] = mkt["z"].abs()
    top_dates = mkt.sort_values("abs_z", ascending=False).head(macro_top)
    reasoning.append(
        f"2. Market-level aggregate deal amounts produce {len(mkt)} anomaly "
        f"clusters across {len(sessions)} sessions; top {len(top_dates)} by |z| "
        f"selected for CCTV macro-news correlation (archive covers all history).")

    # ---- step 3: CCTV correlation (market level) ------------------------------
    for _, row in top_dates.iterrows():
        d = pd.Timestamp(row["date"])
        try:
            cctv = fetch_cctv_news(d)
            calibration["cctv_dates_fetched"] += 1
        except Exception as exc:  # noqa: BLE001 - disclosed, not fatal
            reasoning.append(
                f"3x. CCTV fetch failed for {d.date()}: {type(exc).__name__} — "
                "market-level correlation unavailable for this date (watchlisted).")
            continue
        matched, classes = correlate_cctv_news(d, cctv)
        if matched:
            heads = "; ".join(m["headline"][:40] for m in matched[:2])
            points_market.append({
                "timestamp": str(d.date()),
                "deal_amount": f"{row['amount']:.0f} CNY (universe aggregate)",
                "rate_of_change": f"{row['rate']:+.1%} d-o-d, robust z={row['z']:.1f}",
                "relevant_news": f"{heads} (CCTV, {d.date()})",
                "rationale": (
                    f"Per steps 2-3: universe-aggregate deal amount anomaly "
                    f"(robust z={row['z']:.1f}, d-o-d {row['rate']:+.1%}) is "
                    f"temporally aligned with {len(matched)} CCTV macro news "
                    f"item(s) on the same date hitting keyword classes "
                    f"{classes}. Dual-requirement satisfied at daily "
                    f"resolution; alignment language only, no causal claim."),
            })
            calibration["market_dual_flags"] += 1
        else:
            calibration["watchlisted_quant_only"] += 1

    # ---- step 4: stock-level recent anomalies + per-stock news ----------------
    recent_start = sessions[-min(recent_sessions, len(sessions))]
    recent = []
    for sym, anom in per_symbol_anomalies.items():
        r = anom[anom["date"] >= recent_start]
        if len(r):
            recent.append((sym, r.sort_values("z", key=lambda s: s.abs(),
                                              ascending=False).iloc[0]))
    recent.sort(key=lambda t: abs(t[1]["z"]), reverse=True)
    reasoning.append(
        f"4. Stock level: {len(recent)} symbols have anomaly clusters in the "
        f"last {recent_sessions} sessions; per-stock news feeds are only "
        f"available for the RECENT window (source limitation), so historical "
        f"stock anomalies are watchlisted, not flagged. Fetching news for the "
        f"top {min(stock_top, len(recent))} symbols.")

    next_session_map = {d: (sessions[i + 1] if i + 1 < len(sessions) else None)
                        for i, d in enumerate(sessions)}
    for sym, row in recent[:stock_top]:
        try:
            news = fetch_stock_news(sym)
            calibration["stock_news_symbols_fetched"] += 1
        except Exception as exc:  # noqa: BLE001
            reasoning.append(
                f"4x. News fetch failed for {sym}: {type(exc).__name__} — "
                "anomaly watchlisted (dual-requirement unsatisfied).")
            calibration["watchlisted_quant_only"] += 1
            continue
        matched = correlate_stock_news(row["date"], news,
                                       next_session_map.get(row["date"]))
        if len(matched):
            item = matched.iloc[0]
            points_stock.append({
                "timestamp": str(row["date"].date()),
                "deal_amount": f"{row['amount']:.0f} CNY",
                "rate_of_change": (f"{row['rate']:+.1%} d-o-d, "
                                   f"robust z={row['z']:.1f}"),
                "relevant_news": (f"{item['headline']} ({item['source']}, "
                                  f"{item['timestamp'].date()})"),
                "rationale": (
                    f"Per steps 1-4: {sym} deal-amount anomaly (robust "
                    f"z={row['z']:.1f}, d-o-d {row['rate']:+.1%}) on "
                    f"{row['date'].date()} is temporally aligned with "
                    f"{len(matched)} news item(s) from its keyword-matched "
                    f"feed (published {item['timestamp'].date()}, within the "
                    f"allowed anomaly-day/+1-session window). Dual-requirement "
                    f"satisfied at daily resolution; no causal claim."),
            })
            calibration["stock_dual_flags"] += 1
        else:
            calibration["watchlisted_quant_only"] += 1

    # ---- verification pass -----------------------------------------------------
    reasoning.append(
        "5. Verification: calibration — "
        f"{calibration}; every flagged point satisfies the dual-requirement "
        "(quantitative anomaly + same-day/next-session news correlation at "
        "daily resolution); watchlisted quantitative-only anomalies are "
        "disclosed above and excluded from critical_information_points; no "
        "causal language used.")
    reasoning.append(
        "6. Scope: report is monitoring information for review; thresholds are "
        "research configuration (DECISIONS 2.8); no trading authority.")

    report_market = emit_report(reasoning, points_market)
    report_stocks = emit_report(reasoning, points_stock)
    write_json(study_dir / "critical_points_market.json", report_market)
    write_json(study_dir / "critical_points_stocks.json", report_stocks)

    full = {
        "run_id": run_id,
        "pre_registration": "docs/DECISIONS.md section 2.8 (H5)",
        "provider": PROVIDER_NAME,
        "news_sources": {"stock": PROVIDER_NEWS_SOURCE_STOCK,
                         "macro": PROVIDER_NEWS_SOURCE_MACRO},
        "calibration": calibration,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_json(study_dir / "report.json", full)
    _write_md(report_market, report_stocks, calibration, study_dir / "report.md")
    return study_dir


def _load_bars(bars_dir: Path) -> dict[str, pd.DataFrame]:
    """Load cached bar frames with amount columns (same snapshot as H2-H4)."""
    bars = {}
    for path in sorted(bars_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["date"])
        if "amount" in df.columns and len(df) >= BASELINE_WINDOW + 1:
            bars[path.stem] = df
    return bars


def _write_md(market: dict, stocks: dict, calibration: dict, path: Path) -> None:
    lines = [
        "# Critical information point monitor (H5)",
        "",
        f"- Pre-registration: docs/DECISIONS.md 2.8 | calibration: {calibration}",
        "- Dual-requirement: CRITICAL = deal-amount anomaly AND correlated news. "
        "Watchlisted quantitative-only anomalies are disclosed in reasoning_steps.",
        "",
        "## Market-level critical points (CCTV macro correlation)",
        "",
    ]
    if market["critical_information_points"]:
        for p in market["critical_information_points"]:
            lines += [f"- **{p['timestamp']}** — {p['deal_amount']} "
                      f"({p['rate_of_change']}) — {p['relevant_news']}",
                      f"  - {p['rationale']}"]
    else:
        lines.append("- None (all market anomalies watchlisted: no same-day "
                     "macro keyword match, or fetch failure disclosed).")
    lines += ["", "## Stock-level critical points (recent window)", ""]
    if stocks["critical_information_points"]:
        for p in stocks["critical_information_points"]:
            lines += [f"- **{p['timestamp']}** — {p['deal_amount']} "
                      f"({p['rate_of_change']}) — {p['relevant_news']}",
                      f"  - {p['rationale']}"]
    else:
        lines.append("- None (recent anomalies without correlatable news, or "
                     "fetch failures, are watchlisted).")
    lines += ["", "## Reasoning trail (both reports share the trail)", ""]
    for step in market["reasoning_steps"]:
        lines.append(f"- {step}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
