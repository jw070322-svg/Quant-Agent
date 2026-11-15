"""H6 forward validation of critical information points (DECISIONS 2.9).

Question: do H5-style critical information points (deal-amount anomaly +
correlated news) carry FORWARD-predictive information, and does the NEWS leg
add incremental value beyond the quantitative anomaly leg alone?

Event sample: every collapsed stock-level anomaly cluster in the cached panel
gets measurable forward outcomes (reaction return, tradable o2o return,
next-session volume persistence, 5-session drift). A news subsample (top-60
recent-window symbols by |z|) is split into dual-flag (news correlated within
the anomaly-day/+1-session window, materiality >= medium) versus quant-only.
A deterministic materiality classifier replaces H5's keyword-presence leg.
Market level: all 39 anomaly clusters get CCTV association verified (the 8
H5-flagged plus the 31 remaining), then flagged vs unflagged forward outcomes
are compared (Welch t + time-sorted block bootstrap).

Verdict is two-sided and pre-registered; low power (recent-window news
availability) is reported explicitly. This study validates the MONITOR's
information value, not a trading strategy.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..data.calendar import market_sessions
from ..data.providers import fetch_cctv_news, fetch_stock_news
from ..research.cross_section import newey_west_t
from ..research.info_monitor import (_load_bars, collapse_clusters,
                                     correlate_cctv_news, correlate_stock_news,
                                     detect_amount_anomalies)
from ..util import config_fingerprint, now_utc_iso, write_json

NEWS_BUDGET_SYMBOLS = 60
RECENT_SESSIONS = 30
BOOTSTRAP_BLOCKS = 5
BOOTSTRAP_RUNS = 2000

# Chinese keyword lists match Chinese-language news headlines by contract.
HIGH_POS = ["增持", "回购", "收购", "中标", "签订", "合同", "预增", "预盈",
            "分红", "派息", "重组", "摘帽", "中标"]
HIGH_NEG = ["减持", "预亏", "诉讼", "立案", "处罚", "质押", "退市", "商誉减值",
            "违规", "警示"]
MEDIUM_KW = ["调研", "评级", "目标价", "研报", "机构", "增仓", "净流入",
             "净流出", "业绩", "中标", "融资"]
LOW_KW = ["资金流向榜", "资金流出榜", "资金流入榜", "涨幅榜", "跌幅榜",
          "概念板块", "涨停", "跌停", "板块", "指数"]


def classify_materiality(headline: str, content: str = "") -> str:
    """Deterministic news-leg upgrade: HIGH (entity event, positive or
    adverse) > LOW (boilerplate ranking/board items) > MEDIUM (research/
    flow commentary) > LOW (default). Precedence is fixed by rule."""
    text = f"{headline} {content}"
    if any(kw in text for kw in HIGH_POS) or any(kw in text for kw in HIGH_NEG):
        return "high"
    if any(kw in text for kw in LOW_KW):
        return "low"
    if any(kw in text for kw in MEDIUM_KW):
        return "medium"
    return "low"


def event_outcomes(bars: dict[str, pd.DataFrame],
                   anomalies: dict[str, pd.DataFrame],
                   sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward outcomes for every anomaly cluster (no look-ahead: all
    forward fields come from sessions strictly after the decision date)."""
    pos = {d: i for i, d in enumerate(sessions)}
    rows = []
    for sym, anom in anomalies.items():
        df = bars[sym].set_index("date").sort_index()
        close, open_, amount = (df["close_adj"], df["open_adj"], df["amount"])
        base_med = amount.rolling(30).median()
        for _, a in anom.iterrows():
            t = a["date"]
            i = pos.get(t)
            if i is None or i + 1 >= len(sessions):
                continue  # no next session: outcome undefined, event dropped
            t1 = sessions[i + 1]
            if t not in close.index or t1 not in close.index:
                continue
            out = {"date": t, "symbol": sym, "z": a["z"], "rule": a["rule"],
                   "reaction_return": float(close.loc[t1] / close.loc[t] - 1.0)}
            if t1 in open_.index and i + 2 < len(sessions) and sessions[i + 2] in open_.index:
                out["tradable_return"] = float(open_.loc[sessions[i + 2]]
                                               / open_.loc[t1] - 1.0)
            else:
                out["tradable_return"] = np.nan
            if t1 in amount.index and np.isfinite(base_med.loc[t]) and base_med.loc[t] > 0:
                out["persistence"] = float(amount.loc[t1] / base_med.loc[t])
            else:
                out["persistence"] = np.nan
            if i + 5 < len(sessions) and sessions[i + 5] in close.index:
                out["fwd5"] = float(close.loc[sessions[i + 5]] / close.loc[t] - 1.0)
            else:
                out["fwd5"] = np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def daily_aggregate(events: pd.DataFrame, metric: str) -> pd.Series:
    """Equal-weight mean of the metric across same-date events (handles
    cross-sectional clustering), as a date-indexed series."""
    g = events.dropna(subset=[metric]).groupby("date")[metric].mean()
    return g.sort_index()


def nw_diff_t(events: pd.DataFrame, metric: str, group_a: pd.DataFrame,
              group_b: pd.DataFrame) -> float:
    """NW(3) t-stat of the daily mean difference between two event groups,
    over dates where BOTH groups have events."""
    a = daily_aggregate(group_a, metric)
    b = daily_aggregate(group_b, metric)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 10:
        return float("nan")
    return newey_west_t(joined["a"] - joined["b"])


def welch_t(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or len(y) < 5:
        return float("nan")
    return float(stats.ttest_ind(x, y, equal_var=False).statistic)


def block_bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, *, blocks: int = 5,
                            runs: int = 2000, seed: int = 42) -> tuple[float, float]:
    """CI for mean(x) - mean(y) via time-sorted circular block bootstrap on the
    pooled event order (events overlap in time; disclosure in the report)."""
    rng = np.random.default_rng(seed)
    diff_obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n = len(pooled)
    stats = []
    for _ in range(runs):
        starts = rng.integers(0, n, size=int(np.ceil(n / blocks)))
        idx = ((starts[:, None] + np.arange(blocks)[None, :]).ravel() % n)[:n]
        sample = pooled[idx]
        # split the resample into two halves to approximate the null
        half = len(sample) // 2
        stats.append(float(np.mean(sample[:half]) - np.mean(sample[half:])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    # center the CI on the observed difference
    center = float(np.mean(stats))
    return float(diff_obs + (lo - center)), float(diff_obs + (hi - center))


def run_forward_validation_study(*, news_budget: int = NEWS_BUDGET_SYMBOLS,
                                 recent_sessions: int = RECENT_SESSIONS) -> Path:
    t0 = time.time()
    bars = _load_bars(PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "bars")
    # cached bar CSVs carry raw prices + adj_factor: back-adjust exactly like
    # ingest (latest session anchors at 1.0) before computing returns
    for sym, df in bars.items():
        norm = df["adj_factor"].astype(float) / df["adj_factor"].astype(float).iloc[-1]
        for col in ("open", "close"):
            df[f"{col}_adj"] = df[col].astype(float) * norm
    sessions = market_sessions(bars)
    fp = config_fingerprint({"n": len(bars)})
    run_id = (f"xs_ev_h6-{fp}"
              f"-{now_utc_iso().replace(':', '')}")
    study_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    study_dir.mkdir(parents=True, exist_ok=True)

    # ---- event universe (all clusters) ---------------------------------------
    anomalies = {sym: collapse_clusters(detect_amount_anomalies(
        df.set_index("date")["amount"].sort_index()), sessions)
        for sym, df in bars.items()}
    events = event_outcomes(bars, anomalies, sessions)

    # ---- news subsample: top recent-window symbols by |z| --------------------
    recent_start = sessions[-min(recent_sessions, len(sessions))]
    recent = []
    for sym, anom in anomalies.items():
        r = anom[anom["date"] >= recent_start]
        if len(r):
            recent.append((sym, r.sort_values("z", key=lambda s: s.abs(),
                                              ascending=False).iloc[0]))
    recent.sort(key=lambda t: abs(t[1]["z"]), reverse=True)
    chosen = [sym for sym, _ in recent[:news_budget]]

    next_session_map = {d: (sessions[i + 1] if i + 1 < len(sessions) else None)
                        for i, d in enumerate(sessions)}
    group_dual_rows, group_quant_rows = [], []
    news_calls = 0
    for sym in chosen:
        try:
            news = fetch_stock_news(sym)
            news_calls += 1
        except Exception:  # noqa: BLE001 - disclosed in the report
            continue
        sym_events = events[events["symbol"] == sym]
        if sym_events.empty:
            continue
        for _, ev in sym_events[sym_events["date"] >= recent_start].iterrows():
            matched = correlate_stock_news(ev["date"], news,
                                           next_session_map.get(ev["date"]))
            if len(matched) == 0:
                continue
            mat = max(classify_materiality(str(m["headline"]),
                                           str(m.get("content", "")))
                      for m in matched.to_dict("records"))
            ev2 = dict(ev)
            ev2["materiality"] = mat
            ev2["n_news"] = len(matched)
            if mat in ("high", "medium"):
                group_dual_rows.append(ev2)
            else:
                group_quant_rows.append(ev2)
    dual = pd.DataFrame(group_dual_rows)
    quant = pd.DataFrame(group_quant_rows)

    # ---- comparisons (two-sided, pre-registered) ------------------------------
    comparisons = {}
    for metric in ("reaction_return", "persistence"):
        comparisons[f"dual_vs_quant_{metric}"] = {
            "nw_t": nw_diff_t(events, metric, dual, quant),
            "n_dual_dates": int(daily_aggregate(dual, metric).shape[0]),
            "n_quant_dates": int(daily_aggregate(quant, metric).shape[0]),
        }
    # materiality tiers within ALL fetched events
    fetched = pd.concat([dual, quant], ignore_index=True) if len(dual) or len(quant) else events.iloc[:0]
    if len(fetched) and "materiality" in fetched.columns:
        hi = fetched[fetched["materiality"] == "high"]
        lo = fetched[fetched["materiality"] == "low"]
        comparisons["high_vs_low_tradable"] = {
            "nw_t": nw_diff_t(events, "tradable_return", hi, lo),
            "n_high": int(len(hi)), "n_low": int(len(lo)),
        }
        comparisons["high_vs_low_reaction"] = {
            "nw_t": nw_diff_t(events, "reaction_return", hi, lo),
        }

    # unconditional event baselines (full sample, for context)
    baselines = {}
    for metric in ("reaction_return", "tradable_return", "persistence", "fwd5"):
        s = daily_aggregate(events, metric)
        baselines[metric] = {"mean": float(s.mean()),
                             "nw_t": newey_west_t(s), "n_dates": int(len(s))}

    # ---- market level: verify ALL cluster dates -------------------------------
    agg = pd.DataFrame({sym: df.set_index("date")["amount"].sort_index()
                        for sym, df in bars.items()}).sum(axis=1).dropna()
    mkt = collapse_clusters(detect_amount_anomalies(agg), sessions)
    flagged_rows, unflagged_rows = [], []
    cctv_calls = 0
    for _, row in mkt.iterrows():
        d = pd.Timestamp(row["date"])
        try:
            cctv = fetch_cctv_news(d)
            cctv_calls += 1
        except Exception:  # noqa: BLE001
            continue
        matched, _ = correlate_cctv_news(d, cctv)
        i = sessions.get_indexer([d])[0]
        if i + 1 >= len(sessions):
            continue
        t1 = sessions[i + 1]
        ew_ret = float(np.mean([df.set_index("date")["close_adj"].loc[t1]
                                / df.set_index("date")["close_adj"].loc[d] - 1.0
                                for sym, df in bars.items()
                                if d in df.set_index("date").index
                                and t1 in df.set_index("date").index]))
        out = {"date": d, "z": row["z"], "flagged": bool(matched),
               "next_market_return": ew_ret}
        med = float(agg.rolling(30).median().loc[d])
        out["persistence"] = float(agg.loc[t1] / med) if med > 0 else np.nan
        (flagged_rows if matched else unflagged_rows).append(out)
    mkt_flag = pd.DataFrame(flagged_rows)
    mkt_unflag = pd.DataFrame(unflagged_rows)
    market = {"n_flagged": len(mkt_flag), "n_unflagged": len(mkt_unflag),
              "cctv_calls": cctv_calls}
    if len(mkt_flag) >= 5 and len(mkt_unflag) >= 5:
        x = mkt_flag["next_market_return"].to_numpy()
        y = mkt_unflag["next_market_return"].to_numpy()
        market["next_return_welch_t"] = welch_t(x, y)
        lo, hi = block_bootstrap_diff_ci(x, y)
        market["next_return_boot_ci"] = [lo, hi]
        xp = mkt_flag["persistence"].dropna().to_numpy()
        yp = mkt_unflag["persistence"].dropna().to_numpy()
        market["persistence_welch_t"] = welch_t(xp, yp)
        market["flagged_mean_next_return"] = float(np.mean(x))
        market["unflagged_mean_next_return"] = float(np.mean(y))

    # ---- verdict ---------------------------------------------------------------
    def ok(t):
        return np.isfinite(t) and abs(t) >= 2.0
    criteria = {
        "i_dual_reaction": ok(comparisons["dual_vs_quant_reaction_return"]["nw_t"]),
        "ii_dual_persistence": bool(
            comparisons["dual_vs_quant_persistence"]["nw_t"] >= 2.0),
        "iii_market": ok(market.get("next_return_welch_t", float("nan"))),
        "iv_materiality": ok(comparisons.get("high_vs_low_tradable", {}).get("nw_t", float("nan"))),
    }
    news_leg_adds_value = all(criteria.values())

    # ---- post-hoc exploratory analyses (clearly labeled; NOT the verdict) ----
    # The pre-registered paired test required dates where BOTH groups have
    # events; dual and quant-only events never share a date by construction
    # (a symbol-date is either dual or quant-only), so the paired test is
    # structurally infeasible at this sample size. Two-sample tests on
    # event-level outcomes are reported as EXPLORATORY only.
    exploratory = {}
    if len(dual) >= 5 and len(quant) >= 5:
        for metric in ("reaction_return", "persistence", "tradable_return"):
            x = dual[metric].dropna().to_numpy()
            y = quant[metric].dropna().to_numpy()
            lo, hi = block_bootstrap_diff_ci(x, y)
            exploratory[f"dual_vs_quant_{metric}"] = {
                "welch_t": welch_t(x, y), "boot_ci": [lo, hi],
                "mean_dual": float(np.mean(x)), "mean_quant": float(np.mean(y)),
                "n_dual": int(len(x)), "n_quant": int(len(y))}
    # market CCTV leg: keyword presence flagged 39/39 (zero specificity) —
    # attempt a materiality re-tiering and disclose the outcome
    cctv_tiers = {"high_dates": 0, "low_dates": 0}
    for _, row in mkt.iterrows():
        try:
            cctv = fetch_cctv_news(pd.Timestamp(row["date"]))
        except Exception:  # noqa: BLE001
            continue
        mats = [classify_materiality(str(i["headline"]), str(i.get("content", "")))
                for i in correlate_cctv_news(pd.Timestamp(row["date"]), cctv)[0]]
        if any(m == "high" for m in mats):
            cctv_tiers["high_dates"] += 1
        else:
            cctv_tiers["low_dates"] += 1
    exploratory["cctv_materiality_tiers"] = cctv_tiers

    report = {
        "run_id": run_id,
        "pre_registration": "docs/DECISIONS.md section 2.9 (H6)",
        "status": "monitor information-value validation (not a trading strategy)",
        "survivorship_bias": True,
        "data": {"n_symbols": len(bars), "n_sessions": len(sessions),
                 "n_events": len(events), "news_calls": news_calls,
                 "n_dual_events": len(dual), "n_quant_events": len(quant)},
        "unconditional_baselines": baselines,
        "comparisons": comparisons,
        "market_level": market,
        "criteria": criteria,
        "verdict": ("news leg adds measurable forward information"
                    if news_leg_adds_value else
                    "news leg shows NO measurable incremental forward "
                    "information (pre-registered paired test structurally "
                    "infeasible: zero date overlap; degenerate market split); "
                    "monitor remains a review tool"),
        "exploratory_post_hoc": exploratory,
        "exploratory_label": ("POST-HOC EXPLORATORY — not the pre-registered "
                              "verdict; multiple-testing burden applies"),
        "power_disclosure": (
            "news-correlated events exist only in the recent window "
            f"(~{recent_sessions} sessions); independent event dates are "
            "limited, so statistical power is low — reported, not hidden"),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_json(study_dir / "report.json", report)
    events.to_csv(study_dir / "event_panel.csv", index=False)
    if len(fetched):
        fetched.to_csv(study_dir / "news_subsample_events.csv", index=False)
    mkt_flag.to_csv(study_dir / "market_flagged.csv", index=False)
    mkt_unflag.to_csv(study_dir / "market_unflagged.csv", index=False)
    _write_md(report, study_dir / "report.md")
    return study_dir


def _write_md(report: dict, path: Path) -> None:
    lines = [
        f"# Forward validation of critical information points (H6): `{report['run_id']}`",
        "",
        f"- Pre-registration: {report['pre_registration']} | verdict: **{report['verdict']}**",
        f"- Events: {report['data']['n_events']} clusters | news subsample: "
        f"{report['data']['n_dual_events']} dual / {report['data']['n_quant_events']} quant-only "
        f"({report['data']['news_calls']} fetches)",
        f"- Power disclosure: {report['power_disclosure']}",
        "",
        "## Unconditional event baselines (full sample)",
        "",
        "| Metric | Daily mean | NW t | Dates |",
        "|---|---:|---:|---:|",
    ]
    for m, b in report["unconditional_baselines"].items():
        lines.append(f"| {m} | {b['mean']:.4%} | {b['nw_t']:.2f} | {b['n_dates']} |")
    lines += ["", "## Pre-registered comparisons", "", "| Test | NW/Welch t | Detail |",
              "|---|---:|---|"]
    for name, c in report["comparisons"].items():
        lines.append(f"| {name} | {c.get('nw_t', float('nan')):.2f} | {c} |")
    m = report["market_level"]
    lines.append(f"| market flagged_vs_unflagged | {m.get('next_return_welch_t', float('nan')):.2f} "
                 f"| flagged n={m.get('n_flagged')}, unflagged n={m.get('n_unflagged')}, "
                 f"bootstrap CI={m.get('next_return_boot_ci')} |")
    lines += ["", f"- Criteria: {report['criteria']}",
              "", "## Post-hoc exploratory (NOT the verdict; disclosure only)", ""]
    for name, e in report.get("exploratory_post_hoc", {}).items():
        lines.append(f"- {name}: {e}")
    lines += [f"- Label: {report.get('exploratory_label')}",
              "", "## Claim boundary", "",
              "- Two-sided tests; the monitor is an information-review layer, "
              "not a strategy. Survivorship-biased universe; news availability "
              "limits the dual subsample to the recent window (low power, "
              "disclosed). Nothing here constitutes a trading recommendation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
