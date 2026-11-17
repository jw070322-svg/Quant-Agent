"""Cross-sectional factor studies (H2 valuation, H3 technical attribution).

Shared evaluation core (`_evaluate_factor_df`): per-date winsorize/z-score ->
(a) Spearman RankIC with Newey-West t on the IC series, (b) quintile
portfolios run through the SAME simulator (next-open fills, commission/
slippage/stamp tax, T+1) -> net-of-cost direction-aligned spreads,
(c) Fama-MacBeth regressions with NW(lag3) t-stats.

Both studies are EXPLORATORY SCREENING on the current-membership universe
(SURVIVORSHIP BIAS) and can never approve a bundle — pre-registered in
docs/DECISIONS.md 2.5 (H2) and 2.6 (H3).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_market_config, load_risk_config
from ..data.calendar import market_sessions
from ..data.providers import (PROVIDER_NAME, PROVIDER_UNIVERSE_SOURCE,
                              PROVIDER_VALUATION_SOURCE, update_xs_universe)
from ..execution.simulator import run_simulation
from ..features.indicators import FEATURE_NAMES, compute_features
from ..research.metrics import summarize
from ..risk.checks import RiskLimits
from ..util import config_fingerprint, now_utc_iso, write_json

VALUATION_INDICATORS = ("市盈率(TTM)", "市净率", "总市值")
REBALANCE_SESSIONS = 20
LOOKBACK_GAP_DAYS = 5          # as-of tolerance for the ~2-day-sampled valuation
MIN_HISTORY_SESSIONS = 240     # H2: for mom_12m_1m
MIN_HISTORY_TECHNICAL = 61     # H3: v1 feature warm-up
NW_LAG = 3
WINSOR = (0.01, 0.99)

FACTOR_EXPECTED_SIGN = {
    "EP": +1, "BP": +1, "size": -1, "mom_1m": -1,
    "mom_12m_1m": +1, "lowvol": +1, "reversal_5": +1, "turnover": 0,
}

# H3 technical attribution: classic-TA conventions pre-registered in §2.6
TA_EXPECTED_SIGN = {
    "ret_1": -1, "ret_5": -1, "ret_20": -1,
    "close_over_ma5": +1, "close_over_ma20": +1, "close_over_ma60": +1,
    "ma5_over_ma20": +1, "macd_norm": +1, "macd_hist_norm": +1,
    "rsi14": -1, "bb_pos20": -1, "bb_width20": 0,
    "vol_5": -1, "vol_20": -1, "atr14_norm": -1, "volume_ratio20": 0,
    "ta_composite": +1,
}


def _winsorize_zscore(xs: pd.Series) -> pd.Series:
    """Per-date cross-section: winsorize at 1%/99%, then z-score."""
    s = pd.to_numeric(xs, errors="coerce").dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])
    s = s.clip(lo, hi)
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(dtype=float)
    return (s - s.mean()) / sd


def newey_west_t(series: pd.Series, lag: int = NW_LAG) -> float:
    """t-stat of the mean with Newey-West HAC variance (lags as pre-registered)."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    demeaned = x - x.mean()
    gamma0 = float(np.mean(demeaned ** 2))
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        gamma_l = float(np.mean(demeaned[l:] * demeaned[:-l]))
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gamma_l
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def _load_xs_frames(bars_dir: Path, val_dir: Path) -> dict:
    """bars: symbol -> frame (with back-adjusted *_adj columns); valuation:
    symbol -> wide date-indexed frame(pe, pb, mv)."""
    bars, valuation = {}, {}
    for path in sorted(bars_dir.glob("*.csv")):
        symbol = path.stem
        df = pd.read_csv(path, parse_dates=["date"])
        if "adj_factor" in df.columns:
            # back-adjust exactly like ingest: latest session anchors at 1.0
            norm = df["adj_factor"].astype(float) / df["adj_factor"].astype(float).iloc[-1]
            for col in ("open", "high", "low", "close"):
                df[f"{col}_adj"] = df[col].astype(float) * norm
        if len(df) >= 100:
            bars[symbol] = df
    for path in sorted(val_dir.glob("*.csv")):
        stem = path.stem
        for ind in VALUATION_INDICATORS:
            if stem.endswith(f"_{ind}"):
                symbol = stem[: -len(f"_{ind}")]
                v = pd.read_csv(path, parse_dates=["date"])
                valuation.setdefault(symbol, {})[ind] = v.set_index("date")["value"]
    wide = {}
    for symbol, series_by_ind in valuation.items():
        if len(series_by_ind) < len(VALUATION_INDICATORS):
            continue
        wide[symbol] = pd.DataFrame(series_by_ind)
    return {"bars": bars, "valuation": wide}


def _forward_window(sessions: pd.DatetimeIndex, t: pd.Timestamp):
    """(entry_date, exit_date) for the pre-registered 20-session o2o window."""
    idx = sessions.get_indexer([t])[0]
    if idx == -1 or idx + 1 + REBALANCE_SESSIONS >= len(sessions):
        return None
    return sessions[idx + 1], sessions[idx + 1 + REBALANCE_SESSIONS]


def _factor_rows(frames: dict, sessions: pd.DatetimeIndex,
                 rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """H2 valuation factors: one row per (rebalance_date, symbol)."""
    close_adj = {s: df.set_index("date")["close_adj"] for s, df in frames["bars"].items()}
    open_adj = {s: df.set_index("date")["open_adj"] for s, df in frames["bars"].items()}
    volume = {s: df.set_index("date")["volume"] for s, df in frames["bars"].items()}
    valuation = frames["valuation"]

    rows = []
    for t in rebalance_dates:
        fw = _forward_window(sessions, t)
        if fw is None:
            continue
        entry_d, exit_d = fw
        for sym, ca in close_adj.items():
            if t not in ca.index:
                continue
            hist = ca.loc[:t]
            if len(hist) < MIN_HISTORY_SESSIONS:
                continue
            val = valuation.get(sym)
            if val is None:
                continue
            # R1 fix: BACKWARD-ONLY point-in-time join — the valuation row must
            # exist at or before the signal date (available_at <= signal_at),
            # with a maximum permitted age. The previous forward window
            # (t..t+5d) let future valuation records enter historical signals.
            window = val.loc[:t].tail(1)
            if window.empty or (t - window.index[0]).days > LOOKBACK_GAP_DAYS:
                continue
            pe_ttm, pb, mv = window.iloc[0]["市盈率(TTM)"], window.iloc[0]["市净率"], window.iloc[0]["总市值"]
            if not np.isfinite(pb) or pb <= 0 or not np.isfinite(mv) or mv <= 0:
                continue
            close_t = float(ca.loc[t])
            vol20 = hist.pct_change().iloc[-20:].std(ddof=1)
            if not np.isfinite(vol20) or vol20 <= 0:
                continue
            # R7 fix: forward outcome availability no longer gates the SIGNAL
            # row — rows with an incomplete forward window are retained (fwd
            # NaN) and drop out only in the evaluation paths.
            fwd = float(open_adj[sym].loc[exit_d] / open_adj[sym].loc[entry_d] - 1.0) \
                if entry_d in open_adj[sym].index and exit_d in open_adj[sym].index else np.nan
            dollar_vol = close_t * float(volume[sym].loc[t]) if t in volume[sym].index else np.nan
            rows.append({
                "date": t, "symbol": sym, "fwd_ret": fwd,
                "EP": 1.0 / pe_ttm if np.isfinite(pe_ttm) else np.nan,
                "BP": 1.0 / pb,
                "size": np.log(mv),
                "mom_1m": close_t / float(hist.iloc[-21]) - 1.0 if len(hist) >= 21 else np.nan,
                "mom_12m_1m": float(hist.iloc[-21]) / float(hist.iloc[-241]) - 1.0 if len(hist) >= 241 else np.nan,
                "lowvol": -vol20,
                "reversal_5": -(close_t / float(hist.iloc[-6]) - 1.0) if len(hist) >= 6 else np.nan,
                "turnover": np.log(dollar_vol / mv) if np.isfinite(dollar_vol) and dollar_vol > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _technical_factor_rows(frames: dict, sessions: pd.DatetimeIndex,
                           rebalance_dates: list[pd.Timestamp],
                           feature_names: list[str]) -> pd.DataFrame:
    """H3 technical attribution: the shared causal feature library computed
    per symbol, sampled at rebalance dates (coverage gate = 61 sessions)."""
    close_adj = {s: df.set_index("date")["close_adj"] for s, df in frames["bars"].items()}
    open_adj = {s: df.set_index("date")["open_adj"] for s, df in frames["bars"].items()}
    feats = {s: compute_features(df, feature_names).set_index(
        frames["bars"][s]["date"]) for s, df in frames["bars"].items()}

    rows = []
    for t in rebalance_dates:
        fw = _forward_window(sessions, t)
        if fw is None:
            continue
        entry_d, exit_d = fw
        for sym, ft in feats.items():
            if t not in ft.index:
                continue
            if len(close_adj[sym].loc[:t]) < MIN_HISTORY_TECHNICAL:
                continue
            vals = ft.loc[t]
            if not np.isfinite(vals.to_numpy(dtype=float)).all():
                continue  # warm-up or degenerate rows are excluded, never faked
            fwd = float(open_adj[sym].loc[exit_d] / open_adj[sym].loc[entry_d] - 1.0) \
                if entry_d in open_adj[sym].index and exit_d in open_adj[sym].index else np.nan
            # R7 fix: incomplete forward windows no longer gate the
            # signal row (outcomes join in the evaluation path only)
            row = {"date": t, "symbol": sym, "fwd_ret": fwd}
            row.update({name: float(vals[name]) for name in feature_names})
            rows.append(row)
    return pd.DataFrame(rows)


def _quintile_sims(panel: dict, sessions: pd.DatetimeIndex, factor_df: pd.DataFrame,
                   factor: str, *, mcfg: dict, limits: RiskLimits,
                   initial_equity: float, run_id: str,
                   bars_cache: dict | None = None) -> dict[str, pd.Series]:
    """Five long-only quintile portfolios, one simulation EACH, through the
    same broker/costs; returns daily-return series per quintile (net)."""
    grp = factor_df[["date", "symbol", factor]].dropna()
    members: dict[pd.Timestamp, list[list[str]]] = {}
    for date, sub in grp.groupby("date"):
        ranked = sub.sort_values(factor, ascending=False)
        n = len(ranked)
        q = max(int(np.ceil(n / 5)), 1)
        buckets = []
        for qi in range(5):
            chunk = ranked.iloc[qi * q:(qi + 1) * q]
            buckets.append(list(chunk["symbol"]))
        members[pd.Timestamp(date)] = buckets

    def build_fn(buckets_qi: int):
        target_for = {}
        for d, buckets in members.items():
            names = buckets[buckets_qi]
            if not names:
                continue
            idx = sessions.get_indexer([d])
            if idx[0] == -1 or idx[0] + 1 >= len(sessions):
                continue
            # within-quintile equal weight, scaled to the gross-exposure policy
            w = min(1.0 / len(names), limits.max_weight_per_name) * limits.max_gross_exposure
            # R6 fix: submit at the SIGNAL session close; the simulator
            # fills at the next open (signal+1), matching the label entry
            target_for[sessions[idx[0]]] = {s: w for s in names}

        def fn(t, account, prices):
            if t in target_for:
                return dict(target_for[t])
            return None  # HOLD between rebalances

        return fn

    out = {}
    for qi in range(5):
        sim = run_simulation(panel, sessions, build_fn(qi), market_cfg=mcfg,
                             limits=limits, initial_equity=initial_equity,
                             run_id=f"{run_id}-q{qi + 1}", bars_cache=bars_cache)
        out[f"Q{qi + 1}"] = sim.daily_returns
    return out


def _evaluate_factor_df(factor_df: pd.DataFrame, factors: list[str],
                        expected_signs: dict[str, int], *, panel: dict,
                        sessions: pd.DatetimeIndex, mcfg: dict, limits: RiskLimits,
                        initial_equity: float, run_id: str, meta: dict,
                        bars_cache: dict | None = None) -> tuple[dict, Path]:
    """Shared H2/H3 evaluation core (pre-registered protocol)."""
    t0 = time.time()
    factor_df = factor_df.copy()
    raw_factors = [f for f in factors if f != "ta_composite"]
    for f in raw_factors:
        factor_df[f"z_{f}"] = factor_df.groupby("date")[f].transform(
            lambda s: _winsorize_zscore(s))
    if "ta_composite" in factors:
        # equal-weight mean of the per-date z-scores (pre-registered §2.6);
        # row-wise: each row already belongs to one date
        z_cols = [f"z_{f}" for f in raw_factors]
        factor_df["z_ta_composite"] = factor_df[z_cols].mean(axis=1)
    factor_df = factor_df.dropna(subset=["fwd_ret"])

    # ---- (a) RankIC ---------------------------------------------------------
    ic_rows = []
    for date, sub in factor_df.groupby("date"):
        if len(sub) < 10:
            continue
        for f in factors:
            z = sub[f"z_{f}"].dropna()
            joined = pd.concat([z, sub.loc[z.index, "fwd_ret"]], axis=1)
            if len(joined) < 10:
                continue
            ic = joined[f"z_{f}"].corr(joined["fwd_ret"], method="spearman")
            if np.isfinite(ic):
                ic_rows.append({"date": date, "factor": f, "ic": ic, "n": len(joined)})
    ic_df = pd.DataFrame(ic_rows)
    ic_summary = {}
    for f in factors:
        s = ic_df[ic_df["factor"] == f]["ic"]
        mean_ic = float(s.mean()) if len(s) else float("nan")
        t = newey_west_t(s)
        expected = expected_signs[f]
        ic_summary[f] = {
            "mean_ic": mean_ic, "nw_t": t, "n_windows": int(len(s)),
            "expected_sign": expected,
            "mean_ic_aligned": mean_ic * expected if expected else mean_ic,
            "passes_screen": bool(np.isfinite(t) and abs(t) >= 2.0
                                  and (expected == 0 or np.sign(mean_ic) == expected)),
        }

    # ---- (b) quintile portfolios (net of costs, same simulator) -------------
    quintile = {}
    for f in factors:
        try:
            rets = _quintile_sims(panel, sessions, factor_df, f"z_{f}",
                                  mcfg=mcfg, limits=limits,
                                  initial_equity=initial_equity,
                                  run_id=f"{run_id}-{f}", bars_cache=bars_cache)
            expected = expected_signs[f]
            obs_sign = ic_summary[f]["mean_ic"] if np.isfinite(ic_summary[f]["mean_ic"]) else 0.0
            direction = expected if expected != 0 else (1 if obs_sign > 0 else -1)
            spy = max(int(round(len(sessions) / max((sessions[-1] - sessions[0]).days / 365.25, 1e-9))), 1)
            q_stats = {f"Q{qi}": float((1 + rets[f"Q{qi}"].fillna(0)).prod() - 1)
                       for qi in range(1, 6) if f"Q{qi}" in rets}
            if direction > 0:
                spread = (rets.get("Q1", pd.Series(dtype=float))
                          - rets.get("Q5", pd.Series(dtype=float))) / 2.0
                monotone = all(q_stats.get(f"Q{i}", -9) >= q_stats.get(f"Q{i + 1}", 9) - 1e-12
                               for i in range(1, 5))
            else:
                spread = (rets.get("Q5", pd.Series(dtype=float))
                          - rets.get("Q1", pd.Series(dtype=float))) / 2.0
                monotone = all(q_stats.get(f"Q{i}", 9) <= q_stats.get(f"Q{i + 1}", -9) + 1e-12
                               for i in range(1, 5))
            spread_stats = summarize(spread, sessions_per_year=spy)
            passes_q = bool(np.isfinite(spread_stats["total_return"])
                            and spread_stats["total_return"] > 0
                            and (monotone is True or expected == 0))
            quintile[f] = {"direction": direction,
                           "quintiles_total_return": q_stats,
                           "spread_total_return": spread_stats["total_return"],
                           "spread_sharpe": spread_stats["sharpe"],
                           "spread_maxdd": spread_stats["max_drawdown"],
                           "monotone_expected_direction": monotone,
                           "passes_screen": bool(ic_summary[f]["passes_screen"] and passes_q)}
        except Exception as exc:  # noqa: BLE001
            quintile[f] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}

    # ---- (c) Fama-MacBeth ---------------------------------------------------
    fm_rows = []
    z_cols = [f"z_{f}" for f in factors]
    for date, sub in factor_df.groupby("date"):
        data = sub[["fwd_ret"] + z_cols].dropna()
        if len(data) < max(10, len(z_cols) + 5):
            continue
        X = np.column_stack([np.ones(len(data)), data[z_cols].to_numpy()])
        y = data["fwd_ret"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for j, f in enumerate(factors):
            fm_rows.append({"date": date, "factor": f, "beta": beta[j + 1]})
    fm_df = pd.DataFrame(fm_rows)
    fm_summary = {}
    for f in factors:
        s = fm_df[fm_df["factor"] == f]["beta"]
        fm_summary[f] = {"mean_beta_monthly": float(s.mean()) if len(s) else float("nan"),
                         "nw_t": newey_west_t(s), "n_windows": int(len(s))}

    study_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    study_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "meta": meta,
        "data": {"provider": PROVIDER_NAME,
                 "n_symbols_bars": len(panel),
                 "rebalance_sessions": REBALANCE_SESSIONS,
                 "n_windows": int(factor_df["date"].nunique()),
                 "avg_cross_section": float(factor_df.groupby("date").size().mean())},
        "ic": ic_summary, "quintile": quintile, "fama_macbeth": fm_summary,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_json(study_dir / "report.json", report)
    ic_df.to_csv(study_dir / "ic_series.csv", index=False)
    fm_df.to_csv(study_dir / "fm_series.csv", index=False)
    factor_df.to_csv(study_dir / "factor_panel.csv", index=False)
    _write_md(report, study_dir / "report.md")
    return report, study_dir


def run_cross_section_study(*, index: str = "000300", n_symbols: int = 150,
                            seed: int = 42, refresh_data: bool = False,
                            initial_equity: float = 1_000_000.0) -> Path:
    """H2 valuation-style factors (docs/DECISIONS.md 2.5)."""
    mcfg = load_market_config("cn")
    rcfg = load_risk_config()
    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})

    bars_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "bars"
    val_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "valuation"
    if refresh_data or not bars_dir.exists() or not list(bars_dir.glob("*.csv")):
        update_xs_universe(index, n_symbols=n_symbols, seed=seed, end=now_utc_iso()[:10])

    frames = _load_xs_frames(bars_dir, val_dir)
    bars = frames["bars"]
    if len(bars) < 30:
        raise RuntimeError(f"insufficient cross-sectional data: {len(bars)} symbols loaded")
    sessions = market_sessions(bars)
    start_idx = min(REBALANCE_SESSIONS, max(int(0.05 * len(sessions)), 5))
    rebalance_dates = [sessions[i] for i in range(start_idx, len(sessions), REBALANCE_SESSIONS)]

    factor_df = _factor_rows(frames, sessions, rebalance_dates)
    factors = list(FACTOR_EXPECTED_SIGN)
    run_id = f"xs_cn_h2-{config_fingerprint({'i': index, 'n': n_symbols, 's': seed})}-{now_utc_iso().replace(':', '')}"
    meta = {
        "pre_registration": "docs/DECISIONS.md section 2.5 (H2)",
        "universe": f"CSI {index} CURRENT constituents, seeded sample n={n_symbols}, seed={seed}",
        "survivorship_bias": True,
        "status": "EXPLORATORY SCREENING — can never approve a bundle",
        "valuation_source": PROVIDER_VALUATION_SOURCE,
        "n_symbols_valuation": len(frames["valuation"]),
    }
    _report, study_dir = _evaluate_factor_df(
        factor_df, factors, FACTOR_EXPECTED_SIGN, panel=bars, sessions=sessions,
        mcfg=mcfg, limits=limits, initial_equity=initial_equity,
        run_id=run_id, meta=meta)
    return study_dir


def run_xs_technical_study(*, index: str = "000300", n_symbols: int = 150,
                           seed: int = 42, initial_equity: float = 1_000_000.0) -> Path:
    """H3 technical-indicator attribution (docs/DECISIONS.md 2.6).
    Reuses the cached universe; no valuation data required."""
    mcfg = load_market_config("cn")
    rcfg = load_risk_config()
    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})

    bars_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "bars"
    frames = {"bars": _load_xs_frames(bars_dir, bars_dir)["bars"], "valuation": {}}
    bars = frames["bars"]
    if len(bars) < 30:
        raise RuntimeError(f"insufficient cross-sectional data: {len(bars)} symbols loaded")
    sessions = market_sessions(bars)
    start_idx = max(int(0.05 * len(sessions)), MIN_HISTORY_TECHNICAL)
    rebalance_dates = [sessions[i] for i in range(start_idx, len(sessions), REBALANCE_SESSIONS)]

    feature_names = list(FEATURE_NAMES)
    factor_df = _technical_factor_rows(frames, sessions, rebalance_dates, feature_names)
    factors = list(TA_EXPECTED_SIGN)
    run_id = f"xs_ta_h3-{config_fingerprint({'i': index, 'n': n_symbols, 's': seed})}-{now_utc_iso().replace(':', '')}"
    meta = {
        "pre_registration": "docs/DECISIONS.md section 2.6 (H3)",
        "universe": f"CSI {index} CURRENT constituents, seeded sample n={n_symbols}, seed={seed}",
        "survivorship_bias": True,
        "status": "EXPLORATORY SCREENING — can never approve a bundle",
        "factors": "16 shared-library technical indicators + equal-weight ta_composite",
    }
    bars_cache = {sym: df.set_index("date").to_dict("index") for sym, df in bars.items()}
    _report, study_dir = _evaluate_factor_df(
        factor_df, factors, TA_EXPECTED_SIGN, panel=bars, sessions=sessions,
        mcfg=mcfg, limits=limits, initial_equity=initial_equity,
        run_id=run_id, meta=meta, bars_cache=bars_cache)
    return study_dir


def _write_md(report: dict, path: Path) -> None:
    meta = report["meta"]
    lines = [
        f"# Cross-sectional factor screening study: `{report['run_id']}`",
        "",
        f"- Pre-registration: {meta['pre_registration']} | Universe: {meta['universe']}",
        f"- **Survivorship bias: {meta['survivorship_bias']} → exploratory "
        f"screening; results can never approve a bundle**",
        f"- Windows: {report['data']['n_windows']} | average cross-section: "
        f"{report['data']['avg_cross_section']:.0f} names",
        "",
        "| Factor | Expected dir. | meanIC | Aligned | NW t | IC screen | Dir. | "
        "Net spread | Spread Sharpe | Monotone | Portfolio screen |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for f, s in report["ic"].items():
        q = report["quintile"].get(f, {})
        sign = {1: "+", -1: "−", 0: "two-sided"}[s["expected_sign"]]
        lines.append(
            f"| {f} | {sign} | {s['mean_ic']:.4f} | {s['mean_ic_aligned']:.4f} | "
            f"{s['nw_t']:.2f} "
            f"| {'✓' if s['passes_screen'] else '✗'} "
            f"| {q.get('direction', float('nan'))} "
            f"| {q.get('spread_total_return', float('nan')):.2%} "
            f"| {q.get('spread_sharpe', float('nan')):.2f} "
            f"| {q.get('monotone_expected_direction')} "
            f"| {'✓' if q.get('passes_screen') else '✗'} |")
    lines += ["", "## Fama-MacBeth (monthly coefficients)", "",
              "| Factor | Coefficient mean (monthly) | NW t | Windows |",
              "|---|---:|---:|---:|"]
    for f, s in report["fama_macbeth"].items():
        lines.append(f"| {f} | {s['mean_beta_monthly']:.5f} | {s['nw_t']:.2f} "
                     f"| {s['n_windows']} |")
    lines += ["", "Screen passers (IC + portfolio) require a new pre-registration "
              "(with historical constituents data) before entering the "
              "walk-forward pipeline."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
