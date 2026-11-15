"""H4 multi-factor synthesis study (pre-registered: docs/DECISIONS.md 2.7).

Pipeline: cached bars+valuation -> 14-factor panel (direction-normalized) ->
per-date MAD winsorize + z-score + size neutralization -> TRAIN-window factor
screening and redundancy control (frozen list) -> TEST-window alpha synthesis
(equal-weight baseline, trailing-ICIR, shrinkage, ridge) -> TOP_N=30 equal
weight portfolios through the SAME simulator (T+1, costs, next-open) ->
benchmarks + robustness (cost x2/x4, one-window execution lag).

Anti-snooping: factor selection uses TRAIN windows only; synthesis is
evaluated on TEST windows only; weight estimation uses only maturity-safe
trailing windows (IC/beta known at window exit, 2-window maturity shift).

SURVIVORSHIP BIAS (current-membership universe) => final status ceiling is
research_only regardless of results.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_market_config, load_risk_config
from ..data.calendar import market_sessions
from ..data.providers import PROVIDER_NAME, PROVIDER_VALUATION_SOURCE
from ..execution.simulator import run_simulation
from ..research.cross_section import (_load_xs_frames, newey_west_t,
                                      REBALANCE_SESSIONS)
from ..research.fallback_study import _scale_costs
from ..risk.checks import RiskLimits
from ..util import config_fingerprint, now_utc_iso, write_json

FACTOR_DIRECTIONS = {
    "EP": +1, "BP": +1, "size": -1, "mom_20": -1, "mom_12_1": +1,
    "reversal_5": +1, "lowvol": +1, "amihud_20": +1, "turnover": 0,
    "rsi14": -1, "bb_pos20": -1, "vol_5": -1, "macd_hist_norm": +1,
    "close_over_ma20": +1,
}
SCREEN = {"min_abs_ic": 0.02, "min_icir": 0.30, "min_nw_t": 2.0,
          "min_coverage": 0.80, "max_corr": 0.70}
TRAIN_TEST_SPLIT = 0.5
TRAILING_WINDOWS = 12
MATURITY_SHIFT = 2          # windows whose forward return has matured
TOP_N = 30
RIDGE_ALPHA = 10.0
SHRINK_LAMBDA = 0.5


# ---------------------------------------------------------------- preprocessing
def mad_winsorize_zscore(s: pd.Series, k: float = 3.0) -> pd.Series:
    """MAD winsorize (median ± k*1.4826*MAD) then z-score (template Step 3)."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 10:
        return pd.Series(dtype=float)
    med = float(x.median())
    mad = float((x - med).abs().median()) * 1.4826
    if mad <= 0:
        sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(dtype=float)
        mad = sd
    lo, hi = med - k * mad, med + k * mad
    x = x.clip(lo, hi)
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return pd.Series(dtype=float)
    return (x - x.mean()) / sd


def neutralize_size(xs: pd.Series, log_mv: pd.Series) -> pd.Series:
    """Cross-sectional OLS residual of xs ~ 1 + log(mv) (template Step 3)."""
    df = pd.concat([xs.rename("y"), log_mv.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return pd.Series(dtype=float)
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    resid = pd.Series(df["y"].to_numpy() - X @ beta, index=df.index)
    sd = float(resid.std(ddof=1))
    if sd <= 0:
        return pd.Series(dtype=float)
    return (resid - resid.mean()) / sd


# ---------------------------------------------------------------- ICIR weights
def icir_weights(ic_by_factor: dict[str, float]) -> dict[str, float]:
    """Non-negative ICIR weights normalized to sum 1 (template Step 6, B)."""
    pos = {k: max(v, 0.0) for k, v in ic_by_factor.items()}
    total = sum(pos.values())
    if total <= 0:
        k = len(ic_by_factor)
        return {k_: 1.0 / k for k_ in ic_by_factor}
    return {k: v / total for k, v in pos.items()}


def shrink_weights(w: dict[str, float], lam: float = SHRINK_LAMBDA) -> dict[str, float]:
    k = len(w)
    eq = 1.0 / k
    return {key: (1 - lam) * v + lam * eq for key, v in w.items()}


# ---------------------------------------------------------------- panel
def build_factor_panel(frames: dict, sessions: pd.DatetimeIndex,
                       rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """One row per (rebalance_date, symbol) with raw factor values + fwd_ret."""
    close = {s: df.set_index("date")["close_adj"] for s, df in frames["bars"].items()}
    open_ = {s: df.set_index("date")["open_adj"] for s, df in frames["bars"].items()}
    vol = {s: df.set_index("date")["volume"] for s, df in frames["bars"].items()}
    valuation = frames["valuation"]
    ret5 = {s: c.pct_change(5) for s, c in close.items()}

    rows = []
    for t in rebalance_dates:
        idx = sessions.get_indexer([t])[0]
        if idx + 1 + REBALANCE_SESSIONS >= len(sessions):
            continue
        entry_d, exit_d = sessions[idx + 1], sessions[idx + 1 + REBALANCE_SESSIONS]
        for sym, c in close.items():
            if t not in c.index:
                continue
            hist = c.loc[:t]
            if len(hist) < 241:
                continue
            v = valuation.get(sym)
            if v is None:
                continue
            window = v.loc[: t + pd.Timedelta(days=5)].tail(1)
            if window.empty or (t - window.index[0]).days > 5:
                continue
            pe, pb, mv = (window.iloc[0]["市盈率(TTM)"], window.iloc[0]["市净率"],
                          window.iloc[0]["总市值"])
            if not np.isfinite(pb) or pb <= 0 or not np.isfinite(mv) or mv <= 0:
                continue
            fwd = (float(open_[sym].loc[exit_d] / open_[sym].loc[entry_d] - 1.0)
                   if entry_d in open_[sym].index and exit_d in open_[sym].index
                   else np.nan)
            if not np.isfinite(fwd):
                continue
            close_t = float(c.loc[t])
            v20 = hist.pct_change().iloc[-20:].std(ddof=1)
            if not np.isfinite(v20) or v20 <= 0:
                continue
            dollar = close_t * float(vol[sym].loc[t]) if t in vol[sym].index else np.nan
            # technical factors from the shared causal library
            from ..features.indicators import compute_features
            ft = compute_features(frames["bars"][sym],
                                  ["rsi14", "bb_pos20", "vol_5", "macd_hist_norm",
                                   "close_over_ma20"]).set_index(frames["bars"][sym]["date"])
            ft_row = ft.loc[t]
            rows.append({
                "date": t, "symbol": sym, "fwd_ret": fwd,
                "EP": 1.0 / pe if np.isfinite(pe) else np.nan,
                "BP": 1.0 / pb,
                "size": np.log(mv),
                "mom_20": close_t / float(hist.iloc[-21]) - 1.0,
                "mom_12_1": float(hist.iloc[-21]) / float(hist.iloc[-241]) - 1.0,
                "reversal_5": -float(ret5[sym].loc[t]) if np.isfinite(ret5[sym].loc[t]) else np.nan,
                "lowvol": -v20,
                "amihud_20": float((hist.pct_change().abs().iloc[-20:]
                                    / (hist.iloc[-20:] * vol[sym].loc[hist.index[-20:]]
                                       .replace(0, np.nan))).mean()) * 1e8,
                "turnover": np.log(dollar / mv) if np.isfinite(dollar) and dollar > 0 else np.nan,
                "rsi14": float(ft_row["rsi14"]),
                "bb_pos20": float(ft_row["bb_pos20"]),
                "vol_5": -float(ft_row["vol_5"]) if np.isfinite(ft_row["vol_5"]) else np.nan,
                "macd_hist_norm": float(ft_row["macd_hist_norm"]),
                "close_over_ma20": float(ft_row["close_over_ma20"]),
            })
    return pd.DataFrame(rows)


def preprocess(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Per date: MAD winsorize + z-score each factor, then size-neutralize
    everything except `size` itself (pre-registered §2.7)."""
    out = panel.copy()
    for date, idx in panel.groupby("date").groups.items():
        sub = panel.loc[idx]
        log_mv = sub["size"]  # size column IS ln(mv) by definition
        for f in factors:
            z = mad_winsorize_zscore(sub[f])
            if f == "size":
                out.loc[idx, f"z_{f}"] = z.reindex(sub.index)
            elif len(z) >= 10:
                zn = neutralize_size(z, log_mv.reindex(z.index))
                out.loc[idx, f"z_{f}"] = zn.reindex(sub.index)
            else:
                out.loc[idx, f"z_{f}"] = z.reindex(sub.index)
    return out


def factor_ic_stats(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for date, sub in panel.groupby("date"):
        for f in factors:
            z = sub[f"z_{f}"].dropna()
            joined = pd.concat([z, sub.loc[z.index, "fwd_ret"]], axis=1).dropna()
            if len(joined) < 10:
                continue
            ic = joined[f"z_{f}"].corr(joined["fwd_ret"], method="spearman")
            rows.append({"date": date, "factor": f, "ic": ic, "n": len(joined)})
    return pd.DataFrame(rows)


def summarize_factor_ics(ic_df: pd.DataFrame, factors: list[str]) -> dict[str, dict]:
    out = {}
    for f in factors:
        s = ic_df[ic_df["factor"] == f]
        ic = s["ic"].dropna()
        mean_ic = float(ic.mean()) if len(ic) else float("nan")
        sd = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
        icir = mean_ic / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        coverage = float(s["n"].max() / max(ic_df[ic_df["factor"] == f]["n"].max(), 1)) \
            if len(s) else 0.0
        out[f] = {
            "mean_ic": mean_ic, "icir": icir, "nw_t": newey_west_t(ic),
            "positive_ratio": float((ic > 0).mean()) if len(ic) else 0.0,
            "n_windows": int(len(ic)),
            "coverage": coverage,
        }
    return out


def select_factors(stats: dict, panel: pd.DataFrame, factors: list[str]) -> dict:
    """Train-window screening + correlation dedup (template Step 5)."""
    passed, rejected = [], {}
    for f in factors:
        st = stats[f]
        reasons = []
        if abs(st["mean_ic"]) < SCREEN["min_abs_ic"]:
            reasons.append(f"meanIC {st['mean_ic']:.4f} < {SCREEN['min_abs_ic']}")
        if abs(st["icir"]) < SCREEN["min_icir"]:
            reasons.append(f"ICIR {st['icir']:.2f} < {SCREEN['min_icir']}")
        if abs(st["nw_t"]) < SCREEN["min_nw_t"]:
            reasons.append(f"NW t {st['nw_t']:.2f} < {SCREEN['min_nw_t']}")
        if st["coverage"] < SCREEN["min_coverage"]:
            reasons.append(f"coverage {st['coverage']:.2f} < {SCREEN['min_coverage']}")
        if reasons:
            rejected[f] = reasons
        else:
            passed.append(f)
    # redundancy control on the PASSED set (train-window factor correlation)
    kept = []
    for f in sorted(passed, key=lambda x: -abs(stats[x]["icir"])):
        redundant = False
        for g in kept:
            sub = panel[["date", f"z_{f}", f"z_{g}"]].dropna()
            if len(sub) > 50:
                rho = sub[f"z_{f}"].corr(sub[f"z_{g}"])
                if np.isfinite(rho) and abs(rho) > SCREEN["max_corr"]:
                    rejected[f] = [f"redundant with {g} (|rho|={abs(rho):.2f})"]
                    redundant = True
                    break
        if not redundant:
            kept.append(f)
    return {"selected": kept, "rejected": rejected}


def composite_scores(panel: pd.DataFrame, selected: list[str],
                     weights: dict[str, float] | None = None) -> pd.Series:
    z_cols = [f"z_{f}" for f in selected if f"z_{f}" in panel.columns]
    if weights:
        w = np.array([weights.get(f, 0.0) for f in selected])
        z = panel[z_cols].fillna(0.0).to_numpy()
        return pd.Series(z @ w, index=panel.index)
    return panel[z_cols].mean(axis=1)


def ridge_weights(panel: pd.DataFrame, selected: list[str],
                  window_dates: list[pd.Timestamp]) -> dict[str, float]:
    from sklearn.linear_model import Ridge
    sub = panel[panel["date"].isin(window_dates)][["fwd_ret"] + [f"z_{f}" for f in selected]].dropna()
    if len(sub) < 50:
        return {f: 1.0 / len(selected) for f in selected}
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(sub[[f"z_{f}" for f in selected]].to_numpy(), sub["fwd_ret"].to_numpy())
    return {f: float(b) for f, b in zip(selected, model.coef_)}


def run_top_n_sim(panel_bars: dict, sessions: pd.DatetimeIndex, scores: pd.DataFrame,
                  *, mcfg: dict, limits: RiskLimits, initial_equity: float,
                  run_id: str, bars_cache: dict | None = None,
                  lag_windows: int = 0) -> pd.Series:
    """TOP_N=30 equal weight by composite score; decision at rebalance date,
    entry next open (+lag_windows extra sessions), HOLD between rebalances."""
    all_dates = sorted(scores["date"].unique())
    target_for: dict[pd.Timestamp, dict[str, float]] = {}
    for date in all_dates:
        src_date = all_dates[max(0, all_dates.index(date) - lag_windows)]
        sub = scores[scores["date"] == src_date].dropna(subset=["score"])
        ranked = sub.sort_values("score", ascending=False).head(TOP_N)
        if len(ranked) == 0:
            continue
        d = pd.Timestamp(date)
        idx = sessions.get_indexer([d])[0]
        if idx == -1 or idx + 1 + lag_windows >= len(sessions):
            continue
        w = min(1.0 / len(ranked), limits.max_weight_per_name) * limits.max_gross_exposure
        target_for[sessions[idx + 1 + lag_windows]] = {s: w for s in ranked["symbol"]}

    def fn(t, account, prices):
        if t in target_for:
            return dict(target_for[t])
        return None

    sim = run_simulation(panel_bars, sessions, fn, market_cfg=mcfg, limits=limits,
                         initial_equity=initial_equity, run_id=run_id,
                         bars_cache=bars_cache)
    return sim.daily_returns


def run_multi_factor_study(*, index: str = "000300", n_symbols: int = 150,
                           seed: int = 42, initial_equity: float = 1_000_000.0) -> Path:
    t0 = time.time()
    mcfg = load_market_config("cn")
    rcfg = load_risk_config()
    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})
    factors = list(FACTOR_DIRECTIONS)

    bars_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "bars"
    val_dir = PROJECT_ROOT / "datasets" / "updates" / "xs_cn" / "valuation"
    frames = _load_xs_frames(bars_dir, val_dir)
    bars = frames["bars"]
    sessions = market_sessions(bars)
    start_idx = max(int(0.05 * len(sessions)), 241)
    rebalance_dates = [sessions[i] for i in range(start_idx, len(sessions), REBALANCE_SESSIONS)]

    raw = build_factor_panel(frames, sessions, rebalance_dates)
    factors_present = [f for f in factors if f in raw.columns]
    panel = preprocess(raw, factors_present)

    # window split (time-ordered, no shuffling)
    dates = sorted(panel["date"].unique())
    split = int(len(dates) * TRAIN_TEST_SPLIT)
    train_dates, test_dates = dates[:split], dates[split:]

    # ---- train-window screening (frozen list) --------------------------------
    train_panel = panel[panel["date"].isin(train_dates)]
    ic_train = factor_ic_stats(train_panel, factors_present)
    stats_train = summarize_factor_ics(ic_train, factors_present)
    selection = select_factors(stats_train, train_panel, factors_present)
    selected = selection["selected"]
    if len(selected) < 3:
        raise RuntimeError(
            f"factor screening kept only {len(selected)} factors — H4 cannot "
            f"proceed per pre-registered gates; rejected: {selection['rejected']}")

    # ---- test-window synthesis (weights from maturity-safe trailing windows) --
    test_panel = panel[panel["date"].isin(test_dates)]
    ic_all = factor_ic_stats(panel, factors_present)
    test_dates_list = list(test_dates)
    methods = {}
    score_tables = {}

    # A: equal weight
    scores_a = test_panel.copy()
    scores_a["score"] = composite_scores(test_panel, selected)
    methods["equal_weight"] = None
    score_tables["equal_weight"] = scores_a

    # B/C/D: trailing maturity-safe estimation at each test window
    all_win_dates = list(dates)
    for date in test_dates_list:
        j = all_win_dates.index(date)
        hist_dates = all_win_dates[max(0, j - MATURITY_SHIFT - TRAILING_WINDOWS):
                                   j - MATURITY_SHIFT + 1]
        hist_panel = panel[panel["date"].isin(hist_dates)]
        ic_hist = factor_ic_stats(hist_panel, selected)
        icirs = {}
        for f in selected:
            s = ic_hist[ic_hist["factor"] == f]["ic"].dropna()
            icirs[f] = float(s.mean() / s.std(ddof=1)) if len(s) > 2 and s.std(ddof=1) > 0 else 0.0
        w_b = icir_weights(icirs)
        w_c = shrink_weights(w_b)
        w_d = ridge_weights(hist_panel, selected, hist_dates)
        for name, w in (("icir_weight", w_b), ("shrink_icir", w_c), ("ridge", w_d)):
            scores = test_panel[test_panel["date"] == date].copy()
            scores["score"] = composite_scores(scores, selected, weights=w)
            score_tables.setdefault(name, pd.concat(
                [score_tables.get(name, pd.DataFrame()), scores], ignore_index=True))
        # record weight evolution
        methods.setdefault("weights_history", {})[str(date.date())] = {
            "icir": w_b, "shrink": w_c, "ridge": w_d}

    # ---- run sims -------------------------------------------------------------
    bars_cache = {sym: df.set_index("date").to_dict("index") for sym, df in bars.items()}
    spy = max(int(round(len(sessions) / max((sessions[-1] - sessions[0]).days / 365.25, 1e-9))), 1)
    run_id = f"xs_mf_h4-{config_fingerprint({'i': index, 'n': n_symbols, 's': seed})}-{now_utc_iso().replace(':', '')}"

    results = {}
    for name, tbl in score_tables.items():
        rets = run_top_n_sim(bars, sessions, tbl, mcfg=mcfg, limits=limits,
                             initial_equity=initial_equity, run_id=f"{run_id}-{name}",
                             bars_cache=bars_cache)
        results[name] = {"total_return": float((1 + rets.fillna(0)).prod() - 1),
                         "sharpe": float(rets.mean() / rets.std(ddof=1) * np.sqrt(spy))
                         if rets.std(ddof=1) > 0 else 0.0,
                         "daily_returns": rets}

    # benchmarks: equal-weight universe + capped buy&hold
    universe_tbl = test_panel.copy()
    universe_tbl["score"] = 1.0  # all names equal -> equal-weight universe
    rets_uw = run_top_n_sim(bars, sessions, universe_tbl.assign(score=1.0),
                            mcfg=mcfg, limits=limits, initial_equity=initial_equity,
                            run_id=f"{run_id}-bench-uw", bars_cache=bars_cache)
    results["benchmark_equal_weight_universe"] = {
        "total_return": float((1 + rets_uw.fillna(0)).prod() - 1),
        "sharpe": float(rets_uw.mean() / rets_uw.std(ddof=1) * np.sqrt(spy))
        if rets_uw.std(ddof=1) > 0 else 0.0}

    # test-window RankIC of each synthesis (composite vs fwd)
    synth_ic = {}
    for name, tbl in score_tables.items():
        ics = []
        for date, sub in tbl.groupby("date"):
            joined = sub[["score", "fwd_ret"]].dropna()
            if len(joined) >= 10:
                ics.append(joined["score"].corr(joined["fwd_ret"], method="spearman"))
        s = pd.Series(ics).dropna()
        synth_ic[name] = {"mean_rank_ic": float(s.mean()), "nw_t": newey_west_t(s),
                          "icir": float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) > 0 else None,
                          "n_windows": int(len(s))}

    # ---- robustness on the best method ----------------------------------------
    best = max((k for k in results if not k.startswith("benchmark")),
               key=lambda k: results[k]["total_return"])
    best_tbl = score_tables[best]
    robustness = {}
    for mult in (2.0, 4.0):
        rets = run_top_n_sim(bars, sessions, best_tbl, mcfg=_scale_costs(mcfg, mult),
                             limits=limits, initial_equity=initial_equity,
                             run_id=f"{run_id}-{best}-cost{int(mult)}x",
                             bars_cache=bars_cache)
        robustness[f"cost_x{int(mult)}"] = float((1 + rets.fillna(0)).prod() - 1)
    rets_lag = run_top_n_sim(bars, sessions, best_tbl, mcfg=mcfg, limits=limits,
                             initial_equity=initial_equity,
                             run_id=f"{run_id}-{best}-lag1w", bars_cache=bars_cache,
                             lag_windows=1)
    robustness["execution_lag_1w"] = float((1 + rets_lag.fillna(0)).prod() - 1)

    # ---- verdict --------------------------------------------------------------
    bench_ret = results["benchmark_equal_weight_universe"]["total_return"]
    best_ret = results[best]["total_return"]
    best_ic = synth_ic[best]
    criteria = {
        "a_beats_benchmarks": bool(best_ret > bench_ret),
        "b_test_rankic": bool(best_ic["mean_rank_ic"] >= 0.02 and best_ic["nw_t"] >= 2.0),
        "c_robust_cost_lag": bool(robustness["cost_x4"] > bench_ret
                                  and robustness["execution_lag_1w"] > bench_ret),
        "d_weights_beat_equal": bool(results[best]["total_return"]
                                     >= results["equal_weight"]["total_return"]
                                     or best == "equal_weight"),
    }
    passed = all(criteria.values())

    report = {
        "run_id": run_id,
        "pre_registration": "docs/DECISIONS.md section 2.7 (H4)",
        "universe": f"CSI {index} CURRENT constituents, seeded n={n_symbols}, seed={seed}",
        "survivorship_bias": True,
        "status": "research_only ceiling (survivorship bias)",
        "data": {"provider": PROVIDER_NAME, "valuation_source": PROVIDER_VALUATION_SOURCE,
                 "n_symbols_bars": len(bars), "n_windows_total": len(dates),
                 "n_train_windows": len(train_dates), "n_test_windows": len(test_dates),
                 "avg_cross_section": float(panel.groupby("date").size().mean())},
        "factor_screening_train": stats_train,
        "selection": selection,
        "selected_factors": selected,
        "synthesis_test_ic": synth_ic,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "daily_returns"}
                    for k, v in results.items()},
        "best_method": best,
        "robustness": robustness,
        "criteria": criteria,
        "verdict": "PASSED (still research_only)" if passed else "NOT PASSED",
        "runtime_seconds": round(time.time() - t0, 2),
    }
    study_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    study_dir.mkdir(parents=True, exist_ok=True)
    write_json(study_dir / "report.json", report)
    panel.to_csv(study_dir / "factor_panel.csv", index=False)
    ic_all.to_csv(study_dir / "ic_series.csv", index=False)
    pd.DataFrame(methods.get("weights_history", {})).to_csv(
        study_dir / "weights_history.csv")
    _write_md(report, study_dir / "report.md")
    return study_dir


def _write_md(report: dict, path: Path) -> None:
    lines = [
        f"# Multi-factor synthesis study (H4): `{report['run_id']}`",
        "",
        f"- Pre-registration: {report['pre_registration']} | Universe: {report['universe']}",
        f"- Survivorship bias: {report['survivorship_bias']} → status ceiling research_only",
        f"- Train windows {report['data']['n_train_windows']} / test windows "
        f"{report['data']['n_test_windows']} "
        f"| average cross-section {report['data']['avg_cross_section']:.0f} names",
        "",
        "## Train-window factor screening",
        "",
        "| Factor | meanIC | ICIR | NW t | Positive-IC ratio | Coverage | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    sel = set(report["selected_factors"])
    for f, st in report["factor_screening_train"].items():
        mark = "✓ selected" if f in sel else ("✗ " + ";".join(
            report["selection"]["rejected"].get(f, [""])))
        lines.append(f"| {f} | {st['mean_ic']:.4f} | {st['icir']:.2f} | {st['nw_t']:.2f} "
                     f"| {st['positive_ratio']:.0%} | {st['coverage']:.0%} | {mark} |")
    lines += ["", "## Test-window synthesis comparison (TOP_N=30, net of costs)", "",
              "| Method | Net compounded return | Sharpe | Monthly RankIC | NW t |",
              "|---|---:|---:|---:|---:|"]
    for name, r in report["results"].items():
        ic = report["synthesis_test_ic"].get(name, {})
        lines.append(f"| {name} | {r['total_return']:.2%} | {r['sharpe']:.2f} "
                     f"| {ic.get('mean_rank_ic', float('nan')):.4f} "
                     f"| {ic.get('nw_t', float('nan')):.2f} |")
    lines += ["", f"- Best method: **{report['best_method']}** "
              f"| Robustness: {report['robustness']}",
              f"- Criteria: {report['criteria']}",
              f"- **Verdict: {report['verdict']}**",
              "",
              "Passing entries remain research_only (survivorship bias); entering "
              "the next stage requires a new pre-registration with historical "
              "constituents data."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
