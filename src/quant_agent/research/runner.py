"""Research runner: nested walk-forward evaluation with honest accounting.

Protocol (handover 8.5):
- Fold fitting uses ONLY labels matured before the fit cutoff.
- Per OOS window: first half tunes the policy threshold via mini-simulations,
  second half is reported. During a fold's tune window the final simulation
  runs the PREVIOUS fold's threshold (activation schedule), so no reported
  date is ever used for its own selection.
- Every OOS (date, symbol) prediction exists exactly once.
- Strategy, cash, buy-and-hold, and a trend rule run through the SAME
  simulator with the SAME costs.
- Results are reported with calibration, block-bootstrap uncertainty, and a
  health record; a fail blocks any complete-run claim.

Distributed execution (GPU swarm readiness): the EXPENSIVE phase is the
per-fold fitting, which is embarrassingly parallel because folds are disjoint
by construction. `run_research(..., fit_shard="i/n")` fits only the folds
assigned to shard i (fold_id % n == i) and writes shard outputs into a SHARED
run directory (pass an explicit run_id so every worker targets the same
directory). `finalize_research(...)` merges the shard outputs (verifying
disjoint fold coverage), then runs the sequential evaluation locally. Fold
fitting on a GPU worker: set model.gbm.device = "cuda" in the strategy
config (models.py falls back to CPU automatically when CUDA is unusable).
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..agent.report import build_markdown_report
from ..alpha.expected_return import PayoffModel, net_edge, round_trip_cost_bps
from ..config import (PROJECT_ROOT, load_market_config, load_risk_config,
                      load_strategy_config)
from ..data.ingest import ingest_market, load_normalized
from ..data.calendar import market_sessions
from ..execution.simulator import run_simulation
from ..features.pipeline import build_panel
from ..labels.forward import LabelSpec, add_labels, training_eligible
from ..monitoring.health import run_health_checks
from ..portfolio.optimizer import PortfolioConfig, target_weights
from ..registry.store import new_run_dir, save_bundle
from ..research.metrics import (block_bootstrap_sharpe_ci, calibration,
                                completed_trade_stats, reliability_bins,
                                summarize)
from ..research.models import fit_predict_proba, make_model
from ..research.splits import build_folds
from ..research.trials import TrialLedger
from ..risk.checks import RiskLimits
from ..util import config_fingerprint, now_utc_iso, read_json, sha256_file, write_json

DEFAULT_THRESHOLD = 0.55


def _sessions_per_year(sessions: pd.DatetimeIndex) -> int:
    if len(sessions) < 2:
        return 252
    years = (sessions[-1] - sessions[0]).days / 365.25
    return max(int(round(len(sessions) / max(years, 1e-9))), 1)


def _mini_sim_return(panel: dict, sessions: pd.DatetimeIndex, pred_df: pd.DataFrame,
                     threshold: float, pcfg: PortfolioConfig, mcfg: dict,
                     limits: RiskLimits, initial_equity: float, run_id: str) -> float:
    def fn(t, account, prices):
        rows = pred_df[(pred_df["date"] == t) & pred_df["p_up"].notna()]
        rows = rows[rows["p_up"] >= threshold]
        if rows.empty:
            return {}
        res = target_weights(rows.set_index("symbol")["mu_net"],
                             rows.set_index("symbol")["vol"], pcfg)
        return {k: float(v) for k, v in res.weights.items() if v > 0}

    sim = run_simulation(panel, sessions, fn, market_cfg=mcfg, limits=limits,
                         initial_equity=initial_equity, run_id=f"{run_id}-tune")
    if len(sim.equity_curve) == 0:
        return 0.0
    return float(sim.equity_curve.iloc[-1] / initial_equity - 1.0)


# ------------------------------------------------------------------ context
def _prepare_context(strategy: str, market: str | None,
                     initial_equity: float) -> dict:
    """Shared preparation for fitting and finalization: configs, data gate,
    labeled panel, sessions, folds, policy objects."""
    scfg = load_strategy_config(strategy)
    market = market or scfg["market"]
    mcfg = load_market_config(market)
    rcfg = load_risk_config()

    manifest_path = PROJECT_ROOT / "datasets" / f"manifest_{market}.json"
    if not manifest_path.exists():
        ingest_market(market)
    manifest = read_json(manifest_path)
    unusable = [s for s, e in manifest["symbols"].items() if not e.get("usable")]
    if unusable:
        raise RuntimeError(
            f"complete-run gate blocked: unusable symbols {unusable}; resolve "
            "quarantines or update data, then re-run `qa ingest`.")

    frames = load_normalized(market, mcfg["universe"])
    panel = build_panel(frames, scfg["features"])
    for sym, df in panel.items():
        df["symbol"] = sym
    spec = LabelSpec(**scfg["label"])
    panel = {sym: add_labels(df, spec) for sym, df in panel.items()}
    sessions = market_sessions(frames)
    folds = build_folds(
        sessions,
        initial_train_fraction=scfg["walk_forward"]["initial_train_fraction"],
        test_sessions=scfg["walk_forward"]["test_sessions"],
        step_sessions=scfg["walk_forward"]["step_sessions"],
        policy_tune_fraction=scfg["walk_forward"]["policy_tune_fraction"],
    )
    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})
    return {
        "scfg": scfg, "mcfg": mcfg, "rcfg": rcfg, "manifest": manifest,
        "frames": frames, "panel": panel, "sessions": sessions, "folds": folds,
        "limits": limits, "spy": _sessions_per_year(sessions),
        "feat_cols": list(scfg["features"]),
        "cost_bps": round_trip_cost_bps(mcfg),
        "shrink": float(scfg["portfolio"]["shrink_conditional_returns"]),
        "seed": int(scfg["seed"]),
        "initial_equity": initial_equity,
        "primary_kind": scfg["model"]["primary"],
        "pcfg": PortfolioConfig(
            edge_threshold=scfg["portfolio"]["edge_threshold"],
            lambda_vol=scfg["portfolio"]["lambda_vol"],
            max_names=scfg["portfolio"]["max_names"],
            max_weight_per_name=rcfg["max_weight_per_name"],
            max_gross_exposure=rcfg["max_gross_exposure"],
        ),
    }


def _fit_assigned_folds(ctx: dict, assigned_folds: list, run_dir: Path,
                        trials: TrialLedger, initial_equity: float) -> tuple[
        pd.DataFrame, dict[int, float], list[dict]]:
    """Fit + predict + policy-tune for the ASSIGNED folds only (shard unit).
    Returns (predictions, chosen_thresholds, fold_records)."""
    panel, feat_cols = ctx["panel"], ctx["feat_cols"]
    mcfg, pcfg, limits = ctx["mcfg"], ctx["pcfg"], ctx["limits"]
    seed, shrink, cost_bps = ctx["seed"], ctx["shrink"], ctx["cost_bps"]
    threshold_grid = [round(x, 2) for x in np.arange(0.40, 0.71, 0.05)]
    pred_rows: list[dict] = []
    chosen_thresholds: dict[int, float] = {}
    fold_records = []

    for fold in assigned_folds:
        train_parts, pred_parts = [], []
        for sym, df in panel.items():
            elig = df.loc[training_eligible(df, fit_cutoff=fold.fit_cutoff)]
            train_parts.append(elig.dropna(subset=feat_cols))
            window = df[(df["date"] >= fold.predict_start) & (df["date"] <= fold.predict_end)]
            pred_parts.append(window.dropna(subset=feat_cols))
        train_df = pd.concat(train_parts, ignore_index=True)
        pred_df = pd.concat(pred_parts, ignore_index=True)
        if train_df.empty or pred_df.empty:
            trials.log("fold_skipped", fold_id=fold.fold_id,
                       n_train=len(train_df), n_pred=len(pred_df))
            continue
        X_tr = train_df[feat_cols].to_numpy(dtype=float)
        y_tr = (train_df["label"].to_numpy(dtype=float) > 0).astype(int)
        X_pr = pred_df[feat_cols].to_numpy(dtype=float)

        payoff = PayoffModel.fit(train_df, shrink=shrink)
        trials.log("fit", fold_id=fold.fold_id, n_train_rows=len(train_df),
                   base_rate=float(y_tr.mean()), fit_cutoff=str(fold.fit_cutoff.date()))

        fold_pred_frames = {}
        for kind in (ctx["primary_kind"], ctx["scfg"]["model"]["challenger"]):
            params = ctx["scfg"]["model"].get(kind, {})
            model = make_model(kind, params, seed)
            p = fit_predict_proba(model, X_tr, y_tr, X_pr)
            pdf = pred_df[["date", "symbol", "vol_20"]].copy()
            pdf["model"] = kind
            pdf["fold_id"] = fold.fold_id
            pdf["p_up"] = p
            pdf["vol"] = pdf["vol_20"].astype(float)
            pdf = pdf.drop(columns=["vol_20"])
            # guard: sizing needs a positive finite volatility estimate
            pdf = pdf[pdf["vol"].notna() & (pdf["vol"] > 0)]
            if pdf["p_up"].notna().any():
                pdf["e_event"] = [payoff.e_event.get(s, payoff.global_event) for s in pdf["symbol"]]
                pdf["e_complement"] = [payoff.e_complement.get(s, payoff.global_complement) for s in pdf["symbol"]]
                pdf["mu_gross"] = pdf["p_up"] * pdf["e_event"] + (1 - pdf["p_up"]) * pdf["e_complement"]
                pdf["cost_bps"] = cost_bps
                pdf["mu_net"] = [net_edge(m, cost_bps) if np.isfinite(p_) else np.nan
                                 for m, p_ in zip(pdf["mu_gross"], pdf["p_up"])]
                pdf["window"] = np.where(pdf["date"] <= fold.tune_end, "tune", "report")
                fold_pred_frames[kind] = pdf
                pred_rows.append(pdf)
            else:
                trials.log("degenerate_fold", fold_id=fold.fold_id, model=kind,
                           note="single class in training window; no predictions")

        tune_pdf = fold_pred_frames.get(ctx["primary_kind"])
        best_thr, best_score = DEFAULT_THRESHOLD, -np.inf
        if tune_pdf is not None and len(tune_pdf[tune_pdf["window"] == "tune"]):
            tune_sessions = ctx["sessions"][(ctx["sessions"] >= fold.tune_start)
                                            & (ctx["sessions"] <= fold.tune_end)]
            for thr in threshold_grid:
                score = _mini_sim_return(panel, tune_sessions, tune_pdf, thr, pcfg,
                                         mcfg, limits, initial_equity, trials.run_id)
                trials.log("policy_tune", fold_id=fold.fold_id, threshold=thr,
                           tune_net_return=score)
                if score > best_score + 1e-12:
                    best_thr, best_score = thr, score
        chosen_thresholds[fold.fold_id] = best_thr
        trials.log("policy_selected", fold_id=fold.fold_id, threshold=best_thr,
                   tune_net_return=best_score if np.isfinite(best_score) else None)
        fold_records.append({**fold.to_dict(), "chosen_threshold": best_thr,
                             "n_train_rows": int(len(train_df)),
                             "n_pred_rows": int(len(pred_df))})

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(
        columns=["date", "symbol", "model", "fold_id", "p_up", "mu_net", "vol", "window"])
    return predictions, chosen_thresholds, fold_records


def _finalize(ctx: dict, run_id: str, run_dir: Path, predictions: pd.DataFrame,
              chosen_thresholds: dict[int, float], fold_records: list[dict],
              trials: TrialLedger, initial_equity: float,
              model_sha256: str | None = None) -> None:
    """Sequential evaluation phase (runs once, locally): activation schedule,
    continuous simulations, benchmarks, metrics, calibration, health,
    artifacts, and bundle registration."""
    panel = ctx["panel"]
    sessions = ctx["sessions"]
    mcfg, rcfg, scfg = ctx["mcfg"], ctx["rcfg"], ctx["scfg"]
    limits, pcfg = ctx["limits"], ctx["pcfg"]
    primary_kind = ctx["primary_kind"]
    name_cap = rcfg["max_weight_per_name"]

    fold_for_date: dict[pd.Timestamp, object] = {}
    for fold in ctx["folds"]:
        for t in sessions[(sessions >= fold.predict_start) & (sessions <= fold.predict_end)]:
            fold_for_date[t] = fold

    primary_pred = predictions[predictions["model"] == primary_kind] if len(predictions) else predictions

    def active_threshold(t) -> float:
        fold = fold_for_date.get(t)
        if fold is None:
            return DEFAULT_THRESHOLD
        if t <= fold.tune_end:
            # Tune windows run the PREVIOUS fold's threshold or the
            # pre-registered default — never this fold's own selection
            # (review doc P0: first-fold threshold pollution fix).
            return chosen_thresholds.get(fold.fold_id - 1, DEFAULT_THRESHOLD)
        return chosen_thresholds.get(fold.fold_id, DEFAULT_THRESHOLD)

    weight_log: list[dict] = []

    def strategy_fn(t, account, prices):
        fold = fold_for_date.get(t)
        if fold is None:
            return {}
        rows = primary_pred[(primary_pred["date"] == t) & primary_pred["p_up"].notna()]
        if rows.empty:
            return {}
        thr = active_threshold(t)
        rows = rows[rows["p_up"] >= thr] if thr is not None else rows
        if rows.empty:
            return {}
        res = target_weights(rows.set_index("symbol")["mu_net"],
                             rows.set_index("symbol")["vol"], pcfg)
        for sym, w in res.weights.items():
            if w > 0:
                weight_log.append({"date": str(t.date()), "symbol": sym,
                                   "weight": float(w), "threshold": thr})
        return {k: float(v) for k, v in res.weights.items() if v > 0}

    sessions_oos = sessions[(sessions >= ctx["folds"][0].predict_start)]

    def buyhold_fn(t, account, prices):
        if t == sessions_oos[0]:
            n = len(mcfg["universe"])
            # same single-name cap as the strategy: equal weight, capped,
            # remainder stays cash (feasible by construction)
            return {s: min(1.0 / n, name_cap) for s in mcfg["universe"]}
        return None  # HOLD: buy once, then never rebalance or liquidate

    def trend_fn(t, account, prices):
        w = {}
        for sym, df in panel.items():
            row = df.loc[df["date"] == t]
            if len(row) and np.isfinite(row["ret_20"].iloc[0]) and row["ret_20"].iloc[0] > 0:
                w[sym] = 1.0
        if not w:
            return {}
        s = sum(w.values())
        return {k: min(v / s, name_cap) for k, v in w.items()}

    from ..alpha.factor_fallback import FALLBACK_DEFAULT_BUDGET, fallback_rule_weights

    def fallback_fn(t, account, prices):
        # OFFLINE STUDY ONLY: this rule is UNVALIDATED design delivery
        # (alpha/factor_fallback.py). It is evaluated here for comparison and
        # stress-testing; it never gates live execution in this state.
        factor_rows = {}
        for sym, df in panel.items():
            row = df.loc[df["date"] == t]
            if not len(row):
                continue

            def _f(col):
                v = row[col].iloc[0]
                return float(v) if pd.notna(v) else None

            factor_rows[sym] = {"ret_20": _f("ret_20"),
                                "close_over_ma60": _f("close_over_ma60"),
                                "vol_20": _f("vol_20")}
        w, _ = fallback_rule_weights(
            factor_rows, budget=FALLBACK_DEFAULT_BUDGET,
            max_weight_per_name=min(0.10, name_cap))
        return w  # {} -> flat (cash is a legitimate output)

    sims = {}
    sims["strategy"] = run_simulation(panel, sessions_oos, strategy_fn, market_cfg=mcfg,
                                      limits=limits, initial_equity=initial_equity, run_id=run_id)
    sims["equal_weight_buyhold"] = run_simulation(panel, sessions_oos, buyhold_fn, market_cfg=mcfg,
                                                  limits=limits, initial_equity=initial_equity,
                                                  run_id=f"{run_id}-bh")
    sims["trend_rule"] = run_simulation(panel, sessions_oos, trend_fn, market_cfg=mcfg,
                                        limits=limits, initial_equity=initial_equity,
                                        run_id=f"{run_id}-trend")
    sims["factor_fallback_rule_study"] = run_simulation(
        panel, sessions_oos, fallback_fn, market_cfg=mcfg, limits=limits,
        initial_equity=initial_equity, run_id=f"{run_id}-fb")

    # ---- evaluation: metrics, calibration, uncertainty ---------------------
    realized = []
    for sym, df in panel.items():
        realized.append(df[["date", "symbol", "label", "label_maturity_date"]])
    realized = pd.concat(realized, ignore_index=True)
    if len(predictions):
        predictions = predictions.merge(
            realized.rename(columns={"label": "label_realized"}),
            on=["date", "symbol"], how="left")
        matured = predictions["label_maturity_date"] <= sessions[-1]
        predictions.loc[~matured, "label_realized"] = np.nan

    strat_summ = summarize(
        sims["strategy"].daily_returns, sessions_per_year=ctx["spy"],
        turnover=float(sims["strategy"].turnover.sum()),
        costs_paid=float(sims["strategy"].costs.iloc[-1]),
        exposure=float(sims["strategy"].exposure.mean()),
        n_trades=len(sims["strategy"].fills))
    strat_summ.update(completed_trade_stats(sims["strategy"].fills))

    # report-window-only summary: the same metrics restricted to sessions in
    # fold REPORT windows (policy selected strictly before each report window;
    # tune windows run previous-fold/default thresholds). Both views reported.
    report_sessions = None
    for fold in ctx["folds"]:
        m = sessions[(sessions >= fold.report_start) & (sessions <= fold.report_end)]
        report_sessions = m if report_sessions is None else report_sessions.union(m)

    def _window_summary(sim):
        dr = sim.daily_returns
        dr_rep = dr[dr.index.isin(report_sessions)] if report_sessions is not None else dr
        return summarize(dr_rep, sessions_per_year=ctx["spy"],
                         turnover=float(sim.turnover[sim.turnover.index.isin(report_sessions)].sum()),
                         costs_paid=float(sim.costs.iloc[-1]),
                         exposure=float(sim.exposure[sim.exposure.index.isin(report_sessions)].mean()),
                         n_trades=int(sim.fills["session"].isin(
                             [str(d.date()) for d in report_sessions]).sum()) if len(sim.fills) else 0)

    strat_report_summ = _window_summary(sims["strategy"])
    strat_report_summ.update(completed_trade_stats(
        sims["strategy"].fills[sims["strategy"].fills["order_type"] != "STOP"]
        if len(sims["strategy"].fills) else sims["strategy"].fills))
    bench_summ = {}
    for name, sim in sims.items():
        if name == "strategy":
            continue
        bench_summ[name] = summarize(sim.daily_returns, sessions_per_year=ctx["spy"],
                                     turnover=float(sim.turnover.sum()),
                                     costs_paid=float(sim.costs.iloc[-1]),
                                     exposure=float(sim.exposure.mean()),
                                     n_trades=len(sim.fills))
    bench_report_summ = {name: _window_summary(sim) for name, sim in sims.items()
                         if name != "strategy"}

    cal, bins = {}, []
    if len(predictions):
        ev = predictions[(predictions["model"] == primary_kind)
                         & predictions["p_up"].notna()
                         & predictions["label_realized"].notna()]
        if len(ev):
            y = (ev["label_realized"].to_numpy(dtype=float) > 0).astype(int)
            cal = calibration(y, ev["p_up"].to_numpy(dtype=float))
            bins = reliability_bins(y, ev["p_up"].to_numpy(dtype=float))
    boot = block_bootstrap_sharpe_ci(sims["strategy"].daily_returns,
                                     sessions_per_year=ctx["spy"],
                                     seed=int(scfg["seed"]))

    health = run_health_checks(manifest=ctx["manifest"], predictions=predictions,
                               sim_result=sims["strategy"],
                               initial_equity=initial_equity,
                               limits_max_weight=rcfg["max_weight_per_name"],
                               as_of=now_utc_iso()[:10])

    metrics = {
        "sessions_per_year": ctx["spy"],
        "oos_span": [str(sessions_oos[0].date()), str(sessions_oos[-1].date())],
        "strategy": strat_summ,
        "strategy_report_windows_only": strat_report_summ,
        "benchmarks": bench_summ,
        "benchmarks_report_windows_only": bench_report_summ,
        "calibration": cal,
        "reliability_bins": bins,
        "bootstrap": boot,
        "chosen_thresholds": {str(k): v for k, v in chosen_thresholds.items()},
        "runtime_seconds": 0.0,
    }

    # ---- persist artifacts --------------------------------------------------
    import sys as _sys
    import importlib.metadata as _md
    env_fingerprint = {
        "python": _sys.version.split()[0],
        "packages": {p: _md.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "xgboost")},
    }
    config_blob = {
        "run_id": run_id, "strategy": scfg["strategy"], "market": mcfg["market"],
        "created_utc": now_utc_iso(),
        "config_fingerprint": config_fingerprint({"s": scfg, "m": mcfg, "r": rcfg}),
        "strategy_config": scfg, "market_config": mcfg, "risk_config": rcfg,
        "dataset_source_sha256": {s: e.get("source_sha256")
                                  for s, e in ctx["manifest"]["symbols"].items()},
        "initial_equity": initial_equity,
        "model_sha256": model_sha256,
        "model_kind": primary_kind,
        "env": env_fingerprint,
    }
    write_json(run_dir / "config.json", config_blob)
    write_json(run_dir / "folds.json", {"folds": fold_records,
                                        "activation": "tune-window uses previous fold's threshold"})
    predictions.to_csv(run_dir / "oos_predictions.csv", index=False)
    pd.DataFrame(weight_log).to_csv(run_dir / "target_weights.csv", index=False)
    eq = pd.DataFrame({"strategy": sims["strategy"].equity_curve,
                       **{k: v.equity_curve for k, v in sims.items() if k != "strategy"}})
    eq.index.name = "date"
    eq.to_csv(run_dir / "equity_curve.csv")
    sims["strategy"].fills.to_csv(run_dir / "fills.csv", index=False)
    sims["strategy"].orders.to_csv(run_dir / "orders.csv", index=False)
    for ev_ in sims["strategy"].account.events:
        trials.log("ledger_event", **ev_)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "health.json", health)
    report_md = build_markdown_report(run_id)
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")
    _plot_equity(eq, run_dir / "equity.png", run_id)
    save_bundle(run_dir, status="candidate",
                note="auto-registered on research completion")


# ------------------------------------------------------------------- shards
def _shard_paths(run_dir: Path, shard: int, n_shards: int) -> dict:
    return {
        "predictions": run_dir / f"oos_predictions_shard{shard}of{n_shards}.csv",
        "thresholds": run_dir / f"chosen_thresholds_shard{shard}of{n_shards}.json",
        "folds": run_dir / f"fold_records_shard{shard}of{n_shards}.json",
    }


# ------------------------------------------------------------------- public
def run_research(strategy: str, *, market: str | None = None,
                 initial_equity: float = 1_000_000.0,
                 fit_shard: str | None = None,
                 run_id: str | None = None) -> str:
    """Fit + evaluate a strategy. Two modes:

    - Single mode (default): fit ALL folds and finalize in one call.
    - Shard mode (`fit_shard="i/n"`): fit only folds with fold_id % n == i and
      write shard outputs into the SHARED run directory named by the required
      `run_id`. After every shard has run, call `finalize_research` with the
      same run_id to merge and evaluate. Designed for distributing fold
      fitting across workers (a GPU swarm: set model.gbm.device = "cuda").
    """
    t0 = time.time()
    ctx = _prepare_context(strategy, market, initial_equity)
    market = ctx["mcfg"]["market"]

    if fit_shard is not None:
        shard_text = str(fit_shard)
        if "/" not in shard_text:
            raise ValueError('fit_shard must be "i/n" (e.g. "0/2")')
        shard, n_shards = (int(x) for x in shard_text.split("/"))
        if not (0 <= shard < n_shards) or n_shards < 1:
            raise ValueError(f"invalid shard spec: {fit_shard}")
        if run_id is None:
            raise ValueError("shard mode requires an explicit --run-id shared "
                             "by every worker")
        run_dir = new_run_dir(strategy, market, run_id)
        trials = TrialLedger(run_dir, run_id)
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            write_json(cfg_path, {
                "run_id": run_id, "strategy": ctx["scfg"]["strategy"],
                "market": market, "created_utc": now_utc_iso(),
                "config_fingerprint": config_fingerprint(
                    {"s": ctx["scfg"], "m": ctx["mcfg"], "r": ctx["rcfg"]}),
                "strategy_config": ctx["scfg"], "market_config": ctx["mcfg"],
                "risk_config": ctx["rcfg"],
                "dataset_source_sha256": {s: e.get("source_sha256")
                                          for s, e in ctx["manifest"]["symbols"].items()},
                "initial_equity": initial_equity,
                "shard_plan": f"{n_shards} shards",
            })
        trials.log("config", strategy_fingerprint=config_fingerprint(ctx["scfg"]),
                   market_fingerprint=config_fingerprint(ctx["mcfg"]),
                   risk_fingerprint=config_fingerprint(ctx["rcfg"]),
                   dataset_hashes={s: e.get("source_sha256")
                                   for s, e in ctx["manifest"]["symbols"].items()},
                   shard=f"{shard}/{n_shards}")
        assigned = [f for f in ctx["folds"] if f.fold_id % n_shards == shard]
        for f in assigned:
            trials.log("fold_defined", **f.to_dict())
        predictions, chosen_thresholds, fold_records = _fit_assigned_folds(
            ctx, assigned, run_dir, trials, initial_equity)
        paths = _shard_paths(run_dir, shard, n_shards)
        predictions.to_csv(paths["predictions"], index=False)
        write_json(paths["thresholds"], {str(k): v for k, v in chosen_thresholds.items()})
        write_json(paths["folds"], fold_records)
        if shard == 0:
            # deterministic registration artifact: identical on every shard;
            # shard 0 materializes it once (F-002 frozen model contract)
            _write_model_artifact(ctx, run_dir, trials)
        trials.log("shard_done", shard=shard, n_shards=n_shards,
                   folds_assigned=len(assigned),
                   predictions=int(len(predictions)),
                   elapsed_seconds=round(time.time() - t0, 2))
        return run_id

    # single mode: fit all folds, then finalize in-process
    fp = config_fingerprint({"s": ctx["scfg"], "m": ctx["mcfg"]})
    run_id = f"{strategy}-{market}-{fp}-{now_utc_iso().replace(':', '')}"
    run_dir = new_run_dir(strategy, market, run_id)
    trials = TrialLedger(run_dir, run_id)
    trials.log("config", strategy_fingerprint=config_fingerprint(ctx["scfg"]),
               market_fingerprint=config_fingerprint(ctx["mcfg"]),
               risk_fingerprint=config_fingerprint(ctx["rcfg"]),
               dataset_hashes={s: e.get("source_sha256")
                               for s, e in ctx["manifest"]["symbols"].items()})
    for f in ctx["folds"]:
        trials.log("fold_defined", **f.to_dict())
    predictions, chosen_thresholds, fold_records = _fit_assigned_folds(
        ctx, list(ctx["folds"]), run_dir, trials, initial_equity)
    model_sha = _write_model_artifact(ctx, run_dir, trials)
    _finalize(ctx, run_id, run_dir, predictions, chosen_thresholds,
              fold_records, trials, initial_equity, model_sha256=model_sha)
    return run_id


def _write_model_artifact(ctx: dict, run_dir: Path, trials: TrialLedger) -> str | None:
    """Frozen decision artifact (F-002): model + conditional payoffs fitted at
    registration time on all matured labels; decide.py loads this immutable
    artifact and NEVER refits at decision time."""
    import pickle
    panel, feat_cols = ctx["panel"], ctx["feat_cols"]
    train_parts = []
    for sym, df in panel.items():
        train_parts.append(df.loc[training_eligible(df, fit_cutoff=ctx["sessions"][-1])]
                           .dropna(subset=feat_cols))
    final_train = pd.concat(train_parts, ignore_index=True)
    if not len(final_train) or final_train["label"].notna().sum() == 0:
        trials.log("model_artifact_skipped",
                   note="no matured labels; bundle will be decision-ineligible")
        return None
    y_final = (final_train["label"].to_numpy(dtype=float) > 0).astype(int)
    if len(np.unique(y_final)) < 2:
        trials.log("model_artifact_skipped",
                   note="single class in final training set; bundle will be "
                        "decision-ineligible")
        return None
    payoff_final = PayoffModel.fit(final_train, shrink=ctx["shrink"])
    final_model = make_model(ctx["primary_kind"],
                             ctx["scfg"]["model"].get(ctx["primary_kind"], {}),
                             ctx["seed"])
    final_model.fit(final_train[feat_cols].to_numpy(dtype=float), y_final)
    artifact = {"model": final_model, "payoff": payoff_final,
                "feature_names": feat_cols, "kind": ctx["primary_kind"],
                "trained_through": str(ctx["sessions"][-1].date())}
    (run_dir / "model.pkl").write_bytes(
        pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    return sha256_file(run_dir / "model.pkl")


def _merge_shard_outputs(run_dir: Path, expected_folds: set[int]) -> tuple[
        pd.DataFrame, dict[int, float], list[dict]]:
    """Concatenate shard outputs with disjointness/coverage validation."""
    shard_files = sorted(glob.glob(str(run_dir / "oos_predictions_shard*.csv")))
    if not shard_files:
        raise FileNotFoundError(f"no shard prediction files under {run_dir}")
    predictions = pd.concat((pd.read_csv(f, parse_dates=["date"])
                             for f in shard_files), ignore_index=True)
    chosen_thresholds: dict[int, float] = {}
    fold_records: list[dict] = []
    seen_folds: set[int] = set()
    for f in sorted(glob.glob(str(run_dir / "chosen_thresholds_shard*.json"))):
        for k, v in read_json(f).items():
            fid = int(k)
            if fid in seen_folds:
                raise RuntimeError(
                    f"fold {fid} predicted by multiple shards — shard outputs "
                    "must be disjoint; do not reuse a run_id across different "
                    "shard plans")
            seen_folds.add(fid)
            chosen_thresholds[fid] = float(v)
    for f in sorted(glob.glob(str(run_dir / "fold_records_shard*.json"))):
        fold_records.extend(read_json(f))
    missing = expected_folds - seen_folds
    if missing:
        raise RuntimeError(
            f"shards incomplete: folds {sorted(missing)} have no predictions — "
            f"run the remaining shards before finalizing")
    predictions = predictions.sort_values(
        ["fold_id", "date", "symbol", "model"]).reset_index(drop=True)
    fold_records.sort(key=lambda r: r["fold_id"])
    return predictions, chosen_thresholds, fold_records


def finalize_research(strategy: str, run_id: str, *, market: str | None = None,
                      initial_equity: float | None = None) -> str:
    """Merge shard outputs from a shared run directory and run the sequential
    evaluation (activation schedule, simulations, metrics, health, artifacts,
    registration). Call once after every `research --fit-shard` worker has
    finished."""
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"run '{run_id}' has no config.json — run the shards first "
            "(qa research --fit-shard i/n --run-id <id>)")
    cfg = read_json(cfg_path)
    market = market or cfg["market"]
    initial_equity = float(initial_equity if initial_equity is not None
                           else cfg.get("initial_equity", 1_000_000.0))
    ctx = _prepare_context(strategy, market, initial_equity)
    predictions, chosen_thresholds, fold_records = _merge_shard_outputs(
        run_dir, {f.fold_id for f in ctx["folds"]})
    trials = TrialLedger(run_dir, run_id)
    model_sha = _write_model_artifact(ctx, run_dir, trials)
    _finalize(ctx, run_id, run_dir, predictions, chosen_thresholds,
              fold_records, trials, initial_equity, model_sha256=model_sha)
    return run_id


def _plot_equity(eq: pd.DataFrame, path: Path, run_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col in eq.columns:
        axes[0].plot(eq.index, eq[col] / eq[col].iloc[0], label=col, linewidth=1.2)
    axes[0].set_title(f"Net equity (initial=1.0), run {run_id}")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)
    dd = eq["strategy"] / eq["strategy"].cummax() - 1.0
    axes[1].fill_between(eq.index, dd, 0, color="tab:red", alpha=0.4)
    axes[1].set_title("Strategy drawdown")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
