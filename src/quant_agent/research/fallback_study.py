"""Independent-acceptance EVIDENCE for the fallback rule (review doc P1).

This module produces the stress evidence the acceptance decision needs —
it does NOT grant acceptance. `fallback_accepted` may only be set by a
human after reviewing this report (see alpha/factor_fallback.py).

Stress dimensions (review doc section II.3.4): cost multiples, one-session decision
lag (delayed fills), churn/whipsaw behavior, and loss co-movement with the
primary strategy over the identical evaluation span.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..alpha.factor_fallback import FALLBACK_DEFAULT_BUDGET, fallback_rule_weights
from ..config import PROJECT_ROOT, load_market_config, load_risk_config, load_strategy_config
from ..data.ingest import load_normalized
from ..data.calendar import market_sessions
from ..execution.simulator import run_simulation
from ..features.pipeline import build_panel
from ..labels.forward import LabelSpec, add_labels
from ..research.metrics import completed_trade_stats, summarize
from ..research.splits import build_folds
from ..risk.checks import RiskLimits
from ..util import config_fingerprint, now_utc_iso, write_json


def _latest_primary_run(strategy: str, market: str) -> Path | None:
    runs = sorted(PROJECT_ROOT.glob(f"artifacts/runs/{strategy}-{market}-*"))
    runs = [r for r in runs if (r / "equity_curve.csv").exists()
            and "-fallback-study" not in r.name and "-fb" not in r.name]
    return runs[-1] if runs else None


def _fallback_weights_by_date(panel: dict, sessions: pd.DatetimeIndex,
                              name_cap: float) -> dict[pd.Timestamp, dict]:
    out = {}
    for t in sessions:
        factor_rows = {}
        for sym, df in panel.items():
            row = df.loc[df["date"] == t]
            if not len(row):
                continue

            def _f(col, row=row):
                v = row[col].iloc[0]
                return float(v) if pd.notna(v) else None

            factor_rows[sym] = {"ret_20": _f("ret_20"),
                                "close_over_ma60": _f("close_over_ma60"),
                                "vol_20": _f("vol_20")}
        w, _ = fallback_rule_weights(factor_rows, budget=FALLBACK_DEFAULT_BUDGET,
                                     max_weight_per_name=min(0.10, name_cap))
        out[t] = w
    return out


def _scale_costs(mcfg: dict, factor: float) -> dict:
    c = dict(mcfg)
    costs = dict(mcfg["costs"])
    for k in ("commission_bps", "slippage_bps", "stamp_tax_sell_bps", "min_commission"):
        costs[k] = costs[k] * factor
    c["costs"] = costs
    return c


def run_fallback_stress_study(strategy: str, *, market: str | None = None,
                              initial_equity: float = 1_000_000.0) -> Path:
    t0 = time.time()
    scfg = load_strategy_config(strategy)
    market = market or scfg["market"]
    mcfg = load_market_config(market)
    rcfg = load_risk_config()
    name_cap = rcfg["max_weight_per_name"]
    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})

    frames = load_normalized(market, mcfg["universe"])
    panel = build_panel(frames, scfg["features"])
    for sym, df in panel.items():
        df["symbol"] = sym
    panel = {sym: add_labels(df, LabelSpec(**scfg["label"])) for sym, df in panel.items()}
    sessions = market_sessions(frames)
    folds = build_folds(sessions,
                        initial_train_fraction=scfg["walk_forward"]["initial_train_fraction"],
                        test_sessions=scfg["walk_forward"]["test_sessions"],
                        step_sessions=scfg["walk_forward"]["step_sessions"],
                        policy_tune_fraction=scfg["walk_forward"]["policy_tune_fraction"])
    sessions_oos = sessions[sessions >= folds[0].predict_start]

    weights_by_date = _fallback_weights_by_date(panel, sessions, name_cap)
    prev_of = {t: sessions[i - 1] for i, t in enumerate(sessions_oos) if i > 0}

    run_id = f"{strategy}-{market}-{config_fingerprint({'s': strategy, 'm': market})}-{now_utc_iso().replace(':', '')}-fallback-study"
    study_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    study_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "baseline": (mcfg, lambda t, a, p: weights_by_date.get(t)),
        "costs_x2": (_scale_costs(mcfg, 2.0), lambda t, a, p: weights_by_date.get(t)),
        "costs_x4": (_scale_costs(mcfg, 4.0), lambda t, a, p: weights_by_date.get(t)),
        "decision_lag_1s": (mcfg, lambda t, a, p: weights_by_date.get(prev_of.get(t))
                            if t in prev_of else None),
    }

    results = {}
    sim_returns = {}
    for name, (cfg_i, fn) in variants.items():
        sim = run_simulation(panel, sessions_oos, fn, market_cfg=cfg_i,
                             limits=limits, initial_equity=initial_equity,
                             run_id=f"{run_id}-{name}")
        spy = max(int(round(len(sessions) / max((sessions[-1] - sessions[0]).days / 365.25, 1e-9))), 1)
        summary = summarize(sim.daily_returns, sessions_per_year=spy,
                            turnover=float(sim.turnover.sum()),
                            costs_paid=float(sim.costs.iloc[-1]),
                            exposure=float(sim.exposure.mean()),
                            n_trades=len(sim.fills))
        summary.update(completed_trade_stats(sim.fills))
        summary["n_stop_events"] = len(sim.stop_events)
        gross = sim.exposure
        summary["exposure_sign_flips"] = int(
            ((gross > 0.01).astype(int).diff() != 0).clip(lower=0).sum())
        worst10 = sim.daily_returns.rolling(10).sum().min() if len(sim.daily_returns) >= 10 else None
        summary["worst_10session_sum"] = float(worst10) if worst10 is not None else None
        results[name] = summary
        sim_returns[name] = sim.daily_returns

    # loss co-movement with the primary strategy (from its completed run)
    correlation = {"primary_run": None, "return_correlation": None,
                   "both_negative_day_fraction": None, "note": None}
    primary_run = _latest_primary_run(strategy, market)
    if primary_run is not None:
        eq = pd.read_csv(primary_run / "equity_curve.csv", index_col=0, parse_dates=True)
        primary_ret = eq["strategy"].pct_change().dropna()
        base = sim_returns["baseline"].dropna()
        joined = pd.concat([primary_ret, base], axis=1, join="inner").dropna()
        joined.columns = ["primary", "fallback"]
        if len(joined) >= 20:
            both_neg = ((joined["primary"] < 0) & (joined["fallback"] < 0)).mean()
            correlation = {
                "primary_run": primary_run.name,
                "return_correlation": float(joined["primary"].corr(joined["fallback"])),
                "both_negative_day_fraction": float(both_neg),
                "n_overlapping_sessions": int(len(joined)),
                "note": ("high co-movement in drawdowns means the fallback does NOT "
                         "diversify the primary strategy's failure mode"),
            }
    else:
        correlation["note"] = "no completed primary run found for this strategy/market"

    report = {
        "run_id": run_id,
        "strategy": strategy, "market": market,
        "created_utc": now_utc_iso(),
        "status": "UNVALIDATED STRESS EVIDENCE — acceptance is a human decision",
        "variants": results,
        "loss_correlation_with_primary": correlation,
        "acceptance": "PENDING HUMAN REVIEW (fallback_accepted must stay false "
                      "until stress + correlation + whipsaw review passes)",
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_json(study_dir / "stress_report.json", report)
    _write_md(report, study_dir / "stress_report.md")
    return study_dir


def _write_md(report: dict, path: Path) -> None:
    lines = [
        f"# Fallback rule stress study: `{report['run_id']}`",
        "",
        f"- Status: **{report['status']}**",
        f"- Acceptance: {report['acceptance']}",
        "",
        "| Variant | Total return | Sharpe | MaxDD | Trades | Costs | Stops | Expos.flips | Worst 10s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in report["variants"].items():
        lines.append(
            f"| {name} | {m['total_return']:.2%} | {m['sharpe']:.2f} | {m['max_drawdown']:.2%} "
            f"| {m['n_trades']} | {m['costs_total']:.0f} | {m['n_stop_events']} "
            f"| {m['exposure_sign_flips']} | {m['worst_10session_sum']:.2%} |")
    corr = report["loss_correlation_with_primary"]
    lines += ["", "## Loss co-movement with primary strategy", ""]
    if corr.get("return_correlation") is not None:
        lines.append(
            f"- primary run: `{corr['primary_run']}` "
            f"(n={corr['n_overlapping_sessions']} overlapping sessions)")
        lines.append(f"- daily-return correlation: **{corr['return_correlation']:.3f}**")
        lines.append(f"- both-negative days: **{corr['both_negative_day_fraction']:.1%}** "
                     f"— {corr['note']}")
    else:
        lines.append(f"- {corr['note']}")
    lines += ["", "## Claim boundary", "",
              "- The fallback rule remains UNVALIDATED design delivery; high returns in a "
              "trending window are not acceptance evidence.",
              "- Acceptance requires pre-registered thresholds (docs/DECISIONS.md) and a "
              "recorded human decision."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
