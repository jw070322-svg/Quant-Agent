"""Grounded markdown report builder over run artifacts (read-only)."""
from __future__ import annotations

import pandas as pd

from . import tools


def build_markdown_report(run_id: str) -> str:
    run = tools.get_run(run_id)
    prov = run["provenance"]
    metrics = run["metrics"] or {}
    health = run["health"] or {}
    lines = [
        f"# Research run report: `{run_id}`",
        "",
        f"- Strategy: `{prov.get('strategy')}` | Market: `{prov.get('market')}` "
        f"| Created: {prov.get('created_utc')}",
        f"- Config fingerprint: `{prov.get('config_fingerprint')}` "
        f"(strategy+market+risk+dataset hashes recorded in config.json)",
        f"- Scope: retrospective RESEARCH on bundled historical data. "
        f"Not investment advice; not a live performance claim.",
        "",
        "## Health",
        "",
        f"- complete-run claim possible: **{health.get('complete_run_claim')}** "
        f"(pass {health.get('pass')} / warn {health.get('warn')} / fail {health.get('fail')})",
        "",
        "## Headline metrics",
        "",
    ]
    strat = metrics.get("strategy", {})
    lines += [
        f"- Net compounded return: **{strat.get('total_return', 0):.2%}** over "
        f"{strat.get('n_sessions', 0)} sessions",
        f"- Annualized: {strat.get('annual_return', 0):.2%} | vol {strat.get('annual_vol', 0):.2%} "
        f"| Sharpe {strat.get('sharpe', 0):.2f} "
        f"(CI {metrics.get('bootstrap', {}).get('sharpe_ci_low')}, "
        f"{metrics.get('bootstrap', {}).get('sharpe_ci_high')})",
        f"- Max drawdown: {strat.get('max_drawdown', 0):.2%} | costs {strat.get('costs_total', 0):.0f} "
        f"| exposure {strat.get('exposure') if strat.get('exposure') is not None else float('nan'):.0%}",
        "",
        "## Benchmark comparison (same costs & execution)",
        "",
        "| Benchmark | Total return | Sharpe | MaxDD |",
        "|---|---:|---:|---:|",
    ]
    for name, m in (metrics.get("benchmarks") or {}).items():
        lines.append(f"| {name} | {m.get('total_return', 0):.2%} | "
                     f"{m.get('sharpe', 0):.2f} | {m.get('max_drawdown', 0):.2%} |")
    lines += ["", "## Calibration (primary model, matured OOS labels)", ""]
    cal = metrics.get("calibration", {})
    if cal:
        lines.append(
            f"- n={cal.get('n')} base_rate={cal.get('base_rate'):.4f} "
            f"brier={cal.get('brier'):.4f} auc={cal.get('auc')}")
    lines += [
        "",
        "## Honest-claims block",
        "",
        "- Costs are UNVERIFIED defaults from configs; results are sensitive to them.",
        "- The universe is a small fixed set chosen with today's knowledge; survivorship "
        "bias is NOT addressed at this scale.",
        "- Search history (all folds/thresholds tried) is in trials.jsonl; the reported "
        "result carries that selection burden.",
        "- No edge claim is made unless pre-registered criteria (docs/DECISIONS.md) are met.",
    ]
    return "\n".join(lines) + "\n"
