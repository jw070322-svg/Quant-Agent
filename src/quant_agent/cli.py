"""Command-line entry point: `qa <command>` (python -m quant_agent.cli ...).

Commands
  ingest     snapshot + normalize + validate + manifest per market
  research   full walk-forward research run (honest evaluation package)
  report     rebuild the markdown report for a run
  health     print a run's health record
  bundles    list registry bundles / set status
  decide     paper decision intents from an APPROVED bundle (no broker)
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import PROJECT_ROOT
from .util import read_json


def main(argv: list[str] | None = None) -> int:
    # Windows legacy console encodings cannot render research-report symbols.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(prog="qa", description="quant_agent pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="snapshot + normalize + validate datasets")
    p.add_argument("--market", required=True, choices=["us", "cn"])
    p.add_argument("--as-of", default=None, help="date to evaluate freshness against")

    p = sub.add_parser("update", help="fetch fresh bars from the provider (akshare)")
    p.add_argument("--market", required=True, choices=["us", "cn"])
    p.add_argument("--symbols", default=None, help="comma-separated subset")
    p.add_argument("--start", default=None, help="start date YYYY-MM-DD (default: config provider_start)")
    p.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--use-legacy-start", action="store_true",
                   help="start from each file's bundled history instead of provider_start")

    p = sub.add_parser("research", help="run walk-forward research")
    p.add_argument("--strategy", required=True)
    p.add_argument("--market", default=None)
    p.add_argument("--initial-equity", type=float, default=1_000_000.0)
    p.add_argument("--fit-shard", default=None, metavar="I/N",
                   help="fit only folds with fold_id %% n == I into the shared "
                        "run-id directory (distributed fold fitting)")
    p.add_argument("--run-id", default=None,
                   help="shared run id (required with --fit-shard)")

    p = sub.add_parser("research-finalize",
                       help="merge research shards and run the sequential evaluation")
    p.add_argument("--strategy", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--market", default=None)
    p.add_argument("--initial-equity", type=float, default=None)

    p = sub.add_parser("report", help="print a run's markdown report")
    p.add_argument("--run", required=True)

    p = sub.add_parser("health", help="print a run's health record")
    p.add_argument("--run", required=True)

    p = sub.add_parser("bundles", help="list registry bundles")
    p.add_argument("--approve", default=None, metavar="BUNDLE_ID")
    p.add_argument("--note", default="")
    p.add_argument("--actor", default="cli")
    p.add_argument("--evidence", default="",
                   help="approval evidence reference (metrics/artifact sha, ticket id)")

    p = sub.add_parser("decide", help="paper decision intents (approved bundles only)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--initial-equity", type=float, default=1_000_000.0)
    p.add_argument("--cash", type=float, default=None)

    p = sub.add_parser("fallback-stress",
                       help="stress evidence for the fallback rule (no acceptance)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--market", default=None)

    p = sub.add_parser("xs", help="cross-sectional factor screening study (H2/H3)")
    p.add_argument("--family", choices=["valuation", "technical"], default="valuation",
                   help="valuation=H2 fundamentals+size, technical=H3 TA attribution")
    p.add_argument("--index", default="000300")
    p.add_argument("--n", type=int, default=150, dest="n_symbols")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-data", action="store_true")

    p = sub.add_parser("multifactor", help="H4 multi-factor synthesis study")
    p.add_argument("--index", default="000300")
    p.add_argument("--n", type=int, default=150, dest="n_symbols")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("infomonitor", help="H5 critical information point monitor")
    p.add_argument("--stock-top", type=int, default=10)
    p.add_argument("--macro-top", type=int, default=8)
    p.add_argument("--recent-sessions", type=int, default=30)

    p = sub.add_parser("eventstudy", help="H6 forward validation of critical info points")
    p.add_argument("--news-budget", type=int, default=60)
    p.add_argument("--recent-sessions", type=int, default=30)

    p = sub.add_parser("snapshot-build",
                       help="assemble the CN canonical factor snapshot (token-free)")
    p.add_argument("--market", default="cn", choices=["cn"])
    p.add_argument("--indices", default="000300,000500",
                   help="comma-separated csindex codes for the universe")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--output", default="datasets/factor_snapshots/cn_v1")
    p.add_argument("--refresh", action="store_true",
                   help="refetch cached per-symbol files")

    p = sub.add_parser('factor-bundle', help='frozen factor research, offline evaluation')
    p.add_argument('--market', required=True, choices=['cn', 'us'])
    p.add_argument('--stage', required=True, choices=['readiness', 'evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--output')
    p = sub.add_parser('factor-family', help='ten-hypothesis family correction')
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('tushare-export', help='explicit immutable raw Tushare export')
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('factor-normalize', help='normalize verified exports using explicit mappings')
    p.add_argument('--raw', required=True)
    p.add_argument('--recipe', required=True)
    p.add_argument('--output', required=True)

    args = ap.parse_args(argv)

    if args.cmd == 'factor-bundle':
        from .research.factor_bundle import run_bundle
        print(run_bundle(args.market, args.stage, args.config, args.output))
        return 0
    if args.cmd == 'factor-family':
        from .research.factor_bundle import family_summary
        print(family_summary(args.reports, args.output))
        return 0
    if args.cmd == 'tushare-export':
        from .data.factor_snapshot import fetch_tushare_request
        result = fetch_tushare_request(args.request, args.output)
        print(json.dumps(result, indent=2))
        return 0 if all(r['status'] == 'retrieved' for r in result['requests']) else 2
    if args.cmd == 'factor-normalize':
        from .data.factor_snapshot import normalize_export
        print(normalize_export(args.raw, args.recipe, args.output))
        return 0

    if args.cmd == "ingest":
        from .data.ingest import ingest_market
        m = ingest_market(args.market, as_of=args.as_of)
        for sym, e in m["symbols"].items():
            if e.get("usable"):
                print(f"  {sym:12s} rows={e['rows']:<5} {e['date_start']}..{e['date_end']} "
                      f"schema={e['schema']} vol={e['volume_unit']} adj={e['adjustment_status']} "
                      f"quarantined={e['hard_quarantine_count']}")
            else:
                print(f"  {sym:12s} UNUSABLE: {e.get('reason')}")
        print(f"manifest: datasets/manifest_{args.market}.json")
        return 0

    if args.cmd == "update":
        from .config import load_market_config
        from .data.providers import update_market
        mcfg = load_market_config(args.market)
        start = args.start or mcfg.get("provider_start", "2019-01-02")
        symbols = args.symbols.split(",") if args.symbols else None
        m = update_market(args.market, symbols=symbols, start=start, end=args.end)
        for sym, e in m["symbols"].items():
            if "error" in e:
                print(f"  {sym:12s} FAILED: {e['error']}")
            else:
                print(f"  {sym:12s} rows={e['rows']:<5} {e['first_date']}..{e['last_date']} "
                      f"factor-changed={e['factor_dates_changed']}")
        print(f"updates: datasets/updates/{args.market}/ (ingest prefers these)")
        return 0

    if args.cmd == "research":
        from .research.runner import run_research
        run_id = run_research(args.strategy, market=args.market,
                              initial_equity=args.initial_equity,
                              fit_shard=args.fit_shard, run_id=args.run_id)
        if args.fit_shard:
            print(f"shard {args.fit_shard} complete for run '{run_id}'; "
                  "run remaining shards, then: qa research-finalize "
                  f"--strategy {args.strategy} --run-id {run_id}")
            return 0
        print(f"run complete: {run_id}")
        print(f"dir: artifacts/runs/{run_id}")
        metrics = read_json(PROJECT_ROOT / "artifacts" / "runs" / run_id / "metrics.json")
        print(json.dumps({
            "strategy_total_return": metrics["strategy"]["total_return"],
            "strategy_sharpe": metrics["strategy"]["sharpe"],
            "benchmarks": {k: v["total_return"] for k, v in metrics["benchmarks"].items()},
            "complete_run_claim": read_json(
                PROJECT_ROOT / "artifacts" / "runs" / run_id / "health.json")["complete_run_claim"],
        }, indent=2))
        return 0

    if args.cmd == "research-finalize":
        from .research.runner import finalize_research
        run_id = finalize_research(args.strategy, args.run_id, market=args.market,
                                   initial_equity=args.initial_equity)
        print(f"run finalized: {run_id}")
        metrics = read_json(PROJECT_ROOT / "artifacts" / "runs" / run_id / "metrics.json")
        print(json.dumps({
            "strategy_total_return": metrics["strategy"]["total_return"],
            "strategy_sharpe": metrics["strategy"]["sharpe"],
            "benchmarks": {k: v["total_return"] for k, v in metrics["benchmarks"].items()},
            "complete_run_claim": read_json(
                PROJECT_ROOT / "artifacts" / "runs" / run_id / "health.json")["complete_run_claim"],
        }, indent=2))
        return 0

    if args.cmd == "report":
        from .reporting.research import build_markdown_report
        print(build_markdown_report(args.run))
        return 0

    if args.cmd == "health":
        from .storage.run_history import get_health
        print(json.dumps(get_health(args.run), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "bundles":
        from .registry.store import list_bundles, set_status
        if args.approve:
            set_status(args.approve, "approved", note=args.note,
                       actor=args.actor, evidence=args.evidence)
            print(f"bundle {args.approve} -> approved (actor={args.actor})")
            return 0
        for b in list_bundles():
            print(f"  {b['bundle_id']}  status={b['status']}  strategy={b.get('strategy')}")
        return 0

    if args.cmd == "decide":
        from .research.decide import paper_decide
        record = paper_decide(args.bundle, initial_equity=args.initial_equity,
                              cash=args.cash)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "fallback-stress":
        from .research.fallback_study import run_fallback_stress_study
        study_dir = run_fallback_stress_study(args.strategy, market=args.market)
        print(f"stress study complete: {study_dir.name}")
        print((study_dir / "stress_report.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "xs":
        if args.family == "technical":
            from .research.cross_section import run_xs_technical_study
            study_dir = run_xs_technical_study(index=args.index, n_symbols=args.n_symbols,
                                               seed=args.seed)
        else:
            from .research.cross_section import run_cross_section_study
            study_dir = run_cross_section_study(index=args.index, n_symbols=args.n_symbols,
                                                seed=args.seed, refresh_data=args.refresh_data)
        print(f"cross-section study complete: {study_dir.name}")
        print((study_dir / "report.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "multifactor":
        from .research.multi_factor import run_multi_factor_study
        study_dir = run_multi_factor_study(index=args.index, n_symbols=args.n_symbols,
                                           seed=args.seed)
        print(f"multi-factor study complete: {study_dir.name}")
        print((study_dir / "report.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "infomonitor":
        from .research.info_monitor import run_info_monitor_study
        study_dir = run_info_monitor_study(stock_top=args.stock_top,
                                           macro_top=args.macro_top,
                                           recent_sessions=args.recent_sessions)
        print(f"information monitor study complete: {study_dir.name}")
        print((study_dir / "report.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "eventstudy":
        from .research.event_study import run_forward_validation_study
        study_dir = run_forward_validation_study(news_budget=args.news_budget,
                                                 recent_sessions=args.recent_sessions)
        print(f"forward validation study complete: {study_dir.name}")
        print((study_dir / "report.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "snapshot-build":
        from .research.snapshot_cn import build_cn_snapshot
        result = build_cn_snapshot(indices=args.indices.split(","),
                                   start=args.start, end=args.end,
                                   output=args.output, refresh=args.refresh,
                                   progress_cb=print)
        print(json.dumps({"output": str(result["output"]),
                          "table_rows": result["table_rows"],
                          "dropped_bars": result["dropped_bars"]["dropped_total"],
                          "limitations": result["manifest"]["limitations"]},
                         indent=2, ensure_ascii=False, default=str))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
