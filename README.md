# Quant-Agent

A reproducible quant research and paper-trading pipeline for A-share and US
daily data, rebuilt from the legacy prototype under
`handover/QUANT_AGENT_HANDOVER.md`. Legacy scripts (`scripts/`, root `*.py`,
`models/`, `outputs/`) are preserved unmodified as unverified research
artifacts; this package is the supported implementation.

## Quick start

```bash
cd "F:/quant agent/quant_project"
export PYTHONPATH=src
export PY=/e/Anaconda/python        # Python 3.13.9 with the full stack

# Data: fetch fresh bars + valuation + news, then validate and normalize
$PY -m quant_agent.cli update --market cn
$PY -m quant_agent.cli ingest --market cn --as-of 2026-11-15

# Research: walk-forward evaluation with pre-registered protocol
$PY -m quant_agent.cli research --strategy us_research_v1
$PY -m quant_agent.cli report  --run <run_id>
$PY -m quant_agent.cli health  --run <run_id>

# Factor studies (all pre-registered in docs/DECISIONS.md)
$PY -m quant_agent.cli xs --family technical      # H3 technical attribution
$PY -m quant_agent.cli multifactor                # H4 multi-factor synthesis
$PY -m quant_agent.cli infomonitor                # H5 critical-info monitor
$PY -m quant_agent.cli eventstudy                 # H6 forward validation

# Registry and paper decisions (gated)
$PY -m quant_agent.cli bundles --approve <bundle_id> --actor risk-owner --evidence <ref>
$PY -m quant_agent.cli decide --bundle <bundle_id>
```

Tests: `$PY -m pytest tests -q` (110 tests, all passing at the time of
writing). Interpreter note: use the Anaconda Python at `E:/Anaconda/python`;
the msys2 Python on PATH has no pip and no scientific stack.

## Layout

- `src/quant_agent/` — the supported package (see docs/ARCHITECTURE.md).
- `configs/` — market, risk, and strategy configurations.
- `datasets/` — raw snapshots, normalized data, update layer, manifests.
- `artifacts/runs/` — immutable research runs (one directory per run).
- `artifacts/models/` — registry bundles (approval-gated).
- `docs/` — architecture, decisions, validation evidence.
- `handover/` — handover history and the original audit documents.

## Documentation

- docs/ARCHITECTURE.md — module contracts, execution timeline, known limits.
- docs/DECISIONS.md — design decisions, pre-registered hypotheses (H1-H6)
  with adjudications, unresolved decisions.
- docs/VALIDATION.md — protocol, results, calibration evidence, evidence
  grades (E0-E6).
- handover/REBUILD_HANDOVER.md — engineering handover narrative.

## Language policy

All project-owned documentation, code comments, and generated reports are in
English. Two historical input documents are preserved in their original
language as immutable evidence: `handover/CURRENT_STATUS_FACTOR_FALLBACK_CN.md`
(independent review) and `handover/audit_checks.py`. Run artifacts under
`artifacts/runs/` are immutable evidence and are not retro-translated.

## Status

Offline research candidate. No strategy has evidence beyond E3 (offline
backtest/screening); nothing is production-ready. See docs/VALIDATION.md §0.1.
