# Quant-Agent

A reproducible quant research and paper-trading pipeline for A-share and US
daily data, rebuilt from the legacy prototype under
`handover/QUANT_AGENT_HANDOVER.md`. Legacy scripts (`scripts/`, root `*.py`,
`models/`, `outputs/`) are preserved unmodified as unverified research
artifacts; this package is the supported implementation.

## Quick start (portable)

The new US/A-share factor research bundle is documented in
[`handover/FACTOR_BUNDLE_IMPLEMENTATION.md`](handover/FACTOR_BUNDLE_IMPLEMENTATION.md).
It provides frozen MOM/ATO/QV/FAR/VPA definitions, offline evaluation, explicit
Tushare export/normalization, raw-share accounting, and blocked-data reports.
Start with `python -m quant_agent.cli factor-bundle --market cn --stage readiness
--config configs/factor_bundles/cn_v1.yaml`. Tushare access and canonical historical
snapshots are required for empirical validation; no strategy is approved.

Works on any machine with Python >= 3.11 and network access for market data —
no hardcoded paths, no specific interpreter required:

```bash
git clone https://github.com/jw070322-svg/Quant-Agent.git
cd Quant-Agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests -q

# Data + research (uses `python` from PATH; everything resolves relative to
# the repository root)
python -m quant_agent.cli update --market cn
python -m quant_agent.cli ingest --market cn --as-of 2026-11-15
python -m quant_agent.cli research --strategy us_research_v1
python -m quant_agent.cli report  --run <run_id>
python -m quant_agent.cli health  --run <run_id>

# Factor studies (all pre-registered in docs/DECISIONS.md)
python -m quant_agent.cli xs --family technical      # H3 technical attribution
python -m quant_agent.cli multifactor                # H4 multi-factor synthesis
python -m quant_agent.cli infomonitor                # H5 critical-info monitor
python -m quant_agent.cli eventstudy                 # H6 forward validation

# Registry and paper decisions (gated)
python -m quant_agent.cli bundles --approve <bundle_id> --actor risk-owner --evidence <ref>
python -m quant_agent.cli decide --bundle <bundle_id>
```

## Distributed fold fitting (GPU swarm)

Fold fitting is embarrassingly parallel — folds are disjoint by construction.
Fit shards on different workers (one shared run id), then merge and evaluate
on any single machine:

```bash
# worker 0 — on a CUDA box, set model.gbm.device: cuda in the strategy config
python -m quant_agent.cli research --strategy us_research_v1 --fit-shard 0/2 --run-id us-run-001
# worker 1
python -m quant_agent.cli research --strategy us_research_v1 --fit-shard 1/2 --run-id us-run-001
# merge + sequential evaluation (validated: disjoint fold coverage enforced;
# merged predictions are identical to an unsharded run)
python -m quant_agent.cli research-finalize --strategy us_research_v1 --run-id us-run-001
```

GPU enablement: set `model.gbm.device: cuda` in the strategy config on a
CUDA machine (xgboost >= 2.0; verified on an RTX 5060). If CUDA is unusable
at fit time, the runner falls back to the CPU path automatically.

## Development-machine notes

The original development environment (Windows, Anaconda Python 3.13.9 at
`E:/Anaconda/python`) is recorded in `requirements.lock`; `pip install -e
".[dev]"` on Python >= 3.11 is the portable equivalent and is the supported
path for other machines.

## Layout

| Location | Responsibility |
|---|---|
| `src/quant_agent/` | Supported Python package. See the [layer map](docs/ARCHITECTURE.md#1-module-map-and-contracts). |
| `configs/` | Market, strategy, risk, and factor-bundle settings. |
| `data/`, `datasets/` | Legacy source data, plus research snapshots, normalized data, updates, and manifests. |
| `artifacts/` | Research evidence, including immutable runs under `runs/` and approval-gated bundles under `models/`. |
| `tests/` | Unit, integration, and replay checks. |
| `scripts/`, root training scripts | Standalone research scripts and the legacy prototype. |
| `docs/`, `handover/` | Current architecture and validation, plus historical plans and audit evidence. |

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
