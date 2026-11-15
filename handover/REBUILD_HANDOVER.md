# Quant Agent Rebuild Engineering Handover (v3.2, English)

Date: 2026-11-15 (v3.2 converts the handover to English; the adjudications of
hypotheses H1-H6 are incorporated)
Project root: `F:\quant agent\quant_project`
Document chain: `QUANT_AGENT_HANDOVER.md` (original audit) →
`REBUILD_HANDOVER.md` (v1-v3 rebuild record; this file) →
`CURRENT_STATUS_FACTOR_FALLBACK_CN.md` (independent review + fallback design,
preserved in Chinese as an immutable historical input) →
`ZCODE_NEXT_ITERATION_PLAN.md` (2026-11-12 plan from another session, whose
findings were addressed by the audit round; preserved as a translated
historical plan below). Legacy prototypes (root `*.py`, `scripts/`, `models/`,
`outputs/`) are preserved unmodified as unverified research artifacts.

## 0. Audit-fix round (2026-11-15; every item followed "failing test → minimal
fix → full regression")

| ID | Severity | Location | Defect | Fix | Regression tests |
|---|---|---|---|---|---|
| F-001 | High | `alpha/factor_fallback.py` | After HALT, an infra recovery **automatically** returned to PRIMARY with a fabricated "operator-confirmed" audit reason | Split into `HALT→RECOVERY (automatic, restricted observation)` and `RECOVERY→PRIMARY (OPERATOR_RECOVERY_APPROVED, requires evidence + cooldown)`; the observation counter resets on entry; audit records from/to/event/evidence/actor | 7 tests (`test_halt_recovery_requires_operator_approval` etc.) |
| F-002 | Critical | `registry/store.py` + `research/decide.py` + `research/runner.py` | Bundles had no model artifact; decide retrained at decision time; dataset hashes recorded but never compared | The runner freezes `model.pkl` (model + conditional payoffs + feature names, SHA-256 in REGISTRY/config) at registration; decide only loads approved artifacts and verifies the hash; dataset drift → PermissionError | 4 decide tests incl. tamper/missing/drift |
| F-003 | High | `registry/store.py` | Approval status accepted arbitrary strings with no transition validation and no actor | Legal status set {candidate, approved, revoked, expired} + transition table; illegal writes → ValueError; history records actor/evidence/from/to | `tests/unit/test_registry_gate.py` (4 tests) |
| F-004 | High | `risk/checks.py` + `execution/simulator.py` + `decide.py` | Cash checks ignored frozen pending-buy notional → double commitment | `check_order_intents` gains `pending_buy_value`; simulator and decide pass the pending BUY notional | 2 tests (risk + decide) |
| F-005 | Medium | `alpha/factor_fallback.py` | Cooldown counted calls, not unique sessions; corrupt state files not handled conservatively | `last_session` dedup; new `StateCorruptionError` (corruption raises, never auto-resets) | 2 tests |

Test baseline: 81 → **110 passing / 0 failing**. Decision-contract change:
decide now requires a bundle with a model artifact — bundles registered before
2026-11-15 are explicitly refused and must be re-registered (intentional
safety behavior).

## 0.1 Environment facts (must read for any successor)

| Fact | Implication |
|---|---|
| **Use the Anaconda Python** | `E:/Anaconda/python` (3.13.9; numpy/pandas/scipy/sklearn/xgboost/matplotlib/yaml/pytest complete). The msys2 Python on PATH (3.14.7) has no pip and no scientific stack |
| **Run mode** | `PYTHONPATH=src /e/Anaconda/python -m quant_agent.cli <cmd>`; tests: `/e/Anaconda/python -m pytest` (root conftest.py injects src). `-I` isolation drops PYTHONPATH |
| **No git repository** | `.gitignore` ready; initialize and commit a baseline before further changes |
| Data are historical snapshots | US ends 2026-03/06 → refreshed via the provider chain to 2026-11-13; freshness warnings on legacy-only data are factual statements |
| All cost parameters UNVERIFIED | Commission/slippage/stamp tax carry legacy defaults, flagged in configs; every conclusion is sensitive to them |
| Dependencies unlocked in practice | `requirements.lock` records the verified environment (not a hermetic lock) |
| Language policy | All project-owned docs/comments/reports in English; `CURRENT_STATUS_FACTOR_FALLBACK_CN.md` and `audit_checks.py` preserved in the original language as immutable historical evidence |

## 1. Phase status (honest version)

| Phase | Status | Evidence |
|---|---|---|
| Phase 0 evidence preservation / reproducibility | ✅ Done | Snapshots + SHA-256 manifests (changes archived, not clobbered); downloader path fix; pyproject; README.md (English) added |
| Phase 1 deterministic defect fixes | ✅ Code done, regression-tested | 110 tests incl. hand-calculated reconciliations and failure scenarios |
| Phase 2 simulator and research baseline | ✅ Acceptance scenarios tested | Toy reconciliation, zero-trade, partial fills, stops, replay, idempotency all in `tests/` |
| Phase 3 incremental-value adjudication | ✅ US+CN complete; new-data re-check complete | US primary +12.9% (Sharpe 0.80; report window +9.0%) beat buy-and-hold −4.0% for the first time, but criteria still fail (AUC 0.500, CI lower bound −0.17, trend rule +50.9% dominates); CN −11.6%, AUC 0.494. **No approvable bundle** |
| Data update chain (P2) | ✅ Done (2026-11-14/15) | akshare token-free (CN = Sina, US = Yahoo); CN adjustment-factor gap closed; uniform US 7.4-year history; update/provenance/freshness tested |
| Phase 4 paper execution | ✅ Mechanism done; ⬜ no live feed/broker | decide integration 5 tests; staleness blocking refuses all BUYs on stale snapshots by design |
| Phase 5 controlled deployment | ⬜ Not started (gates unmet) | docs/DECISIONS.md §2.3 lists entry conditions |
| Factor mining H1 (2026-11-15) | ✅ Pre-registered → run → adjudicated | 8-factor v2 pack: delta-AUC (US +0.006/CN +0.001) far below the pre-registered +0.02 → **H1 rejected, v2 pack retired**; US v2 returns improved (+26.1%, first positive CI) but the calibration gate failed; bundle stays candidate |
| Cross-sectional screening H2 (2026-11-15) | ✅ Pre-registered → fetched → evaluated → adjudicated | akshare three sources; 150-symbol sample, 60 windows; **no factor passed the full screen**; size the only IC-screen pass (t −2.82, net spread +37%) but non-monotone |
| Technical attribution H3 (2026-11-15) | ✅ Pre-registered → run → adjudicated | 16 indicators individually attributed (74 windows × 145 names): **none passed the full screen**; FM-consistent significance only bb_pos20/vol_5 (net portfolios failed); ret_1 shows 1-day momentum (t +2.87) opposite to the pre-registered reversal |
| Multi-factor synthesis H4 (2026-11-15) | ✅ Pre-registered → run → adjudicated | Train/test split 30/30, 14 factors MAD + size-neutralized; screening selected BP/amihud/turnover/vol_5; test-window **equal-weight synthesis +12.6% < benchmark +18.6%; all three weight-learning schemes turned negative → H4 rejected**; final status rejected |
| Info-point monitor H5 (2026-11-15) | ✅ Pre-registered → implemented → run → validated | Deal-amount anomaly (robust z / d-o-d) × news correlation dual-requirement module; full-history scan of 150 symbols → 9556 anomaly clusters; market 8/8 and stock 5/10 dual-confirmed (incl. 2024-09-24 z = 12.1); JSON contract + reasoning-first enforced by tests; CCTV keyword low specificity disclosed; E3 |
| Forward validation H6 (2026-11-15) | ✅ Pre-registered → run → adjudicated | Forward outcomes for 9548 clusters + 60 news fetches (24 dual / 16 quant-only) + CCTV verification of all 39 market clusters: **the news leg shows no measurable incremental forward information → H6 rejected; the monitor stays review-only (E3)**; recorded findings: volume persistence 210% (t 73.5), 5-session drift +0.42% (t 2.62, unadjusted) |
| Fallback scheme | ✅ Design + stress evidence; 🔶 acceptance pending human | US passes costs ×4 / lag / correlation stress; CN fails the lag leg (−1.2%); `fallback_accepted` stays false |
| Engineering baseline | ✅ | 110 tests green; requirements.lock; docs trio in English; README.md added |

## 2. Current file map

```text
quant_project/
├── README.md                          # English project overview and quick start
├── pyproject.toml  requirements.txt  requirements.lock  conftest.py  .gitignore
├── configs/
│   ├── markets/{us,cn}.yaml           # market rules/costs (UNVERIFIED)/T+1/lots/provider_start
│   ├── risk/default.yaml              # independent risk policy
│   └── strategies/{us,cn}_research_v1.yaml, {us,cn}_factor_v2.yaml
├── src/quant_agent/
│   ├── cli.py                         # qa update|ingest|research|report|health|
│   │                                  #   bundles|decide|fallback-stress|xs|
│   │                                  #   multifactor|infomonitor|eventstudy
│   ├── config.py  util.py
│   ├── data/{ingest,schema,calendar,providers}.py
│   ├── features/{indicators,pipeline}.py   # 24 causal dimensionless features
│   ├── labels/forward.py
│   ├── research/{runner,splits,models,metrics,trials}.py
│   ├── research/{cross_section,multi_factor,info_monitor,event_study,
│   │              fallback_study,decide}.py
│   ├── alpha/{expected_return,factor_fallback}.py
│   ├── portfolio/optimizer.py
│   ├── risk/checks.py
│   ├── ledger/account.py
│   ├── execution/{orders,broker,simulator}.py
│   ├── registry/store.py
│   ├── monitoring/health.py
│   └── agent/{tools,report}.py
├── datasets/{raw,updates,normalized}/ + manifest_{market}.json
├── artifacts/{runs,models,decisions}/
├── tests/{unit,integration,replay}/   # 110 tests
├── docs/{ARCHITECTURE,DECISIONS,VALIDATION}.md   # English
└── handover/                          # this file + historical inputs
```

## 3. Research adjudications (all pre-registered, all backfilled)

| Hypothesis | Question | Verdict | Key numbers |
|---|---|---|---|
| H1 | v2 factor-pack incrementality | **Rejected** | ΔAUC +0.006/+0.001 < +0.02; US v2 +26.1% (first positive CI 0.512) but calibration gate failed |
| H2 | Cross-sectional value/size/momentum/low-vol/liquidity | **No full-screen pass** | size only IC pass (t −2.82; net spread +37.2%) but non-monotone; BP/amihud/turnover/vol_5 observed in H4 train windows |
| H3 | Technical indicator attribution | **No full-screen pass** | ret_1 shows 1-day momentum (t +2.87) opposite to the reversal pre-registration; bb_pos20/vol_5 FM-consistent but portfolios failed |
| H4 | Multi-factor synthesis + TOP_N | **Rejected** | equal-weight synthesis +12.6% < universe benchmark +18.6%; learned weights all negative; weight learning rejected under the rules |
| H5 | Info-point monitor module | **Validation passed** | 13 dual-confirmed points incl. 2024-09-24 (z = 12.1); low CCTV specificity disclosed |
| H6 | Forward value of the news leg | **Rejected** | paired test structurally infeasible (zero date overlap); market 39/39 flagged; exploratory tests insignificant; persistence 210% (t 73.5) recorded |

Every adjudication lives in `docs/DECISIONS.md §2.4-§2.9`; every run's trial
ledger is `artifacts/runs/<run_id>/trials.jsonl`. **No forward value has been
established for any hypothesis; the monitor is a review layer, not a signal.**

## 4. Remaining work (priority order)

1. **P0**: historical constituents data (removes survivorship bias) — external
   input required before any revived factor/synthesis can be re-registered.
2. **P0**: policy-domain materiality model for the market news leg (CCTV
   keyword presence has zero specificity — 39/39 flagged).
3. **P0**: git initialization and a baseline commit (no version auditability
   exists; not executed without authorization).
4. **P1**: paper-forward shadow evaluation of the monitor and any revived
   candidate under new pre-registrations (E3 → E4 requires held-out evidence).
5. **P1**: pinned dependency verification on a clean environment
   (`requirements.lock` is an environment record).
6. **P2**: broker paper integration, reconciliation, alerting — all gated by
   DECISIONS §2.3 and by the absence of any approvable bundle.

## 5. Command quick reference

```bash
cd "F:/quant agent/quant_project"
export PYTHONPATH=src PY=/e/Anaconda/python
$PY -m pytest                      # 110 tests
$PY -m quant_agent.cli update --market cn
$PY -m quant_agent.cli ingest --market cn --as-of 2026-11-15
$PY -m quant_agent.cli research --strategy us_research_v1
$PY -m quant_agent.cli xs --family technical
$PY -m quant_agent.cli multifactor
$PY -m quant_agent.cli infomonitor
$PY -m quant_agent.cli eventstudy
$PY -m quant_agent.cli report --run <run_id>
$PY -m quant_agent.cli health --run <run_id>
$PY -m quant_agent.cli bundles [--approve <id> --actor <who> --evidence <ref>]
$PY -m quant_agent.cli decide --bundle <id> --cash 800000 \
     --holdings '{"600519.SH": 100}' --pending-orders '[{"symbol":"600519.SH","side":"BUY","qty":100,"ref_price":1300.0}]'
```

Artifacts: `artifacts/runs/<run_id>/` (config/folds/metrics/health/report.md/
equity/oos_predictions/fills/orders/trials.jsonl; studies add factor panels,
IC series, critical-point JSONs); registry `artifacts/models/<bundle_id>/`.

## 6. Honest declaration

- This rebuild delivers **correctness and protocol**: net-of-cost, compounded,
  unique-OOS, matured-label, independently-risk-checked, replayable accounting
  with uncertainty reporting. It constitutes **no profitability claim**. Six
  pre-registered hypotheses (H1-H6) produced no approvable candidate; negative
  results are preserved verbatim as the deliverable of Phase 3.
- Unverified list: cost parameters, the fallback rule, real-time health
  integration, forward/paper evidence, wider universes. Any step toward
  paper/live trading must first pass docs/DECISIONS.md §2.3 and the
  authorization boundaries of the governing workflow.

## 7. Translated historical plan (ZCODE_NEXT_ITERATION_PLAN.md, 2026-11-12)

That plan (from another session, based on handover v3) identified state-machine
auto-recovery, registry-decide inconsistency, and frozen-funds gaps — all now
fixed under F-001/F-002/F-004 in §0. Its work packages map as follows:
A (state machine + auditable recovery) → done; B (frozen artifacts + decision
consistency) → done for the local contract; C (README/docs/environment) → done
in English; D (fallback independent acceptance) → stress evidence complete
(US passes, CN fails lag), acceptance still pending human review. Its
completion definition ("decision-source consistency, deterministic fault
handling, verifiable risk constraints") is the state this handover describes.
The original file is translated in place below for a single-language context.

**Translated content of `handover/ZCODE_NEXT_ITERATION_PLAN.md`:**

> # ZCode next-round engineering optimization and acceptance plan
>
> Date: 2026-11-12. Scope: `F:\quant agent\quant_project`. Basis:
> `handover/REBUILD_HANDOVER.md` v3, current source, and an independent test
> re-run.
>
> 1. Verified baseline and boundaries: 57 tests passed via
>    `E:\Anaconda\python.exe -m pytest -q`; the system-default Python differs
>    from that environment, so later commands must state the interpreter
>    explicitly. v3 recorded US/CN research runs, health checks, and the
>    fallback framework — engineering and research evidence, not proof of
>    strategy effectiveness or live-readiness. The v3 return comparison did
>    not prove ML incremental value over a simple trend baseline; negative
>    results are retained and re-tuning on the same test set is not a goal.
>    The fallback rule is long/cash and must not be described as a completed
>    margin-capable long-short system. History, fixed pool, survivorship
>    bias, and unverified costs limit extrapolation. This round: local R&D,
>    offline tests, documentation only; no broker connections, no real orders,
>    no approving existing candidates, no overwriting run evidence.
> 2. Findings: (A) fallback state machine — duplicate infra faults could
>    raise illegal transitions; healthy infra auto-recovered while logging
>    "operator-confirmed" without an explicit confirmation event;
>    `primary_healthy` unused; cooldown counted calls not unique sessions;
>    audit "from" taken from history tail; SHADOW_OK lacked evidence
>    conditions; acceptance was a bare boolean; factor-score ranking is not a
>    top-N selection. (B) registry→decision — bundles stored configs/reports
>    while decide re-fit; arbitrary status values; pool-external holdings
>    could break panel indexing; all holdings treated as sellable; no frozen
>    funds; research/decision thresholds must be aligned with evidence.
> 3. Work order and acceptance: A deterministic state transitions with
>    auditable recovery (explicit operator events, restricted observation
>    state, unique-session cooldown, previous-state capture, conservative
>    corruption behavior); B frozen artifacts and offline decision
>    consistency (immutable model/feature contract/thresholds/config/
>    dependency versions/data fingerprints; decide loads only verified
>    artifacts; enumerated statuses and legal transitions; sellable
>    quantities, pending orders, frozen funds; alignment of research and
>    decision features/thresholds); C README, docs trio, pinned dependencies,
>    handover v4 with accurate commands/exit codes/evidence paths; D
>    fallback independent acceptance offline (shared-risk disclosure, cost
>    amplification, delay, missing-data stress; unpolluted holdout or
>    forward shadow; acceptance criteria fixed before experiments).
> 4. ZCode execution requirements: Plan Mode first, then authorized
>    autonomous execution A→B→C, D offline-only; failing tests first, minimal
>    fixes, no unrelated changes, no fabricated approvals, no live features.
> 5. Definition of done: actual changes, passing verification, clear residual
>    limitations, v4 handover. Test counts alone must never be announced as a
>    "complete quantitative system" or "live-ready".

(End of the translated historical plan.)
