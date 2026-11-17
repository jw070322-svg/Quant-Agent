# Improvement plan: research validity before additional factor mining

Prepared: 2026-11-16 (America/Los_Angeles). Repository baseline: `23e7e7c`.
Scope: review and planning; no strategy implementation, approvals, broker actions, commits, or publication performed.

## 1. Source identification and verification

The user confirmed that the intended new handoff is `F:\quant agent\quant_project\README.md`. It was re-read after that clarification and is the primary entry point for this plan. Supporting evidence comes from its linked architecture/decisions/validation documents, `handover/REBUILD_HANDOVER.md` v3.2, current source, and saved H4 results. The source-path ambiguity is resolved.

README-specific follow-through:

- **Portable quick start:** its example updates/ingests CN but then runs the US strategy. Provide separate, internally consistent CN and US paths and a small offline smoke test. Replace the historical as-of date with an explicitly supplied example parameter.
- **Installation:** verify provider dependencies for the advertised update/study commands, not only core offline tests; document optional extras and supported Python versions based on actual clean-environment checks. A minimum Python version alone does not establish compatibility with every later version.
- **Distributed fitting:** distinguish disjoint OOS prediction windows from training windows, which can overlap. Independent worker execution requires immutable input snapshots, validated shard provenance and explicit shared-storage or transfer instructions. Document numerical equivalence with backend/version/tolerance evidence rather than an unconditional identical-predictions claim.
- **Approval examples:** label the commands as conditional on passing eligibility evidence; the README itself states there is no production-ready strategy. Link the enforced approval contract and show a read-only status check before any promotion example.
- **Current status:** preserve the offline-research status, and link the protocol erratum and corrected replay when available. More hardware does not resolve the causal-validity defects identified below.

The handover's statement that Git is absent is stale: the repository has commits `810e763`, `7f48feb`, and `23e7e7c`. The newest commit adds portability and distributed fold fitting. Do not reinitialize Git.

Verification command: `.\.venv\Scripts\python.exe -m pytest tests -q` from the repository root; exit code 0. Independent collection confirms **125 tests**, rather than the handover's 110. A pandas date-parsing deprecation warning remains. Passing tests establish covered behavior, not validity of the research conclusions below.

Test side effect: the checkout was initially clean; the suite regenerated `datasets/manifest_us.json`, changing four normalized-data hashes, and its ingestion path writes normalized CSVs. Calls to real `ingest_market("us")` appear in `tests/replay/test_research_protocol.py:132` and `:143`. The generated manifest was left in place so it remains paired with the regenerated data; restoring only its hashes would create a misleading pair. Test isolation is a prerequisite for future verification.

## 2. Evidence and analysis

### 2.1 Confirmed research defects

| ID | Evidence | Consequence |
|---|---|---|
| R1 | `research/multi_factor.py:128` and `research/cross_section.py:145` select the last valuation row up to signal date plus five days. The subsequent age check does not reject a negative age. | Future valuation records can enter historical signals. A backward tolerance was implemented as a forward lookup. |
| R2 | `research/multi_factor.py:382` uses `score_tables.setdefault(name, concat(...))` inside the date loop. Existing dictionary values are not replaced. | Learned methods retain only their first score window. The saved H4 report confirms 30 IC windows for equal weight and **one each** for ICIR, shrinkage, and ridge. |
| R3 | `research/multi_factor.py:213` divides a factor's maximum observation count by the same maximum. | Nonempty factor coverage is always 1.0; the coverage gate has no discriminatory effect. |
| R4 | `select_factors`, lines 224 onward, tests thresholds without first rejecting nonfinite statistics. | A diagnostic call with NaN IC/ICIR/t-stat and coverage 1.0 returned `selected=['x']`. Undefined evidence can pass screening. |
| R5 | The universe benchmark at lines 402–410 calls the same helper that applies `.head(TOP_N)` at line 291. | The named equal-weight-universe benchmark is actually a tie-selected subset of at most 30 stocks. The comparison does not measure the stated benchmark. |
| R6 | `run_top_n_sim` schedules its callback at signal session +1; `execution/simulator.py` invokes the callback after close and fills pending orders at the following open. | Baseline execution is effectively signal +2 open, inconsistent with factor labels at +1 open. The lag option additionally changes both signal age and execution delay. |
| R7 | Factor-panel construction drops rows without finite future returns before preprocessing and selection (`multi_factor.py:135–139`; analogous cross-sectional paths). | Future outcome availability changes the historical eligible cross-section. Signals and labels are not cleanly separated. |

Additional source-review concerns to reproduce in tests: `FACTOR_DIRECTIONS` values are not applied consistently; some factors are already negated, while screening takes absolute IC and composition uses raw orientations. The last training-window label may mature after the first test decision: splitting on signal date alone is insufficient. The trailing-history slice can contain 13 windows despite the constant being 12. Technical features are recomputed for each symbol/date instead of once per symbol.

### 2.2 Remaining approval and paper-decision gaps

- `registry/store.py:85–100`: legal status transitions exist, but default actor `unknown` and empty evidence are accepted; health/performance requirements are not validated by this function. `save_bundle` can also receive `approved` directly. An audit field alone is not an approval gate.
- `research/decide.py:44–55`: market/risk configurations are loaded from the working tree, while provenance references the bundle; hash comparison skips missing hashes and missing members. Exact historical-source identity also blocks ordinary data extension without distinguishing it from a historical revision.
- `research/decide.py:135`: a pool-external holding can index a missing panel entry. Missing marks are excluded from equity. Line 164 treats all holdings as immediately sellable, and line 191 rounds to decimals rather than market execution units.
- `risk/checks.py:90–100`: each new BUY is checked against the same available balance, without reserving funds for other approved BUYs in the batch. Individual approval is not batch feasibility. Universe eligibility also rejects SELLs for names outside the current selection universe.
- Research uses a selected probability threshold (`runner.py:248–268`); paper inference needs an explicitly frozen, equivalent policy rather than only a frozen estimator.

These are paper-intent limitations, not evidence of real orders: this review did not connect any broker.

### 2.3 Distributed execution and inference quality

`runner.py:599–655` checks fold coverage via threshold files, concatenates prediction files independently, and rebuilds context during finalization. It needs validation of each shard's actual prediction keys, matching data/code/config fingerprints, and model provenance. Current tests exercise synthetic file merging; the broader real-data equivalence claim needs a preserved reproducibility manifest and declared numerical tolerance, especially across CPU/GPU environments.

H6's absent date overlap and empty market control group do not establish that news has no economic effect. They make the specified comparison non-identifiable. Preserve its observed estimates, but distinguish `inconclusive_design` from an adequately powered negative result. Similarly, H4's implementation defects invalidate the intended method comparison; fixing them must not be described as proving learned weights work.

## 3. Prioritized implementation plan

Estimates below are engineering effort, excluding vendor delivery and forward observation time.

### P0-A — Isolate verification and correct the evidence index (0.5–1 day)

1. Move ingestion/replay fixtures under temporary roots; inject storage locations instead of writing repository datasets. Audit other tests for persistent artifact writes.
2. Add a test-suite boundary check: tracked files and protected data/artifact hashes must be unchanged after tests. Network-dependent checks must be explicit and separate.
3. Add an append-only research erratum referencing R1–R7 and the affected H2/H3/H4 modules; preserve original reports. Label H4 method comparison `invalid_protocol_pending_replay`.
4. Update active documentation with the actual commit/test baseline and distinguish completed mechanisms from uncovered contracts. Verify historical pre-registration claims against timestamped artifacts; claims lacking independent ordering evidence remain unverified.

Acceptance: a full isolated test run leaves the checkout and protected artifacts unchanged; every affected historical claim has a linked erratum. Existing raw results remain intact.

### P0-B — Repair the factor research protocol (2–4 days; depends on A)

1. Create a shared Point-in-Time join: `available_at <= signal_at`, backward-only alignment, and maximum permitted age. Preserve vendor availability separately from report period/date; date-only data require a documented conservative convention.
2. Build features/universe without forward labels. Join future outcomes only in the evaluation path. Persist `signal_at`, `entry_at`, `label_end_at`, and missing-outcome reasons.
3. Fit/select only on rows with `label_end_at < decision_at`; apply the same rule at the train/test boundary and every rolling refit. Replace magic maturity offsets with timestamps.
4. Append all method score frames, concatenate once, and assert equal expected decision-date coverage for each method. Persist actual per-date factor weights and missing-score policy.
5. Define coverage as finite factor observations divided by the point-in-time eligible universe on each date; report aggregate and worst-period coverage. Reject NaN/Inf, insufficient windows, and degenerate cross-sections before numerical threshold checks.
6. Define a single direction convention: raw factor definition plus exactly one orientation transform. Economic directions are configured ex ante; learned signs are estimated inside training data only.
7. Give benchmarks their own explicit membership/weight construction. Use the same dates, executable universe, costs, capital constraints and reporting interval; publish any residual cash caused by caps.
8. Submit target weights at the signal-date callback; let the simulator implement next-open fills. Make stale-signal stress and extra execution-session lag independent parameters with independently tested effects.
9. Move feature computation outside the rebalance loop after correctness is established.

Acceptance scenarios:

- Adding or altering rows dated after a signal cannot change its features, membership, preprocessing or score.
- Removing future labels does not change historical selection eligibility.
- A 30-date fixture yields 30 date groups per method, with no duplicates.
- 12 valid values among 20 eligible names yield 0.60 coverage; NaN statistics fail screening.
- With loose caps and 50 eligible names, the universe benchmark includes 50 names while a top-30 strategy includes 30.
- A signal at session t produces the first possible fill at t+1; an additional one-session delay produces t+2, without changing the source signal.
- A training label ending after the first test decision is excluded, even if its signal date belongs to training.

### P0-C — Close approval and portfolio-intent contracts (2–3 days; after A, may be implemented independently of B)

1. Centralize all approval entry points. Require actor, resolvable evidence and passing machine-readable prerequisites; reject direct approved registration. Synthetic plumbing tests use a separate temporary registry.
2. Freeze estimator, features, orientation, policy threshold and configuration together. Verify an integrity manifest before model loading. Record separately authorized runtime risk overrides explicitly.
3. Separate immutable training provenance from inference snapshots: unchanged history plus valid new rows may be permitted; missing required hashes, altered history, schema changes or invalid membership must fail closed. Never set `dataset_drift=false` without complete validation.
4. Accept an account snapshot containing dated lots, sellable quantities, pending orders and marks. Value all holdings independently of the selection universe; incomplete valuation blocks new risk. Permit valid de-risking of former universe members.
5. Reserve funds and sellable shares across the entire intent batch, including fees and existing orders. Validate aggregate exposures and turnover after market-unit rounding. Share this behavior with the simulator.

Acceptance: two individually affordable BUYs whose combined cost exceeds available cash cannot both pass; unknown valuation never understates equity silently; pool-external holdings do not crash; T+1, units, policy equivalence, missing provenance and failed-evidence approval have explicit negative tests.

### P1-A — Reproducible runs and distributed provenance (1–2 days)

1. Persist the resolved dependency environment, code revision, data snapshot identifiers, full parameters, random seeds, device/backend and run manifest. Separate optional provider dependencies from core offline dependencies.
2. Require a shared immutable shard manifest. Validate expected fold/model/date/symbol keys, duplicates, unexpected folds, missing predictions, configuration mismatches and partial files; publish outputs atomically.
3. Finalize against the frozen context. A model trained on a different snapshot must be a separate artifact, not silently joined to old evaluation evidence.
4. Compare serial and distributed runs in the same backend with declared tolerance; CPU/GPU differences receive separately reported tolerances and metrics. Replace absolute equivalence claims where unsupported.

Acceptance: deliberately mixed data/config/code shards are rejected; interruption/retry preserves committed evidence; a serial/distributed fixture matches and records its tolerance; a clean environment can reproduce the offline smoke test.

### P1-B — Corrective replay, then point-in-time universe coverage (1–2 days plus external data lead time)

1. Before viewing corrected results, freeze a bugfix-replay manifest with parent run IDs, defect list, corrected code hash, unchanged economic settings and a bounded run budget. Replaying known data is a correction, not a new holdout or a profitability search.
2. Replay affected H2/H3/H4 studies and publish old/new differences attributable to protocol fixes. Compute statistics on a common test interval and report coverage for every method. Do not select the best test result and present its significance as selection-free evidence.
3. Build a historical-universe adapter with effective-dated membership, listings/delistings, available industry classifications and corporate-action provenance. Synthetic fixtures can be built now; empirical validation awaits adequate historical records.
4. Resolve raw-price/share-unit accounting and effective-dated execution/cost rules before treating A-share capacity or execution estimates as realistic. Adjusted prices may support signals, but executable share quantities require an explicit mapping.

Acceptance: correction reports retain the old evidence and disclose changes; survivorship-biased runs stay exploratory; missing historical membership is explicit rather than synthesized from current constituents. Market-rule values are verified from authoritative sources when implementing this package.

### P2 — Prospective evaluation and optional news research

Start an immutable, observation-only forward capture once data contracts pass; capturing future evidence need not wait for a profitable model. Record forecast timestamps and later label maturity, with no retrospective rewrites. Strategy promotion remains a separate gate.

For news, first repair the comparison design: comparable event/control dates, publication and ingestion timestamps, duplicate-event handling, a held-out materiality-label set, and date-clustered or block-resampled uncertainty. Assess information extraction quality separately from incremental return prediction. A more elaborate LLM classifier is optional and follows these requirements, not a P0 dependency.

Any new strategy experiment needs a bounded trial budget, fixed baselines, untouched evaluation data and a predefined stop rule. Cash and simple trend rules remain transparent reference portfolios. No broker integration is needed to repair the current research validity defects.

## 4. Execution sequence and handover checklist

Order: **P0-A → P0-B and P0-C → P1-A → corrective replay → historical-universe validation → prospective evaluation**. Local implementation can proceed while external universe data are being sourced. Estimated local engineering effort is roughly 7–12 days, subject to tests and the chosen accounting scope; forward evidence duration cannot be compressed into that estimate.

Each package delivers a focused change, behavior tests, an evidence manifest, residual limitations and an updated active handover. Use English for project-owned documentation. Preserve user changes, historical reports and negative results. The current request authorizes this plan, not its implementation or ZCode dispatch.

## 5. Conclusions

The project has advanced beyond the prior 57-test baseline: frozen models, auditable state transitions, provider updates, factor/event studies and distributed fitting now exist. However, the current multi-factor implementation does not support its stated comparison protocol, and some decision/approval guarantees remain documentary rather than enforced.

The next milestone should be **a causally valid, fully covered, reproducible factor-study run with fail-closed decisions**, not a larger factor pool or more GPU workers. Repairing the experiment may change positive or negative findings; it does not establish profitability. The plan is now reconciled with the user-confirmed README entry point; its portability and distributed-computing additions are incorporated above and in P1-A.
