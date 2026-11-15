# ZCode Next-Round Engineering Optimization and Acceptance Plan

Date: 2026-11-12. Scope: `F:\quant agent\quant_project`. Basis:
`handover/REBUILD_HANDOVER.md` v3, current source, and an independent test
re-run.

> **Language note (added 2026-11-15):** this file was originally written in
> Chinese; it is converted to English as part of the project-wide language
> policy. Supersession status: findings (A) and (B) were fixed by the
> 2026-11-15 audit round (F-001..F-005, handover v3.2 §0); work package C was
> completed in the same round; work package D remains partially open (US
> passes stress evidence, CN fails the lag leg, human acceptance pending).

## 1. Verified baseline and boundaries

- Tests re-run via `E:\Anaconda\python.exe -m pytest -q`: 57 passed. The
  system-default Python differs from that environment; later commands must
  state the interpreter explicitly.
- v3 recorded US/CN full research runs, run health checks, and the fallback
  framework — engineering and research evidence, not proof of strategy
  effectiveness or live-readiness.
- The v3 return comparison did not prove ML incremental value over the simple
  trend baseline. Failed/negative results are retained; re-tuning on the same
  test set is not a goal of this round.
- The fallback rule is long/cash and must not be described as a completed
  margin-capable long-short system. Historical sample, fixed pool,
  survivorship bias, and unverified costs still limit extrapolation.
- This round: local R&D, offline tests, and documentation only; no broker
  connections, no real orders, no approving existing research candidates, no
  overwriting existing run evidence. Tests may create isolated temporary
  approval samples.

## 2. Code-review findings (re-verified before implementation)

### A. Fallback state machine: `src/quant_agent/alpha/factor_fallback.py`

1. HALT receiving a duplicate infrastructure fault could raise an illegal
   transition; healthy infra auto-recovered while logging an
   "operator-confirmed" reason with no explicit confirmation event.
2. `primary_healthy` did not participate in decisions; the cooldown counted
   calls, so same-day retries counted as multiple days rather than a valid
   observation period.
3. `handle` derived `from` from the last history entry instead of the actual
   pre-transition state; restarts or truncated history would distort audit.
4. RECOVERY's SHADOW_OK lacked evidence conditions; fallback acceptance was a
   bare boolean without strategy version, data version, validity window, or
   bound validation report; `load` behavior on acceptance state needed
   clarification.
5. Factor-score ranking is not a top-N selection; without a top-N design the
   documentation must state that it is ranking only.

### B. Registry to decision: `registry/store.py`, `research/decide.py`

1. Bundles stored mainly configs and research reports while decide re-fit
   rather than using a frozen model; old config hashes could mix with new
   models or new market risk configs.
2. Status transitions accepted arbitrary status values; approval lacked full
   health evidence and validation gates.
3. Pool-external holdings could trigger panel index errors; holdings without
   a latest price were excluded from equity, understating the risk denominator.
4. All holdings were treated as sellable, with no T+1 sellable quantities,
   no pending-order freeze, and no unified cash reserve; decimal rounding is
   not a board-lot constraint.
5. Research-selected thresholds and decide's prediction filtering must be
   verified consistent, with end-to-end evidence.

## 3. Work order and acceptance criteria

### Work package A: deterministic state transitions and auditable recovery
(priority)

- Define the event table and legal transitions; duplicate faults idempotent;
  infra recovery only clears the fault condition, never fabricates an
  operator confirmation.
- HALT recovery uses an explicit event, reason, actor identity, and evidence;
  first enter a restricted observation state; direct restoration of the risk
  budget is forbidden. Programs must not fabricate real operational approvals.
- Separate event time from trading days; cooldown counts only unique valid
  sessions; retries do not advance it; consistent across days and restarts.
- Include `primary_healthy` as an actual condition; recovery requires
  consecutive qualified observations, shadow validation, and the configured
  acceptance policy — not a single boolean bypass.
- Capture the actual previous_state; store event IDs, before/after states,
  reasons, evidence references, and versions; atomic persistence with
  conservative behavior on corruption/missing fields.
- Acceptance: behavior tests for duplicate faults, unconfirmed recovery,
  duplicate sessions, restarts, persistence corruption, illegal events,
  primary still failing, fallback not accepted. Existing tests keep passing.

### Work package B: frozen artifacts and offline decision consistency

- Distinguish "research run", "model build", and "approved decision
  artifact": training produces an immutable model, feature contract, strategy
  thresholds, config, dependency versions, and input-data fingerprints;
  decide loads only verified artifacts and never re-trains implicitly.
- Old bundles missing a model are explicitly refused or marked research_only;
  silently granting decision eligibility is forbidden; models from untrusted
  sources are not loaded.
- Enumerate statuses and legal transitions; approval verifies integrity,
  health, strategy-acceptance evidence, and hashes. Engineering pass and
  return pass are independent fields.
- The decision holdings pool is independent of the research pool; missing
  prices must refuse new risk and report incomplete valuation, never silently
  omit assets.
- Introduce sellable quantities, pending orders, and frozen funds as inputs;
  the final portfolio checks cash, board lots, gross exposure, single-name
  caps, and turnover limits together; unmodelable inputs are refused.
- Align research and decision features, filter thresholds, timing, and market
  constraints; output traceable rejection reasons.
- Acceptance: build/approve synthetic samples in a temporary registry and run
  the full decide path; cover hash tampering, config drift, revoked/expired
  statuses, stale data, pool-external holdings, missing prices, T+1, pending
  funds, board lots; verify no re-training. Never flip a genuinely failed
  candidate to approved for a demo.

### Work package C: documentation, environment, and verification evidence

- Add a README with the shortest offline flow; add `docs/ARCHITECTURE.md`,
  `docs/DECISIONS.md`, `docs/VALIDATION.md` describing the
  data→research→registry→decision→risk contracts and current capability
  boundaries.
- Pin direct project dependencies and the reproducible install method; record
  actual Python/key-library versions; do not freeze the whole Anaconda
  environment as project dependencies indiscriminately.
- Update the handover to v4, itemizing implemented items, test coverage,
  research conclusions, unverified items, and external blockers; give exact
  commands, exit codes, test results, and evidence paths.
- Do not retroactively bless historical "pre-registrations"; future
  experiments save hypothesis, metrics, budget, sample boundaries, and stop
  rules before running. Historical optimization results are labeled
  exploratory.

### Work package D: fallback independent acceptance (offline, after A/B/C)

- Fix the long/cash rule as one candidate and state its shared data/market
  risks with the primary; a data or execution-infrastructure failure must
  HALT, not switch to a factor strategy that depends on the same bad data.
- Compare trend, fallback, buy-and-hold, and cash benchmarks on the same
  tradable pool, costs, risk budget, and timing; add cost amplification,
  execution-delay, missing-data, turnover/capacity stress scenarios.
- Use an unpolluted holdout or forward shadow run; repeatedly-viewed history
  is diagnostic only. Outcomes may be negative; making money is not the
  completion condition.
- Acceptance criteria fixed before experiments: net return, drawdown, risk
  exposure, turnover, stability, and failover switching losses all reported;
  `fallback_accepted` stays false until independent evidence exists, and the
  budget is never expanded automatically.

## 4. ZCode execution requirements

Enter Plan Mode to read this file, v3, and the relevant source, then output
the concrete changed files, interface migrations, test design, execution
order, and real blockers; then execute A→B→C under the user-authorized
autonomous mode, with D limited to offline validation supported by available
data. Routine implementation choices are delegated; external data gaps are
recorded as unverified and do not block independent local work.

Each work package adds behavior tests reproducing the problem first, then
minimal fixes, then targeted and full verification. Do not modify unrelated
projects, delete existing evidence, fabricate approvals, or add live-trading
features. When the plan conflicts with actual code, revise the plan with
reproducible evidence and leave a record.

## 5. Definition of done

Deliver actual changes, passing verification, clear residual limitations, and
a v4 handover. Test counts alone must never announce a "complete quantitative
system" or "live-ready". This round's goal is decision-source consistency,
deterministic fault handling, and verifiable risk constraints, providing a
reliable basis for subsequent independent strategy validation.
