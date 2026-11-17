# Architecture

Project: `quant_agent` rebuild (implemented subset of the target architecture in
handover §8). Status: Phases 0-3 completed and verified; Phases 4-5 not entered
(gated by acceptance criteria).

## 1. Module map and contracts

```text
src/quant_agent/
  data/        Ingestion and validation: schema normalization, OHLC quarantine,
               volume-unit inference, adjustment handling, manifests.
               providers.py Market-data update sources (akshare: CN = Sina,
               US = Yahoo; news = Eastmoney per-stock + CCTV archive).
               Contracts: unusable symbols block complete-run claims; invalid
               rows are quarantined, never repaired; everything is hashed;
               the update layer datasets/updates/ takes precedence over legacy
               snapshots; provider failures are reported per symbol and
               overwrite nothing; CN adjustment factors are derived per day
               from the qfq/raw close ratio (closes the legacy "CN adjustment
               unknown" gap).
  features/    The single shared causal feature library (24 dimensionless
               features, direction-normalized for factor studies).
               Contract: prefix causality (appending future rows never
               changes past values); warm-up is NaN, never zero.
  labels/      Executable forward returns with maturity masks.
               Contract: enter at t+1 open / exit at t+2 open; unknown labels
               are NaN and are never folded into fitting.
  research/    runner (orchestrator), splits (date-grouped folds), models,
               metrics, trials, cross_section (H2/H3), multi_factor (H4),
               event_study (H6), info_monitor (H5).
               Contracts: OOS windows are disjoint; fitting labels must have
               matured before the fit cutoff; every trial is recorded.
  alpha/       Conditional payoff shrinkage estimation
               (p*E[r|event] + (1-p)*E[r|complement]) and factor_fallback
               (unvalidated design delivery).
               Contracts: rule scores are never probabilities; evidence is
               estimated on training folds only.
  portfolio/   Constrained portfolio construction (capped proportional
               water-filling).
               Contract: feasible by construction, never rescaled post hoc;
               cash is a first-class output.
  risk/        Independent risk checks.
               Contract: pure functions; staleness blocks BUYs only
               (de-risking is always allowed); every rejection carries a reason.
  ledger/      Event-sourced account (cash / FIFO lots / T+1 sellable /
               high-water marks / audit events).
               Contract: fully in adjusted space; corporate actions are audit
               events, not lot mutations; state is rebuildable from the event log.
  execution/   Order state machine, PaperBroker, event-driven simulator
               (with an optional precomputed bars cache for speed).
               Contract: next-open fills with slippage/participation caps/
               lots/cash re-check; decision_fn contract is three-state
               (None = HOLD, {} = go flat, dict = rebalance to target).
  registry/    Immutable bundle registry (candidate/approved + approval history).
               Contracts: silent overwrite refused; unapproved bundles cannot
               enter decide; frozen model.pkl artifact with SHA-256 verified
               at decision time; dataset drift is rejected.
  monitoring/  Health checks (data, ledger reconciliation, OOS uniqueness,
               calibration); a fail blocks any complete-run claim.
  agent/       Read-only typed tools and reports (no trading authority;
               unsupported symbols fail explicitly).
  cli.py       qa update|ingest|research|report|health|bundles|decide|
               fallback-stress|xs|multifactor|infomonitor|eventstudy
```

## 2. Data flow and execution timeline

```text
qa update (akshare, token-free: CN = Sina qfq/raw -> adjustment factors,
           US = Yahoo; news = Eastmoney per-stock + CCTV archive)
  -> datasets/updates/{market}/*.csv (simple schema + amount + adj_factor,
     with an update summary for provenance)
  -> qa ingest (update layer preferred over legacy snapshots; quarantine /
     validation / normalization / manifest with provenance)
  -> feature panel -> labels
  -> walk-forward folds (fitting uses only matured labels; the tune half of
     each OOS window selects policy; the report half is evidence)
  -> activation schedule (report windows use the fold's own threshold; tune
     windows use the previous fold's threshold or the pre-registered default)
  -> expected returns (net of costs) -> constrained weights -> independent
     risk checks on order intents -> next-open fills -> continuous ledger
     -> metrics / health / report / registry
```

Per-session simulator loop:
1. Fill pending orders at the OPEN (slippage, participation cap, lots, cash
   and T+1 re-checks, frozen pending-buy notional against new intents).
2. Intraday stops (a breach is an exit intent; fill price = min(open, stop);
   gaps fill at the open).
3. Mark to market at CLOSE; update per-lot high-water marks.
4. After close: decision_fn -> target weights; deltas -> intents -> risk
   checks -> next-session orders.

## 3. Agent boundary

- `agent/tools.py` provides read-only access to immutable run artifacts;
  unknown run/symbol/date inputs fail explicitly and are never substituted.
- No code path in the agent layer can place orders, change risk limits, or
  write to run directories.
- Any future LLM narration may only consume verified report text and is
  treated as untrusted input.

## 4. Research studies registry (pre-registered, adjudicated)

| Hypothesis | Question | Verdict |
|---|---|---|
| H1 | Incremental value of the v2 factor pack | Rejected (delta-AUC below threshold in both markets) |
| H2 | Cross-sectional value/size/momentum/low-vol/liquidity factors | No factor passed the full screen (size passed IC only) |
| H3 | Technical indicator attribution | No indicator passed the full screen |
| H4 | Multi-factor synthesis and TOP_N selection | Rejected (equal-weight synthesis underperformed the universe benchmark; weight learning rejected) |
| H5 | Critical-information-point monitor | Module validation passed (13 dual-confirmed points) |
| H6 | Forward value of the news leg | No measurable incremental forward information |

Details: docs/VALIDATION.md; adjudications: docs/DECISIONS.md §2.4-§2.9.

## 5. Portability and distributed execution

- **Paths**: everything resolves relative to the repository root — no absolute
  paths in code or configs. Data caches live under `datasets/` and are
  regenerable from public sources via `qa update` + `qa ingest`.
- **Install**: `pip install -e ".[dev]"` on Python >= 3.11 (validated in a
  fresh virtual environment); `requirements.lock` records the original
  development environment for reproducibility audits.
- **GPU**: the XGBoost device is configurable (`model.gbm.device` in strategy
  configs). With `device="cuda"` fits run on NVIDIA GPUs (verified on an RTX
  5060); an unusable device falls back to the CPU path automatically at fit
  time.
- **Swarm**: the expensive phase — per-fold fitting — is embarrassingly
  parallel because folds are disjoint by construction. `qa research
  --fit-shard i/n --run-id <shared-id>` fits only `fold_id % n == i` on each
  worker and writes shard outputs into the shared run directory;
  `qa research-finalize` merges the shards (duplicating or missing folds fail
  loudly) and runs the sequential evaluation on one machine. Merged
  predictions are identical to an unsharded run (equivalence validated on
  real data).

## 6. Known boundaries (honest list)

- The simulator runs on daily OHLC: stops trigger on the session low, no
  intraday path; suspended symbols leave orders pending until expiry.
- N5: the participation cap uses the COMPLETED session's volume — an ex-post
  capacity assumption, not an observable opening-liquidity limit; a stop
  threshold does not guarantee fills at that threshold (event-level data
  required to model real liquidity; deferred per plan §4.5).
- Intent sizing uses the prior close; actual fills use the next open; a gap
  can cancel a BUY at fill time (explicit contract, no squeezing).
- Participation caps depend on the inferred volume unit (recorded in the
  manifest); the US simple-schema feed carries no amount column, so its
  volume unit is recorded as assumed.
- Price limits are monitoring-only in v1 (effective-dated rules unverified).
- The fixed pool carries survivorship bias; expanding the universe requires
  historical constituents.
- The market-level CCTV news leg has low specificity (a 10-20 item daily
  bulletin almost always hits a macro keyword); the stock-level
  entity-matched feed is the high-specificity leg.
