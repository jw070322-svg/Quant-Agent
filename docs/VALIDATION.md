# Validation and Evaluation Report (VALIDATION)

Current audit note (2026-11-17): see E-004 in `handover/RESEARCH_ERRATA.md` for
the subsequent H4 replay. Its additional direction/coverage/timing corrections
and training-orientation methodology change do not establish a new holdout.
H4 remains NOT PASSED; historical pending-replay annotations below are superseded.

Updated: 2026-11-15. Protocol: handover §8.5; criteria: docs/DECISIONS.md §2
(pre-registered).

## 0.1 Evidence-grade table (E0-E6, fixed in the 2026-11-15 audit round)

| Strategy / module | Grade | Basis | Usage restriction |
|---|---|---|---|
| v1 composite model (GBM + technical features) | **E3** | US/CN full walk-forward offline backtests (net costs / unique OOS / calibration); failed the pre-registered criteria | Research evidence only; no live use |
| v2 factor-pack composite model | **E3** | Same + H1 incrementality rejected (delta-AUC below threshold) | Retired; variant searches forbidden |
| Trend rule (ret_20 > 0, equal weight) | **E3** | Same-simulator benchmark | Not registered as a candidate; revival requires new pre-registration |
| Fallback rule (factor_fallback rule part) | **E3** | US passed three stress legs; CN failed the lag leg; survivorship-biased universe | Design delivery, not accepted |
| fallback state machine | **E2** | 15 unit tests (recovery safety / idempotency / conservative corruption failure); not wired to live health sources | Sound structure ≠ runtime validation |
| H2 value/size factor screening | **E3** | 60-window cross-sectional screening (survivorship bias) | Screening record only |
| H3 technical indicator attribution | **E3** | 74-window per-indicator attribution (same) | All failed the screen |
| H5 critical-information-point monitor | **E3** | Real-data module validation (13 dual-confirmed points) | Information events, not strategy signals |
| **Any strategy at live/production readiness (E4+)** | **none** | No held-out forward validation, no shadow run, no historical constituents, no broker/account authorization | **No production-ready strategy exists in this project** |

## 0. Data update chain (added 2026-11-14)

| Item | Conclusion |
|---|---|
| Source inventory | akshare 1.18.54 built in (token-free) + tushare (needs token, unused); Yahoo/Eastmoney/Sina/PyPI reachable |
| CN source | akshare `stock_zh_a_daily` (Sina): token-free, volume already in shares, qfq variant provides back-adjusted prices |
| US source | akshare `stock_us_daily` (Yahoo-backed): token-free, raw + qfq |
| Adjustment factors | Derived per day from the qfq/raw close ratio (anchored to latest = 1, matching the pipeline convention) — closes the legacy "CN adjustment unknown" manifest gap; all symbols now `raw_with_adj_factor` |
| Network notes | Windows system proxy intermittently fails Eastmoney endpoints (ProxyError); the provider bypasses via NO_PROXY with backoff retries; Sina/Yahoo are stable. News: `stock_news_em` (Eastmoney) verified live 2026-11-15 for recent items; `news_cctv` archive works for historical dates |
| Data volume | US 4 × 1979 rows (2019-01-02 → 2026-11-13, uniform 7.4-year history; legacy AAPL/AMZN/GOOGL had only 508 rows); CN 6 × 1349-1907 rows |
| Quality gate | Provider data: 0 hard quarantines; the three legacy OHLC defect rows do not reproduce in provider data (reconciliation test stays anchored to the legacy files) |
| Freshness | Ingest age = 1 day (≤ 7-day tolerance; a yesterday bar is no longer misjudged as stale) |
| Provenance | Manifest records source_kind = provider_update / provider / fetched_at / sha256; per-symbol fetch failures are disclosed and overwrite nothing |

Update commands: `qa update --market us|cn` (writes `datasets/updates/`) →
`qa ingest --market ...` (prefers the update layer automatically).

## 1. Protocol essentials (enforced in code, guarded by tests)

- Folds are cut on session DATES; OOS windows are disjoint (constructor
  assertion + health duplication check).
- Fitting uses only labels **matured before the fit cutoff** (test:
  `test_every_fitting_label_matured_before_fit_cutoff`).
- Per OOS window: the first half (tune) selects the policy threshold via
  mini-simulations; the second half (report) is evidence; tune windows run
  the previous fold's threshold or the pre-registered default (test:
  `test_report_window_policy_selected_strictly_before`).
- Strategy and all benchmarks run through the SAME simulator with the SAME
  costs: capped equal-weight buy-and-hold, trend rule, fallback rule (study).
- Metrics: net compounded return (compounded simple returns), drawdown with a
  high-water mark including initial equity, turnover, exposure, costs,
  completed-trade win rate, calibration (Brier/AUC/base rate), dependence-aware
  block-bootstrap Sharpe CI.
- Health fails (unusable data / duplicate OOS / ledger mismatch) →
  `complete_run_claim = false`.

## 2. Results (2026-11-14 data refresh; OOS 2022-12 → 2026-11; health 0 fail;
not a profitability claim)

**US** (run `us_research_v1-us-0b5b882ff6a5a732-2026-11-14T111347Z`, uniform
7.4-year history):

| Portfolio | Total return | Sharpe | MaxDD |
|---|---:|---:|---:|
| Primary strategy (GBM) | **+12.9%** | 0.80 (bootstrap CI −0.17…1.72) | — |
| Capped equal-weight buy-and-hold | −4.00% | — | — |
| Trend rule | **+50.9%** | — | — |
| Fallback rule (study, unvalidated) | +20.3% | 0.92 | −5.7% |

Report-window-only: +9.0% (Sharpe 1.05, n = 480); completed-trade win rate
56.2% (318 trades); calibration n = 3840: **AUC 0.500**, Brier 0.2530
(constant base-rate predictor ≈ 0.2486).

**CN** (run `cn_research_v1-cn-59be59f53bf89524-2026-11-14T111424Z`):

| Portfolio | Total return | Sharpe | MaxDD |
|---|---:|---:|---:|
| Primary strategy (GBM) | **−11.6%** | −0.35 | — |
| Capped equal-weight buy-and-hold | −0.42% | — | — |
| Trend rule | **+89.4%** | — | — |
| Fallback rule (study, unvalidated) | +36.2% | 1.02 | −8.1% |

Report-window-only: −15.9% (Sharpe −1.14); calibration n = 5520: **AUC 0.494**.

### Against the pre-registered criteria (docs/DECISIONS.md §2.1)

| Criterion | US | CN |
|---|---|---|
| Calibration AUC ≥ 0.53 | ✗ 0.500 | ✗ 0.494 |
| Brier better than the constant | ✗ (0.2530 > 0.2486) | ✗ |
| Report-window net return > 0 and beats buy-and-hold | ✓ (+9.0% vs −4.0%) | ✗ (−15.9%) |
| Bootstrap Sharpe CI lower bound > 0 | ✗ (−0.17) | ✗ (−1.25) |
| **Conclusion** | **Not passed** | **Not passed** |

**Verdict: no approvable bundle.** The US improvement mainly comes from the
longer evaluation window (which contains the trending period). One honest
mechanism note: AUC ≈ 0.50 in both markets — the classifier's probability
carries no discrimination; the US positive return comes from the conditional
payoff estimates and risk caps rather than probability quality, consistent
with the calibration gate failing. The trend rule dominates everywhere.

## 2.1 Factor mining H1 (pre-registered 2026-11-15, adjudicated same day)

Hypothesis and criteria: docs/DECISIONS.md §2.4 (registered before any run;
trial budget: 2 registered runs).

| Run | Total return | Sharpe | Report window | RW Sharpe | AUC | Bootstrap Sharpe CI |
|---|---:|---:|---:|---:|---:|---|
| US v1 (baseline) | +12.9% | 0.80 | +9.0% | 1.05 | 0.5003 | −0.17…1.72 |
| **US v2 (+8 factors)** | **+26.1%** | **1.16** | **+13.8%** | **1.09** | 0.5061 | **0.51…1.77** |
| CN v1 (baseline) | −11.6% | −0.35 | −15.9% | −1.14 | 0.4940 | −1.25…0.57 |
| CN v2 (+8 factors) | −10.2% | −0.26 | −11.3% | −0.62 | 0.4951 | −1.21…0.82 |

**Verdict: H1 rejected** — delta-AUC (US +0.006 / CN +0.001) is far below the
pre-registered +0.02 threshold; the v2 pack is retired under the rules. The
US v2 return improvement (CI lower bound positive for the first time) fails
the calibration gate and stays a candidate; it must be read against the full
search history of six registered runs. See DECISIONS.md §2.4 and the runs'
trials.jsonl.

## 2.2 Cross-sectional factor screening H2 (pre-registered 2026-11-15, — **2026-11-17 ERRATUM: invalid_protocol_pending_replay (R1/R6/R7); see handover/RESEARCH_ERRATA.md E-001** —
adjudicated same day)

Data chain: akshare Sina daily bars (150 CSI-300 current constituents, seeded
sample n = 150) + Baidu valuation history (PE_TTM/PB/market cap, ~2-day
sampling, 5-day as-of alignment) + CSI constituents interface. 60 monthly
windows (2021-11 → 2026-11), average cross-section 135 names. Survivorship
bias → exploratory screening, cannot approve anything. Results (net costs,
next-open fills, same simulator): **no factor passed the full screen**; size
was the only IC-screen pass (NW t −2.82, FM t −2.75, net spread +37.2%) but
the quintiles were non-monotone. Report:
`artifacts/runs/xs_cn_h2-2caccc9508e3f8f2-2026-11-15T105527Z/`.

## 2.3 Technical indicator attribution H3 (pre-registered 2026-11-15, — **2026-11-17 ERRATUM: invalid_protocol_pending_replay (R6/R7); see handover/RESEARCH_ERRATA.md E-001** —
adjudicated same day)

Reused the H2 universe (150 names); 16 shared-library technical indicators as
cross-sectional factors + an equal-weight TA composite; 74 monthly windows
(average cross-section 145 names), same simulator/costs. **None of the 17
candidates passed the full screen**: all IC screens failed (ret_1 rejected on
direction — the sample shows 1-day momentum, not reversal); FM-significant and
direction-consistent were only bb_pos20 (t −2.05) and vol_5 (t −2.74), but
their net-cost quintile portfolios failed.
Corrected replay (run `xs_ta_h3-2caccc9508e3f8f2-2026-11-17T153427Z`):
conclusions unchanged — none of the 17 candidates passes; ret_1 still shows
1-day momentum (direction-rejected); FM-consistent bb_pos20/vol_5 still fail
the net-cost portfolios. Composite-model cross-reference (GBM on the same
features, AUC ≈ 0.50). Pre-fix report preserved:
`artifacts/runs/xs_ta_h3-2caccc9508e3f8f2-2026-11-15T110412Z/`. Adjudication:
DECISIONS.md §2.6.

## 2.4 Multi-factor synthesis H4 (pre-registered 2026-11-15; **2026-11-17 ERRATA E-001/E-003: R2/R3/R5/R6 + two-sided orientation; CORRECTED REPLAY completed** — adjudicated same day)

Train/test windows split in time (30/30), 14 factors with MAD winsorization +
size neutralization, four-gate train-window screening (selected:
BP/amihud_20/turnover/vol_5), four synthesis methods with TOP_N = 30 on the
test windows (same simulator, net costs).
Pre-fix run (labelled invalid protocol, preserved): "equal-weight +12.6% vs
benchmark +18.6%, learned weights all negative" — that benchmark was a
30-name subset (R5) and the learned-weight tables retained only one window
(R2).
Corrected replay (run `xs_mf_h4-2caccc9508e3f8f2-2026-11-17T154618Z`):
screening re-selects BP/amihud_20/turnover/vol_5 (coverage now real:
93-100%); test-window synthesis **equal-weight +31.1% (Sharpe 0.78) and Ridge
+22.9% both BEAT the corrected universe benchmark +17.9%**; robustness passes
(costs ×4 +23.1%, lag +30.5%); criteria a/c/d pass. **Criterion (b) still
fails: test-window composite RankIC significance not reached → NOT PASSED per
the pre-registered criteria; promotion refused.** The flawed run's "all
learned weights negative" claim did not survive the corrections; the
improvement is attributed to the protocol fixes, not to a new discovery.
Weight learning: Ridge beats the universe benchmark but trails equal-weight synthesis and shows negative test
RankIC — still refused for promotion. Reports: pre-fix
`...2026-11-15T114232Z/`, corrected `...2026-11-17T154618Z/`. Adjudication:
DECISIONS.md §2.7.

## 2.5 Critical-information-point monitor H5 (pre-registered 2026-11-15,
module validation passed)

Deal-amount anomaly detection (rolling 30-session median/MAD, Z ≥ 3 or
|day-over-day| ≥ 2) × news correlation (per-stock `stock_news_em`
entity-matched feed + market `news_cctv` macro keyword archive), dual-requirement
flags, quant-only anomalies confined to the watchlist. Full-history scan of
150 symbols → 9556 anomaly clusters; market tier 8/8 and stock tier 5/10
dual-confirmed (including 2024-09-24, z = 12.1, the policy day); JSON contract
and reasoning-first ordering enforced by tests. **Disclosed**: the market CCTV
keyword leg has low specificity; the stock entity feed is the high-specificity
leg. Report: `artifacts/runs/xs_info_h5-*/`. Evidence grade E3 (real-data run);
flags are information events, not strategy signals.

## 2.6 Forward validation of critical information points H6 (pre-registered
2026-11-15, adjudicated same day)

Forward outcomes for 9548 anomaly clusters (reaction return / next-session
o2o / volume persistence / 5-session drift) + a news subsample (60 fetches →
24 dual / 16 quant-only) + CCTV verification of all 39 market clusters.
**Verdict: the news leg shows no measurable incremental forward information** —
the pre-registered paired test is structurally infeasible (zero date overlap
between the two groups), the market leg flagged 39/39 (empty control group),
and the exploratory two-sample tests were all insignificant. Recorded findings:
volume persistence 210% (t 73.5) and 5-session drift +0.42% (t 2.62), unadjusted.
The monitor stays review-only (E3). Report:
`artifacts/runs/xs_ev_h6-f572abb6b9693be3-2026-11-15T134803Z/`. Adjudication:
DECISIONS.md §2.9.

## 3. Root-cause evidence (CN calibration, n = 5520 matured OOS predictions,
2026-11-14 data)

- **AUC = 0.494**: out-of-sample discrimination below a coin flip (US: 0.500).
- **Brier = 0.2551**: worse than the constant "always predict base rate
  0.4754" predictor.
- Per-fold "optimal thresholds" jump without structure: selection on noise.
- Conclusion: the classifier's probabilities carry no tradable information;
  the CN strategy's loss comes from noise trading costs; the US positive
  return comes from conditional payoff estimates and risk caps, not
  probability quality.

## 4. Fallback-rule stress study (evidence, not acceptance; 2026-11-14 data)

| Variant | US | CN |
|---|---:|---:|
| Baseline | +20.3% (Sharpe 0.92) | +36.2% (Sharpe 1.02) |
| Costs ×4 | +13.5% ✓ | +20.9% ✓ |
| Decision lag +1 session | +29.9% ✓ | **−1.2% ✗** |
| Daily-return correlation with the primary strategy | 0.238 (<0.6 ✓) | 0.096 (✓) |
| Both-negative days | 11.9% | 11.2% |

Per docs/DECISIONS.md §2.2: US passes the cost/lag/correlation stress evidence
(still requires whipsaw review + a human decision); **CN fails the lag stress**
(net return turns negative — the signal decays within one session).
`fallback_accepted` stays false. Full reports:
`artifacts/runs/*-fallback-study/stress_report.md`. A 2026-11-15 re-run on the
refreshed data confirmed the same pattern (US +13.5% at costs ×4 and +29.9%
under lag; CN −1.2% under lag).

## 5. Mechanism verification (not a performance claim)

- 110 tests pass: hand-calculated reconciliation, zero-trade preservation,
  partial fill + cash-shortfall cancellation, stops/trailing/gaps, T+1,
  corporate actions, ledger event replay, order idempotency and state
  machine, time leakage, OOS uniqueness, optimizer constraints, fallback rule
  and switching state machine, decide integration (unapproved refusal /
  holdings-driven SELL intents / staleness blocking / tampered-artifact
  refusal / drift refusal), provider factor derivation and source preference,
  monitor rules and JSON contract, materiality tiers, event-outcome
  computation.
- Ingestion quarantine reproduces the audit evidence line by line (AMZN
  2026-02-04/02-12, GOOGL 2026-02-02).
- Registry approval gates: arbitrary status strings, illegal transitions,
  tampered model artifacts, and dataset drift are all refused (tests).

## 6. Conclusions and boundaries

1. **No incremental value of the ML primary strategy over simple rules is
   established on the bundled or refreshed data** (consistent across US/CN,
   and across H1-H4). Under the pre-registered criteria: no approvable
   bundle; cash/paper mode retained; no re-opening of the same hypothesis.
2. The trend rule performs well but is **not registered** — registering it as
   a candidate requires a new pre-registration and the same gates.
3. The monitor (H5) is a validated **information-review tool**; its news leg
   adds no measured forward value (H6), so flags are review input only.
4. Cost parameters are UNVERIFIED; the pool is fixed (survivorship bias);
   data are historical snapshots. Any move toward paper/live trading must
   first pass DECISIONS §2.3.
5. Reproduction: `qa ingest --market X` → `qa research --strategy
   X_research_v1` → `qa health --run <id>`; factor studies: `qa xs --family
   valuation|technical`, `qa multifactor`, `qa infomonitor`, `qa eventstudy`.

## 2.7 Factor bundle evaluation H7 (pre-registered 2026-11-18, adjudicated
2026-11-20)

Five frozen strategies (MOM/ATO/QV/FAR/VPA) evaluated on the assembled CN
canonical snapshot (467 CSI 300+500 names, 57 monthly observations, raw-share
execution with T+1, board lots, corporate actions, capacity limits).

| Strategy | Months | Mean annual active | Economic pass | Raw p | Holm p | Gate |
|---|---:|---:|---|---:|---:|---|
| MOM | 57 | +0.41% | ✓ | 0.4253 | 1.0 | ✗ |
| ATO | 57 | −0.63% | ✗ | 0.5549 | 1.0 | ✗ |
| QV | — | blocked (financials coverage) | — | — | 1.0 | ✗ |
| FAR | — | blocked | — | — | 1.0 | ✗ |
| VPA | — | blocked | — | — | 1.0 | ✗ |

**Verdict: H7 rejected — no strategy passes the family-corrected gate.**
MOM shows positive net active return surviving costs (economic_pass=True) but
lacks statistical significance (Holm p = 1.0). QV/FAR/VPA are blocked on
financials coverage (insufficient complete annual statement triples for the
80% gate). The US slots remain blocked (no US snapshot).

Data provenance: canonical snapshot assembled token-free (Sina bars with
outstanding_share → float; Sina annual financials with 公告日期 as
available_at; Eastmoney fhps corporate actions; csindex universe).
Survivorship bias disclosed.

Report: `artifacts/factor_bundles/cn-20261120T160534604683Z/`;
family: `artifacts/factor_bundles/family_h7_cn.json`.
