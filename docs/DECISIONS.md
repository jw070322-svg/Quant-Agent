# Design Decisions and Pre-Registered Criteria (DECISIONS)

This document records the design decisions already made, their rationale, and
the **pre-registered** acceptance criteria. The criteria were written down
before the selections they constrain, to discipline future searches and
activations. All adjudications are in English; two historical input documents
are preserved in their original language as immutable evidence.

## 1. Fixed design decisions

| Decision | Rationale | Handover basis |
|---|---|---|
| Label = enter at t+1 open / exit at t+2 open (configurable) | After-close decision → next open is executable; CN satisfies T+1 sellability by construction | §8.4 |
| Three-state decision contract: `None` = HOLD / `{}` = go flat / `dict` = rebalance to target | Distinguishes "do nothing" from "exit", otherwise the buy-and-hold benchmark gets liquidated daily (observed in testing) | — |
| Threshold tuning on the first half of each OOS window; tune windows run the previous fold's threshold or the pre-registered default 0.55 | A reported date never participates in its own selection; removes first-fold pollution | §4.5/§8.5 |
| Portfolio via capped proportional water-filling, no solver | Eliminates the entire class of "solve a feasible problem, then break it by normalizing" errors | §6.4/§8.6 |
| Expected return = p·E[r\|event] + (1−p)·E[r\|complement], estimated in-fold with shrinkage | p is not a return; the complement of p is not a short signal (project is long-only) | §4.4/§8.4 |
| All prices in adjusted space; corporate actions are audit events | Adjusted series are continuous by construction; rescaling lots fabricates cost basis and misfires stops (observed in testing) | — |
| Staleness blocks BUYs only | A data incident blocks new risk but must never trap existing exposure | §8.6 |
| Stop fill price = min(open, stop level) | A gap through the stop fills at the open; a threshold breach is an intent, not a guaranteed fill | §8.7 |
| Price limits are monitoring-only in v1 | Hard-coding unverified effective-dated rules is riskier than flagging | §5.4/§10 |
| Extreme single-session moves (>35%) are kept and flagged | Could be a genuine move or an adjustment discontinuity; left for reconciliation | §3.1 |
| The simulator is the single source of truth (research/replay/paper share it) | One feature/signal/pricing/risk code path | §8.1 principle 2 |
| Per-market manifests (`manifest_{market}.json`) | Two markets coexist without clobbering each other | §8.1 |

## 2. Pre-registered criteria (written before the selections)

### 2.1 Gate for approving a strategy bundle (Phase 3→4)

A candidate must satisfy ALL of:

1. **Complete-run claim**: health has zero fails (data usable, OOS unique,
   ledger reconciled).
2. **Calibration**: OOS AUC ≥ 0.53 and Brier better than the constant
   base-rate prediction (CN currently: AUC 0.494 / Brier 0.2551 — **fails**).
3. **Net performance**: report-window-only net return and Sharpe positive,
   beating the capped equal-weight buy-and-hold benchmark (same costs, same
   execution).
4. **Uncertainty**: block-bootstrap Sharpe 95% CI lower bound > 0.
5. **Robustness**: net return stays positive at costs ×2 and decision lag
   +1 session.
6. **Discrimination evidence**: not concentrated in a few trades (top-5 trade
   profit share < 50%), and turnover within budget.

Current status: **US and CN both failed → no approvable bundle → Phase 4 is
closed.** Re-opening the search because the result is unfavorable is
forbidden; a new hypothesis requires a new registered trial budget.

### 2.2 Gate for FALLBACK mode (factor_fallback)

1. The stress study (`qa fallback-stress`) shows: net return > 0 at costs ×4
   AND under a +1-session decision lag.
2. Daily-return correlation with the primary strategy < 0.6, with the
   "both-negative days" share explained in the report (high co-movement means
   the fallback does NOT diversify the primary failure mode).
3. Whipsaw/regime review (worst-drawdown windows, exposure sign flips, worst
   10 sessions) passes a recorded human review.
4. An explicit human acceptance decision is written into the registry history;
   `fallback_accepted = True` can only be set by that decision.
5. Switching follows the state machine: only model degradation may degrade
   the primary; infrastructure faults → HALT (never an excuse to switch
   strategies); recovery requires shadow validation and a cooldown.

2026-11-15 evidence: US passes costs ×4 (+13.5%), lag (+29.9%), correlation
(0.238); CN fails the lag stress (−1.2%). Acceptance remains PENDING HUMAN
REVIEW; trending-window returns are not acceptance evidence.

### 2.3 Entry conditions for Phase 4/5

- A bundle passing §2.1 exists, OR an explicit decision registers the trend
  rule as the primary strategy through the full process (it is currently only
  a benchmark).
- Paper operation ≥ one full rebalance cycle with zero reconciliation
  differences.
- Account authorization, effective-dated market rules, cost schedules, and
  risk limits all configured and confirmed by the account/risk owner.

### 2.4 Hypothesis H1: incremental value of the v2 factor pack (pre-registered
2026-11-15, before any run)

**Hypothesis**: beyond the 16 v1 dimensionless price/volume features, the
following 8 causal, dimensionless factors carry tradable predictive
information for the o2o label (t+1 open in / t+2 open out) beyond the v1 pack:

| Factor | Definition (all causal) | Family |
|---|---|---|
| mom_12_1 | close.shift(20)/close.shift(240) − 1 (12-month momentum excluding the last month) | Momentum structure |
| dist_52w_high | close / rolling_max(close,252) − 1 | Distance from high |
| skew_20 | 20-session skewness of ret_1 | Return distribution |
| kurt_20 | 20-session excess kurtosis of ret_1 | Return distribution |
| vol_of_vol_20 | std(std(ret_1,5),20) | Volatility structure |
| amihud_20 | mean(\|ret_1\|/(close×volume)) ×1e8 (20 sessions) | Illiquidity |
| corr_pv_20 | 20-session correlation of ret_1 and log-volume change | Price-volume |
| vol_ratio_60 | volume / MA60(volume) | Volume |

**Trial budget (locked)**: exactly 2 registered runs — `{us,cn} × factor_v2`.
Model hyperparameters, label, walk-forward parameters, threshold grid identical
to v1, seed = 42. No additional feature selection or parameter search.

**Incrementality criteria (H1 passes iff all hold)**:
1. v2 OOS AUC − v1 OOS AUC ≥ **+0.02** in BOTH markets simultaneously;
2. v2 report-window net return is not below v1 in either market;
3. v2 itself meets every §2.1 gate (calibration/CI/benchmark).

**If H1 is rejected**: the v2 pack retires entirely; no variant searches on
this pack; any standout factor requires a new minimal-pack pre-registration.

**Baseline**: v1-pack registered runs on the 2026-11-14 data
(`us_research_v1-us-...-2026-11-14T111347Z`, AUC 0.500, report window +9.0%)
and (`cn_research_v1-cn-...-2026-11-14T111424Z`, AUC 0.494, report window
−15.9%).

**Adjudication (2026-11-15, backfilled after the runs)**: H1 **rejected**.

| Criterion | Requirement | US v2 | CN v2 |
|---|---|---|---|
| ΔAUC ≥ +0.02 (both markets) | ✓ | +0.0058 ✗ | +0.0011 ✗ |
| Report-window net return not below v1 | ✓ | +13.8% (v1 +9.0%) ✓ | −11.3% (v1 −15.9%) ✓ |
| §2.1 calibration gate AUC ≥ 0.53 | ✓ | 0.506 ✗ | 0.495 ✗ |

- Criterion 1 fails in both markets → the v2 pack retires entirely under the
  pre-registered rules; variant searches on this pack are forbidden.
- Any single-factor revival requires a new pre-registration with a minimal
  pack containing only that factor.
- Honest observation: US v2 returns improved materially (+26.1%, Sharpe 1.16,
  bootstrap CI lower bound above zero for the first time at 0.512, report-window
  Sharpe 1.09) while AUC barely moved — again consistent with returns coming
  from conditional payoff estimates and risk caps rather than probability
  discrimination; not approvable under the criteria (US v2 bundle stays
  candidate).
- Multiple-comparison warning: this result carries the search history of all
  six registered runs; the positive US v2 CI is not a "new discovery" and must
  be confirmed by prospective/paper evidence.

### 2.5 Hypothesis H2: cross-sectional value/size/momentum/low-vol/liquidity
factors (A-shares, pre-registered 2026-11-15, before any return computation)

**Data sources (verified live)**: akshare — Sina daily bars (raw + qfq →
adjustment factors), Baidu Gushitong per-symbol valuation history (PE_TTM, PB,
market cap, ~2-day sampling), CSI index constituents (current membership).

**Universe**: 150 symbols sampled with seed = 42 from the CSI 300 **current**
constituents (2026-11-11). **Known limitation (declared at registration)**:
no historical constituents → survivorship bias is unavoidable; this study is
**exploratory screening** and its results, however good, can **never approve**
a bundle (§2.3 preconditions unmet).

**Factor definitions (t-date cross-section, all using information ≤ t; per
date winsorize at 1%/99% then cross-sectional z-score)**:

| Factor | Definition | Expected direction |
|---|---|---|
| EP | 1/PE_TTM (loss-making PE < 0 → EP < 0, retained) | + |
| BP | 1/PB (PB ≤ 0 excluded) | + |
| size | ln(market cap [100M CNY]) | − |
| mom_1m | close[t]/close[t−20] − 1 (adjusted) | − (A-share reversal) |
| mom_12m_1m | close[t−20]/close[t−240] − 1 | + |
| lowvol | −vol_20 (20-session daily-return std) | + |
| reversal_5 | −(close[t]/close[t−5] − 1) | + |
| turnover | ln(traded value proxy / market cap) | two-sided |

**Evaluation protocol**: rebalance every 20 sessions (~monthly), windows
non-overlapping; next-open entry, exit at the window-end open (same executable
convention as the pipeline). Forward return: 20-session o2o adjusted return;
valuation factors use the latest value ≤ rebalance date within 5 calendar
days. Coverage gate: a symbol-date must have (a) factor values, (b) ≥ 240
sessions of history (for mom_12m_1m), (c) a complete forward return.
Metrics: Spearman RankIC series + mean + Newey-West(3) t; quintile portfolios
(equal weight, through the same simulator: next-open fills, commission/
slippage/stamp tax, T+1) Q1..Q5 and the long-short spread (50/50, net);
Fama-MacBeth cross-sectional regressions (fwd_ret ~ z factors), coefficient
means + NW(3) t. **Screen-pass criterion**: |meanIC| NW t ≥ 2 AND the
expected-direction quintile spread (net) monotone and > 0. **Trial budget: 1
registered exploratory run** (fetch + evaluate all 8 factors in one pass); no
post-hoc factor additions; survivors require a new pre-registration before
entering the walk-forward pipeline.

**Adjudication (2026-11-15, backfilled; run
`xs_cn_h2-2caccc9508e3f8f2-2026-11-15T105527Z`)**: 60 windows, average
cross-section 135 names. **No factor passed the full screen** (IC NW t ≥ 2 AND
net spread > 0 AND monotone). The only IC-screen pass: **size** (meanIC
−0.0545, NW t −2.82; FM −0.55%/month, t −2.75; direction-aligned net spread
+37.2%, Sharpe 1.15, MaxDD −6.2%) — but the quintiles were non-monotone
(Q1 +15.0% / Q2 +3.7% / Q3 +3.5% / Q4 +39.2% / Q5 +115.5%), so the portfolio
leg failed → not a screen pass. EP/BP direction positive but insignificant
(t 1.53/1.79) with negative net spreads — value failed in this window;
mom/lowvol/reversal all insignificant; turnover two-sided, not significant.
Under survivorship bias and ~2-day valuation sampling, this result is a
screening record only; reviving size requires a new pre-registration with
historical constituents.

### 2.6 Hypothesis H3: technical-indicator attribution (A-shares,
pre-registered 2026-11-15, before any evaluation run)

**Motivation**: the walk-forward composite model (16 technical features → GBM
classifier) already produced composite-level evidence (US +12.9% / CN −11.6%,
AUC ≈ 0.50), but no per-indicator attribution: which technical indicators
carry cross-sectional information themselves and which are noise. This
hypothesis evaluates each v1 feature as an individual cross-sectional factor
plus an equal-weight TA composite — 17 candidates.

**Data**: reuses the H2 cached universe (150 CSI-300 current constituents,
seed = 42; daily bars 2020-06 → 2026-11-13). No new fetching. Survivorship
bias as H2 → exploratory screening.

**Factor definitions**: the v1 feature library one-by-one (frozen definitions
in features/indicators.py), winsorized 1%/99% and z-scored per date;
`ta_composite` = equal-weight mean of the 16 z-values. **Expected directions
(classic TA conventions, pre-registered)**: ret_1/ret_5/ret_20: − (A-share
short-term reversal); close_over_ma5/20/60, ma5_over_ma20, macd_norm,
macd_hist_norm: + (trend-following); rsi14: − (overbought reversion);
bb_pos20: − (Bollinger upper-band reversion); bb_width20: 0 (two-sided);
vol_5/vol_20/atr14_norm: − (low-vol anomaly); volume_ratio20: 0 (two-sided);
ta_composite: +.

**Protocol**: identical to H2 (20-session rebalance, 20-session o2o forward
return, next-open fills, same simulator and costs, NW(3), FM regression);
minimum history gate lowered to 61 sessions (v1 warm-up). **Screen-pass
criteria same as H2**: IC NW t ≥ 2 AND direction-aligned net spread > 0 AND
monotone (two-sided factors use the observed IC direction). **Trial budget: 1
registered run** (17 candidates in one pass); no post-hoc additions;
survivors require a new pre-registration before entering the walk-forward
pipeline.

**Adjudication (2026-11-15, backfilled; run
`xs_ta_h3-2caccc9508e3f8f2-2026-11-15T110412Z`)**: 74 windows, average
cross-section 145 names. **None of the 17 candidates passed the full screen.**
- All IC screens failed: the closest, ret_1 (|t| = 2.10), was rejected on
  DIRECTION — the sample shows 1-day MOMENTUM (IC +0.045) rather than the
  pre-registered reversal; FM confirms (β +1.00%/month, t 2.87).
- FM-significant and direction-consistent with the pre-registration: only
  **bb_pos20** (β −0.73%/month, t −2.05) and **vol_5** (β −0.79%/month,
  t −2.74) — but their net-cost quintile portfolios failed, so not screen passes.
- FM-significant opposite to the pre-registration: atr14_norm (t +2.02 vs
  expected −) — contradicts vol_5, indicating unstable volatility measurement.
- Composite-model cross-reference: the GBM on the same feature set
  (US AUC 0.500 / CN 0.494) is consistent with weak per-indicator signals.
- **Follow-up constraints**: the three observations (1-day momentum on ret_1,
  bb_pos20, vol_5) may only be revived via separate new pre-registrations
  with evidence-based directions; nothing is adopted post hoc in this round.

### 2.7 Hypothesis H4: multi-factor synthesis and TOP_N selection (A-shares,
pre-registered 2026-11-15, before any evaluation run)

**Data**: reuses the H2/H3 cache (150 CSI-300 current constituents, seed = 42;
bars 2020-06 → 2026-11-13; Baidu valuation 2021-11 → 2026-11-13). Survivorship
bias as H2 → final status ceiling research_only.

**Factor pool (14, directions normalized so larger = more favorable)**:
EP(1/PE_TTM,+), BP(1/PB,+), size(ln cap,−), mom_20(−), mom_12_1(+),
reversal_5(+, i.e. −ret_5), lowvol(−vol_20,+), amihud_20(+), turnover(two-sided,
observed), rsi14(−), bb_pos20(−), vol_5(−), macd_hist_norm(+),
close_over_ma20(+). Covers Value/Size/Momentum/Reversal/Volatility/Liquidity/
Technical (Oscillator/Trend). Unavailable families (Quality/Growth/Sentiment/
industry neutralization) are declared missing, never substituted.

**Preprocessing**: MAD winsorization (K = 3, ×1.4826) → cross-sectional
z-score → size neutralization (cross-sectional OLS z ~ 1+ln(cap), residual;
size itself not neutralized); missing values dropped, never imputed.

**Sample split (anti-snooping)**: the 74 monthly windows are split in time —
TRAIN windows (first half) for factor screening and redundancy control
(criteria: |MeanRankIC| ≥ 0.02, |ICIR| ≥ 0.30, |NW t| ≥ 2, coverage ≥ 80%;
|ρ| > 0.7 dedup keeping the higher |ICIR|), the factor list is then FROZEN,
and synthesis is evaluated only on TEST windows (second half).

**Synthesis methods (weights use only train-window / trailing past-window
information)**: A equal weight (baseline), B trailing ICIR weights (past 12
windows, non-negative, normalized), C shrinkage ICIR (λ = 0.5),
D Ridge (α = 10, pooled rows of the past 12 windows).

**Portfolio**: TOP_N = 30 equal weight (single name ≤ 20%, gross ≤ 80%),
rebalance every 20 sessions, next-open entry/exit, same simulator and costs
(T+1/commission/slippage/stamp tax). Benchmarks: the equal-weight universe
portfolio (same simulator) and capped equal-weight buy-and-hold; the CSI 300
index benchmark is unavailable in this environment (Eastmoney endpoint) and
is declared as such.

**Criteria (all must hold to pass)**: on the test windows the best synthesis
method (a) net compounded return > the equal-weight universe benchmark AND >
capped buy-and-hold; (b) test-window monthly RankIC mean ≥ 0.02 with NW t ≥ 2;
(c) still > benchmark at costs ×4 and lag +1 window; (d) learned weights not
worse than equal weight (otherwise the method is rejected per the template
rule). **Trial budget: 1 registered run** (14 factors × 4 methods in one pass
+ robustness variants); passing entries remain research_only (survivorship
bias) and require a new pre-registration with historical constituents.

**Adjudication (2026-11-15, backfilled; run
`xs_mf_h4-2caccc9508e3f8f2-2026-11-15T114232Z`)**: train 30 / test 30 windows,
average cross-section 138 names.
- **Train-window screening**: BP (t 2.98), amihud_20 (t 4.45), turnover
  (t −4.83, observed negative), vol_5 (t 2.77) passed all four gates; lowvol
  passed but was redundant with vol_5 (|ρ| = 0.71) and dropped; size was
  significant (t −2.23) but rejected on ICIR 0.24 < 0.30; EP t 1.86 narrowly
  failed; technical/oscillator factors all failed (consistent with H3).
  Frozen selected list: **BP, amihud_20, turnover, vol_5**.
- **Test-window synthesis**: equal-weight composite +12.6% (Sharpe 0.34)
  **below the equal-weight universe benchmark +18.6%**; ICIR weights −0.5%,
  shrinkage −0.5%, Ridge −0.5% — **all learned weighting schemes underperformed
  equal weight and turned negative**; test-window RankIC 0.0247 (t 0.52) not
  significant → the train-window IC strength was overfit/weight instability.
- **Criteria a/b/c all False → H4 rejected**; per the template rule the
  weight-learning methods (ICIR/shrinkage/Ridge) are rejected (they did not
  beat the equal-weight baseline).
- Final status: **rejected** (the H4 synthesis configuration); BP/amihud/
  turnover/vol_5 remain pre-registered screening observations.

### 2.8 Hypothesis H5: critical-information-point monitor validation
(deal-amount anomalies × news correlation, pre-registered 2026-11-15, before
any analysis run)

**Objective**: land the real-time monitoring module designed in the session
system-prompt contract as deterministic code and validate it on real data —
identify "critical information points" = deal-amount statistical anomaly AND
news correlation holding simultaneously.

**Data**:
- Deal amounts: cached 150-symbol daily amount series (2020-06 → 2026-11-13, CNY).
- Per-stock news: `stock_news_em` (~10 latest items per symbol, verified live)
  — covers only the recent window; historical per-stock news unavailable (declared).
- Macro news: `news_cctv` xinwenlianbo archive (historical dates, verified live)
  — used for market-level correlation.

**Statistical rules (identical to the system-prompt contract, implemented
deterministically in code)**:
- Baseline: rolling 30-session median/MAD (minimum 10 observations); anomaly =
  Z ≥ 3.0 (robust) or |day-over-day| ≥ 2.0; both Z and R reported with the
  fired rule labeled.
- Market tier: universe-aggregate daily deal amount under the same rules.
- Correlation: stock = news publish date ∈ {anomaly date, +1 session} (daily
  resolution; intraday alignment unavailable — disclosed); market = CCTV items
  on the anomaly date hitting macro keyword classes (monetary/fiscal/capital
  market/macro/industry policy).
- **Dual-requirement flag**: CRITICAL requires BOTH the quantitative anomaly
  and the news correlation; quant-only → watchlist (appears only in
  reasoning_steps, never in critical_information_points).
- Suppression: consecutive anomaly sessions collapse to the first date; the
  same event is never flagged twice.

**Budget**: 1 registered run — full-panel anomaly scan; news fetches ≤ 10
symbols (ranked by recent Z) + ≤ 8 market anomaly dates CCTV. **Criteria
(module validation, not a profitability claim)**: (a) output strictly follows
the system-prompt JSON contract with reasoning before conclusions; (b) at
least one dual-confirmed CRITICAL point with complete evidence; (c) the
calibration table honestly reports anomaly/flag/hit counts; (d) deterministic
statistics reproducible by tests. Known limitations: daily resolution (no
intraday tape); limited news coverage window; news-in-feed ≠ material (no
sentiment/importance model — the weak correlation is disclosed); no causal
claims.

**Adjudication (2026-11-15, backfilled; run `xs_info_h5`)**: **module
validation PASSED.**
- Full-history scan: 150 symbols × 1566 sessions → 9556 collapsed anomaly
  clusters; market tier 39 clusters.
- Market tier: the top-8 anomaly days correlated with CCTV **8/8 dual-confirmed**
  (robust z 4.4-12.1), including 2024-09-24 (z = 12.1, +122% d-o-d; that day's
  CCTV items hit monetary/fiscal/capital_market classes — consistent with the
  known policy-stimulus rally).
- Stock tier (last 30 sessions): 109 symbols had recent anomaly clusters; top-10
  news fetches → **5/10 dual-confirmed** (entity-matched headlines: fund-flow
  rankings, shareholder stake increases, business changes); 5 quant-only
  anomalies watchlisted (recorded in reasoning_steps only, excluded from
  critical_information_points).
- Criteria (a)-(d) all satisfied; **honestly disclosed**: the market CCTV
  keyword leg has low specificity (a 10-20 item daily bulletin almost always
  hits a macro keyword), so the market-tier discriminating power comes from
  the quantitative leg; the stock entity-matched feed is the higher-specificity
  news leg.
- Final status: **monitor tool validation passed (E3, research evidence)**;
  flags are information events, not strategy signals; forward value unproven;
  no profitability or live-trading claim.

### 2.9 Hypothesis H6: forward validation of critical information points
(pre-registered 2026-11-15, before any forward statistic)

**Question**: do H5 critical information points (deal-amount anomaly + news
correlation) carry FORWARD information? **Core incremental question**: does
the news leg add forward value beyond the quantitative anomaly leg alone?

**Event sample**:
- All: every collapsed stock-level anomaly cluster in the cached panel
  (~9556 clusters) — each gets forward outcomes (below).
- News subsample: per-stock news fetched for the **top 60 recent-window
  anomaly symbols by |z|** (budget ≤ 60 calls); within the subsample, split
  into dual-flag (H5 rule: anomaly day / +1 session, entity-matched feed)
  versus quant-only; headlines pass a **deterministic materiality classifier**
  (HIGH/LOW/MEDIUM rules in the implementation, replacing H5's
  keyword-presence determination).
- Market tier: H5's 39 market anomaly clusters (8 CCTV-associated vs 31
  unverified).

**Forward outcomes (per event, all from cached adjusted data, no look-ahead)**:
1. reaction_return = close_adj[t+1]/close_adj[t] − 1 (reaction day);
2. tradable_return = open_adj[t+2]/open_adj[t+1] − 1 (next-session o2o,
   executable);
3. persistence = amount[t+1] / baseline_median(t) (volume persistence);
4. fwd5 = close_adj[t+5]/close_adj[t] − 1 (5-session drift). Events without
   a next bar are dropped.

**Statistics (anti-clustering)**: aggregate by event date (equal-weight mean
across same-date events) → date-level series → Newey-West(3) t; two-group
comparisons use the NW t of the daily mean-difference series over dates where
BOTH groups have events; market-tier 39 events use Welch t + event-level
time-sorted block bootstrap (block = 5), with the overlap limitation disclosed.

**Criteria (two-sided, no direction pre-assumed; all must hold for the news
leg to add value)**:
(i) dual vs quant-only: reaction_return difference |NW t| ≥ 2;
(ii) dual vs quant-only: persistence difference NW t ≥ 2 (must be positive,
per the persistence semantics);
(iii) market tier: flagged vs unflagged next-session market return
|Welch t| ≥ 2;
(iv) materiality tiers: HIGH-group tradable_return vs LOW-group |NW t| ≥ 2.
Any failure → the news leg carries no measurable incremental information and
the monitor stays review-only (E3).

**Budget**: 1 registered run (≤ 60 news calls + full deterministic
computation); power is limited by recent-window news availability (~25-30
independent event dates) — low power reported honestly. The final status does
not change with results: the monitor is an information-review layer, not a
strategy signal.

**Adjudication (2026-11-15, backfilled; run
`xs_ev_h6-f572abb6b9693be3-2026-11-15T134803Z`)**: **the news leg shows no
measurable incremental forward information → H6 rejected; the monitor stays
review-only (E3).**
- The pre-registered paired test was **structurally infeasible**: dual and
  quant-only event dates have ZERO overlap (a symbol-date belongs to one group
  by construction) → no paired difference series exists; the market tier
  flagged 39/39 (CCTV keyword presence matched every date) → the control group
  is empty. All four criteria False (including the non-evaluable NaNs).
- Post-hoc exploration (clearly labeled, multiple-testing burden
  self-acknowledged): dual vs quant reaction return +1.09% vs +0.63%
  (Welch t 0.44, CI spans zero); persistence 2.12× vs 2.11× (t 0.04) — no
  signal; the stock-oriented materiality classifier scored 0/39 CCTV items HIGH
  (domain mismatch) → the market news leg needs a policy-specific
  materiality model before it can be tiered.
- Unconditional baselines (recorded without multiple-testing adjustment):
  after anomaly clusters, **volume persistence is 210% of baseline**
  (t 73.5, strongly mechanical); 5-session drift +0.42% (t 2.62, positive,
  unadjusted).
- Honestly recorded: two implementation defects found and fixed during the run
  (cached bars missing *_adj columns; block-bootstrap index missing circular
  wrap); the actual budget was 60 stock news calls + 39 CCTV calls (the CCTV
  verification of the 31 unverified clusters was necessary for the market
  comparison and is disclosed in the report).

## 3. Unresolved decisions (requiring user/account-owner input; handover §10)

- Primary market and account type (US/CN researched in parallel; undecided).
- Real cost schedule (commission/slippage/stamp tax/borrow — currently
  UNVERIFIED defaults).
- Authoritative data vendor and adjustment policy (currently bundled CSVs +
  akshare updates; CN adjustment factors now derived, but vendor policy is
  still unconfirmed).
- Decision/execution schedule and missed-session handling (currently:
  after-close decision, next-open execution, suspended orders expire).
- Broker integration and operational owner (no broker connection exists).
- Capital scale, benchmark, and risk budget formal authorization.

## 4. Discarded alternatives (to avoid re-stepping on known rakes)

- Solver + post-hoc normalization (root cause of the legacy 80% violation).
- Treating the complement of p_up as a short signal (semantic error; project
  is long-only).
- Scaling lot basis in adjusted space (fabricates cost basis).
- Threshold selection on a report period itself (first-fold pollution).
- Volume-unit assumptions (now inferred by the amount/volume cross-check and
  recorded).
- Market-level news correlation via CCTV keyword presence (zero specificity;
  needs a policy-domain materiality model).
