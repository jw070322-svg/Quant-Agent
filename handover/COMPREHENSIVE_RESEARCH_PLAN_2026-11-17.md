# Comprehensive research and engineering plan

Date: 2026-11-17 (America/Los_Angeles). Reviewed commit: `23e7e7c`.
Primary entry point: `README.md`. This is a planning deliverable, not authorization to launch experiments, approve strategies, or execute trades.

## 1. Evidence, scope and continuity

This plan integrates:

1. [The linked book-analysis conversation](codex://threads/01a1e93f-20e1-7512-a07d-36626335c1e1), read through its completed response. Its source is *Quantitative Trading: Algorithms, Analytics, Data, Models, Optimization*. Book page references below are inherited from that analysis, not a claim of a new independent full-book reading.
2. `handover/IMPROVEMENT_PLAN_2026-11-16.md`, including its source findings R1–R7 and README corrections.
3. Current allocator, ledger, broker, factor-study, metrics, configuration and supporting documentation.

The source commit has not changed since the prior review. The existing US manifest modification and untracked November 16 plan were preserved. The prior verification recorded 125 passing tests; the full suite was **not rerun** here because it writes project datasets. This review used isolated, in-memory diagnostics for new findings. No implementation files or historical reports were edited.

The November 16 plan remains the detailed defect register. This document supersedes its sequencing where explicitly stated, and adds research specifications, data contracts, portfolio mathematics, validation gates and delivery criteria. Neither plan should be read as completed implementation.

## 2. Analysis of the existing plans

### 2.1 What remains valid

Keep causal data alignment, signal/label separation, complete method coverage, real benchmark membership, execution timing, strict approval gates, aggregate order checks and shard provenance as mandatory work. Rechecked source still contains the future valuation lookup, ineffective coverage calculation, first-window-only score accumulation and top-N universe benchmark.

The linked conversation supplies an important organizing principle: predictive return signals, risk estimates and execution algorithms are separate components. Covariance shrinkage or GARCH can improve risk estimation without creating alpha. A statistically significant factor is not automatically economically executable.

### 2.2 What should change

| Earlier emphasis | Revised treatment | Reason |
|---|---|---|
| Correct factor studies, then examine execution realism | Repair accounting and metric semantics before an economically interpreted replay | Correct signals paired with wrong share units or P&L still produce misleading evidence. |
| A generic approval gate built around classifier AUC | Versioned gates by model type, plus common economic/operational gates | Rank scores and return forecasts are not probabilities; AUC/Brier are not universal requirements. Existing classifier failures remain recorded. |
| Four book-inspired strategies | Two daily/monthly research candidates; two explicitly deferred tracks | The repository is long-only and uses daily data; it lacks validated borrow/two-leg execution and event-level queues. |
| A historical-universe dependency late in the process | Design the adapter immediately; block broad empirical claims until coverage is verified | Data acquisition can have a long lead time. Corrective replay on biased data is diagnostic only. |
| Add sophisticated models | First isolate incremental effects through matched ablations | More complex risk, costs, and signals must not be confounded in a single comparison. |
| New plan treated as a fresh research start | Record overlap with rejected H1–H4, including momentum variants | A book citation does not erase prior trials or turn reused historical data into a holdout. |

### 2.3 Additional findings from this review

**N1 — Risk penalties cannot remove an eligible holding.** `portfolio/optimizer.py:87` floors every post-penalty score at `1e-6`. With two expected returns of 0.001, volatilities of 0.1, penalty coefficient 1, a 0.20 single-name cap and 0.80 budget, the diagnostic returned weights `{A: 0.2, B: 0.2}` although both penalized edges were negative. The subsequent all-zero check cannot catch this case. This contradicts the claimed ability to remain in cash when the penalty eliminates all edges.

**N2 — First-period loss can disappear from drawdown.** `research/metrics.py:26–37` builds an equity curve without prepending initial capital. `summarize([-0.1, 0.0])` returned maximum drawdown 0.0 rather than -0.10. Some caller paths may already include an initial point; audit each caller before revising past results.

**N3 — Reported completed trades are not matched net round trips.** `research/metrics.py:110–123` pairs BUY/SELL prices by array position, ignoring quantities and fees. A 100-share purchase at 10 and a one-share sale at 11, with fees of 5 each, returned one completed winning trade with 10% price return. This is not a valid net completed-position statistic. The remaining 99 shares and allocated fees matter.

**N4 — Missing marks revert to purchase price.** `ledger/account.py:140–142` uses each lot's entry price when a mark is absent. This can erase unrealized gains/losses during a suspension. This is a static source finding; a regression fixture is required. Preserve the last validated mark with its age and quality flag, and block new risk when valuation is materially incomplete.

**N5 — Execution assumptions mix units and information times.** `execution/broker.py` applies raw-volume participation and lot rounding to adjusted-price execution. It also uses the completed day's volume to cap an opening fill. Full-day volume can be an explicit ex-post capacity assumption but cannot be described as an observable opening-liquidity limit. Stop handling must share executable constraints rather than assume a threshold guarantees liquidity. These are model limitations to resolve and test, not quantified estimates of historical P&L distortion.

These findings extend the earlier plan; they do not invalidate every existing test or automatically overturn every historical result. Publish an impact matrix by metric, module and run.

## 3. Target architecture and contracts

Keep a modular Python package, not a premature service deployment. Reuse the ledger and simulator after their contracts are corrected.

| Layer | Required contract | Deliverables |
|---|---|---|
| Data and universe | Immutable snapshots; point-in-time availability; historical membership and corporate actions | snapshot manifest, universe intervals, quality report |
| Signals and labels | Scores depend only on available inputs; labels stored separately with maturity timestamps | feature panel, label panel, factor catalog |
| Estimation | Frozen training cutoff; forecast horizon and units; uncertainty distinct from return variance | prediction artifact, model manifest, diagnostics |
| Risk and allocation | Covariance, exposure limits, uncertainty and costs; cash permitted | target portfolio, binding constraints, no-trade reasons |
| Execution and ledger | Raw share/currency units; pending-order reservations; realistic fills and cash events | orders, fills, account snapshots, reconciliation |
| Evaluation | Common time window, honest benchmark, exact fee-aware attribution, dependence-aware uncertainty | metrics, ablations, trial register, errata |
| Governance and observation | Model-type-specific gates; immutable forecasts and later outcomes | approval evidence, shadow report, operational alerts |

Required shared fields: `run_id`, `parent_run_id`, `hypothesis_id`, `code_commit`, `dirty_patch_hash`, `config_hash`, `dataset_snapshot_id`, `source_version`, `event_at`, `published_at`, `ingested_at`, `available_at`, `decision_at`, `earliest_execution_at`, `label_end_at`, `calendar_id`, `universe_version`, `price_basis`, `currency`, `quantity_unit`, `model_kind`, `forecast_horizon`, `seed`, `backend`, `dependency_manifest`.

Historical research availability must be distinguished from prospective ingestion time. Data downloaded today must not automatically be claimed available historically; use verifiable vendor vintages/publication times or an explicit conservative limitation. Do not set historical availability equal to the present download timestamp and then call the resulting absence of history a clean historical experiment.

## 4. Algorithm specification

### 4.1 Shared clock and estimation rules

Proposed starting scope: A-share long-only stock selection at monthly rebalances, daily accounting, local exchange timestamps stored with explicit timezone. These are research defaults, not approved account mandates. US studies remain a separate replication track with separate rules.

At signal close t, compute features using observations available by that cutoff. Let e(t) be the next eligible session's open and x(t) the open one rebalance horizon later. Define the evaluation label as the total-return wealth change of a unit investment over e(t) to x(t), including corporate actions. Store both timestamps; never substitute adjusted price into an executable raw-share order without a mapping.

For fitting at cutoff T, require `label_end_at < T`. Purge observations whose outcome intervals overlap validation/test boundaries. Fit transformations across time on training data only; per-date cross-sectional standardization may use that date's available eligible universe. Retain missing future labels for selection purposes and mask them only during evaluation.

Define coverage at each date as finite factor values divided by eligible universe size. Define factor direction once. Undefined IC, zero dispersion, inadequate history or absent controls produce an explicit invalid/inconclusive result, never an automatic pass.

### 4.2 Candidate A: multi-horizon momentum with cost-aware rebalancing

Source connection: linked discussion of momentum at printed pp. 42–43, dynamic signals/costs at p. 71. The following windows are proposed extensions, not the book's exact example.

Let TRI be a documented point-in-time total-return index. Initial specification:

\[
m^{(L)}_{i,t}=TRI_{i,t}/TRI_{i,t-L}-1,\quad L\in\{63,126,252\},\qquad s_{i,t}=\tfrac13\sum_L z_t(m^{(L)}_{i,t}).
\]

No skipped month in this first version; a skip-month alternative would be a separate registered trial. Use a fixed clipping/standardization policy and deterministic symbol tie-breaking. Start with top-quintile equal weights subject to caps and cash, alongside a single-252-session momentum comparator. Low ranks mean omission or underweighting, not unauthorized short positions.

Hypothesis: the composite improves net active return or predefined drawdown/turnover efficiency relative to single-horizon momentum at comparable exposure. Separately test the same composite with a cost-aware allocation policy; this isolates signal gain from trading-policy gain.

Important dependency: this overlaps prior momentum trials in H1–H4. Register an explicit overlap statement, changed mechanism, new evidence source and trial budget. Without a defensible new hypothesis and adequate unseen evidence, keep it as an exploratory replication rather than reopening a retired strategy.

### 4.3 Candidate B: conditional benchmark-relative return prediction

Source connection: linked conditional-model discussion at pp. 36–40 and parameter uncertainty at pp. 22–29. This is not an implementation of the book's NPEB estimator.

For matured monthly observations, define active return e(i,m) as stock total return minus the chosen benchmark's total return over identical endpoints. Predict:

\[
e_{i,m+1}=a_i+b e_{i,m}+c r_{B,m}+\varepsilon_{i,m+1}.
\]

Start with pooled ridge regression and regularized asset intercepts; fix the predictor set. Shrink sparse-history asset intercepts toward the pooled value; no backfilled identity history. Compare against zero active-return prediction, an intercept-only model and Candidate A. A 120-month per-asset replication of the book cannot be assumed feasible from a roughly 2019-start provider history; either obtain longer history or label a shorter pooled model as a new extension.

Use date-block resampling of whole cross-sections to estimate mean-forecast uncertainty U. Estimate return covariance Sigma separately using a shrinkage baseline. Their units must match the predicted holding period. Defer GARCH until residual diagnostics and a separately budgeted ablation justify it.

No AUC threshold for this continuous-return model. Assess out-of-sample error versus zero/intercept, Rank IC, net active portfolio return, turnover and uncertainty. Benchmark subtraction alone does not create beta neutrality.

### 4.4 Shared portfolio design

A rank score is not an expected return. The first signal benchmark can use rank-based equal weighting; a monetary optimizer requires a training-only mapping from scores to horizon returns, or Candidate B's forecasts.

Let w be proposed stock weights, w-minus current price-drifted weights, d = w - w-minus, mu the expected stock-return vector, Sigma the return covariance, and U the covariance of estimated means. Consider the convex, single-period research objective:

\[
\max_w\;w^\top\hat\mu-\frac{\lambda}{2}w^\top\hat\Sigma w-\kappa\sqrt{w^\top U w}-C(d).
\]

\[
C(d)=\sum_i(c_i^{buy}d_i^+ + c_i^{sell}d_i^-)+\tfrac12 d^\top\Lambda d,
\quad d_i^+=\max(d_i,0),\quad d_i^-=\max(-d_i,0).
\]

All terms are expressed as fractions of current NAV for the same decision horizon. Sigma and U are positive semidefinite, as is impact matrix Lambda. If the prediction is active return, either reconstruct total expected stock returns with an explicit benchmark forecast or use a separately specified active-weight objective and benchmark/cash accounting; never silently optimize active forecasts as total returns.

Use long-only, cash, single-name, industry, turnover and liquidity constraints. U represents estimation uncertainty, not an extra alpha signal; the robust penalty is a proposed design rather than an exact Bayesian theorem. Distinguish it from predictive covariance to avoid counting the same uncertainty twice.

Transaction costs are forecast in optimization and realized once in ledger P&L. Do not subtract the optimizer's predicted cost again from already net realized returns. Model minimum commissions and discrete lots in execution validation; smooth optimization costs are approximations.

Introduce no-trade behavior by comparing improvement over the feasible current portfolio, while allowing mandatory risk reduction even when it costs money. An infeasible solve cannot authorize extra leverage or sell locked shares. Return structured reasons and escalate to a deterministic risk-reduction policy or hold where feasible.

This objective is an initial approximation, not the full dynamic-control solution. Dynamic research explicitly links signal persistence and trading costs; it motivates testing gradual trading without asserting that every smoothing rule is optimal. See [Gârleanu and Pedersen](https://www.nber.org/papers/w15205).

### 4.5 Deferred tracks

- **Relationship mean reversion:** require point-in-time pair selection, relationship-break tests, both-leg executable prices, financing/borrow availability and joint risk. A long-only relative-ranking substitute is a different hypothesis, not a reproduction of a pairs trade.
- **Order-book pressure:** require event-level depth/trades/cancellations, synchronized clocks, fill-probability labels, latency and queue simulation. Daily volume cannot stand in for event-level book imbalance. No implementation budget until data availability is demonstrated.
- **News materiality:** keep review-only. Repair control-group overlap and availability timestamps before a predictive experiment; a better text classifier alone cannot repair a non-identifiable event study.

## 5. Validation design and decision gates

Separate engineering validity, statistical evidence, economic value and operational readiness. Never combine them into a single weighted score that allows good performance to offset a leakage failure.

| Gate | Pass condition | Failure outcome |
|---|---|---|
| G0 Evidence/fixtures | Immutable inputs, isolated tests, complete manifests | Stop empirical claims; repair infrastructure |
| G1 Causality and implementation | R1–R7 resolved; label maturity, full method coverage and timing tested | Invalid protocol, not a losing strategy |
| G2 Accounting/evaluation | Raw units reconciled; first-day loss and fee-aware partial fills correct; benchmark construction verified | Historical performance remains qualified |
| G3 Research | Registered comparisons, budget, family-aware inference and adequate unseen coverage | Rejected under stated criteria or inconclusive; never relabel same data unseen |
| G4 Economics | Net active value and risk constraints under agreed cost/capacity scenarios | Research-only, with breakeven analysis |
| G5 Observation | Frozen prospective outputs, mature outcomes, zero unexplained reconciliation breaks | Remain observation-only |
| G6 Promotion | Explicit strategy-specific approval and operational evidence | No promotion |

Use expanding chronological outer evaluation with inner chronological tuning. Proposed development protocol: 36 months of training, 12 months of validation and 12-month outer blocks if history supports it; folds with inadequate data fail eligibility. Final untouched or prospective evaluation must have a recorded access boundary. Already examined H1–H6 history cannot be made untouched by renaming the split.

Proposed first-wave budget: three candidate specifications (A equal-weight allocation, A cost-aware allocation, B conditional model), plus fixed reference portfolios; one declared ridge grid of {0.1, 1, 10} evaluated only inside training/validation. Register all fits, ablations and failures, not just the three final outputs. Risk/cost coefficients and economic thresholds must be fixed before outer results are examined. Unspecified risk budget or cost schedule limits output to diagnostics, not candidate approval.

Use paired net-return differences on common dates, with whole-date blocks preserving cross-stock dependence. Primary statistic: mean net active return versus the assigned simple reference; report annualized effect, dependence-aware interval, turnover, exposure and drawdown. Predeclare three primary comparisons and apply a family correction such as Holm at 5%; significance is a necessary research criterion if specified, not sufficient proof of economic value. Low power yields inconclusive evidence, not proof of no effect.

Rank IC/ICIR, decay, beta/industry exposures, cost break-even and regime concentration are secondary diagnostics. Do not annualize overlapping IC as independent daily samples. Use arithmetic excess-period returns for Sharpe and geometric compounding for CAGR; include initial NAV in drawdown. Match trades using quantities and allocated fees, with partial-exit and completed-position statistics explicitly distinguished.

Stress cases: costs 1x/2x/4x; execution delay +1/+2 sessions; constrained fills; suspension/limit states; price gaps; corporate actions; missing data; correlated drawdowns; reduced capital capacity. Persist attempted and realized orders so implementation shortfall can be attributed. No guarantee of significance from a fixed number of months; report effective sample size and minimum detectable effect.

Trial history is essential to interpreting selected backtests. PBO or other selection diagnostics may supplement the chronological protocol when the candidate matrix supports them; they do not replace prospective evidence. See [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## 6. Work packages, dependencies and acceptance

| Package | Concrete scope | Acceptance evidence | Depends on |
|---|---|---|---|
| W0: evidence isolation | Temporary test roots; document run impacts; preserve modified manifest/data pairs; pin offline environment | Tests leave protected project files unchanged | None |
| W1: causal research core | Implement November 16 P0-B; shared PIT and label contracts; complete score tables; factor directions | Future-row perturbations leave prior decisions unchanged; 30/30 windows; real coverage; NaN fails | W0 |
| W2: accounting and metrics | Fix N1–N5; raw shares/currency ledger; explicit corporate-action and stale-mark policy; benchmark timing | Hand-worked splits/dividends/partial sells; -10% first loss gives -10% MDD; negative risk-adjusted edges allow cash | W0 |
| W3: decisions and approvals | November 16 P0-C; frozen policy and inference contract; batch reservations | No aggregate overspend/oversell; locked shares retained; missing evidence blocks promotion | W1, W2 contracts |
| W4: reproducibility | Immutable run manifests and shard integrity; exact snapshots; resolved dependencies; serial baseline | Mixed shards rejected; same-backend numerical tolerance met; resume produces no duplicate decisions | W0; finalize after W1/W2 |
| W5: corrective replay | Bounded bugfix replay of affected studies; old/new decomposition and errata | No overwrite; shared evaluation dates; corrected claims separated from original claims | W1, W2, W4 |
| W6: historical data readiness | Membership intervals, classifications, corporate actions, benchmark TR and cost/rule vintages | Listed/delisted and revision fixtures; unavailable history explicitly blocks broad claims | Start contract after W0; external data delivery |
| W7: small research wave | Register A/B and matched ablations; implement shared forecasts/risk/cost objective | G3/G4 report, including negative/inconclusive outcomes | W1–W6 |
| W8: prospective observation | Append-only forecasts; mature outcomes; reconciliation and alerting | Version-consistent observations; no hidden backfill; operations distinct from strategy efficacy | Data contracts and G0–G2; capture may start before W7 finishes |

Suggested implementation locations: extend `data/` with PIT/universe/corporate-action contracts; `research/` with experiment manifests and estimator evaluation; `portfolio/` with risk/cost interfaces; `ledger/` and `execution/` with raw-unit reconciliation; `registry/` with gate enforcement. Keep exploratory candidates behind research-only commands. Avoid a broad rewrite where a tested interface extraction suffices.

Immediate first slice: W0, then W1 and the first-loss/partial-fill/negative-edge fixtures from W2. Only after these pass should corrective replay begin. Full accounting migration is a larger milestone and must not be hidden inside a one-day bugfix estimate.

Indicative effort, not a commitment: W0 1–2 engineer-days; W1 3–5; W2 4–8; W3 2–4; W4 2–3; W5 1–2 plus compute; W6 2–4 plus external-data lead time; W7 4–7. Tasks overlap only where contracts are stable. Prospective evidence takes calendar time and cannot be represented as engineering days. This replaces the earlier 7–12-day estimate for the now-expanded scope.

Every work package delivers: reviewed change set, behavior tests, run manifest, acceptance results, remaining limits, and an active-handover update. Document current-versus-proposed behavior explicitly. Account owner/vendor/benchmark decisions can remain pending while local fixture and interface work proceeds.

## 7. Conclusions and next action

A more comprehensive plan is warranted. The linked book analysis complements the defect-focused plan, but it does not justify launching four strategies immediately. The new critical path is causality, accounting and evaluation correctness, reproducibility, and then a small, registered research wave.

Prioritize multi-horizon momentum and conditional benchmark-relative returns as **proposed, unvalidated research tracks**, subject to overlap review and data readiness. Defer pairs trading, order-book modeling, advanced volatility models and larger GPU runs until their incremental purpose and required data are established.

The next implementable milestone is W0 plus the deterministic W1/W2 regression fixtures. This planning turn has not started those changes, dispatched ZCode work, or promoted any strategy.
