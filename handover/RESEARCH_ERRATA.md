# Research Errata (append-only)

This file records research-validity defects and their impact on published
study claims. Entries are appended in the order discovered; historical run
artifacts under `artifacts/runs/` are preserved unmodified, and corrected
replays are separate runs linked from each entry. Per the governing plans
(`handover/IMPROVEMENT_PLAN_2026-11-16.md` R1-R7,
`handover/COMPREHENSIVE_RESEARCH_PLAN_2026-11-17.md` N1-N5): correcting an
experiment is a correction, not a new holdout, and does not establish
profitability.

## E-001 (2026-11-17): protocol defects R1-R7 in the cross-sectional studies
(H2, H3, H4)

Status before replay: **the affected study verdicts are
`invalid_protocol_pending_replay`.** The corrected replay runs are linked
below; their results replace the originals' claims, with old/new differences
attributed to the protocol fixes.

| ID | Defect | Location (pre-fix) | Affected claims | Corrected replay |
|---|---|---|---|---|
| R1 | Forward valuation lookup: the as-of window extended to signal+5 days, letting future valuation records enter historical signals (the age check rejected negative ages only implicitly) | `research/cross_section.py::_factor_rows`, `research/multi_factor.py::build_factor_panel` | H2/H4 valuation factors (EP/BP/size/turnover) | run `xs_cn_h2-*REPLAY`, `xs_mf_h4-*REPLAY` |
| R2 | Score-table accumulation via `setdefault` retained only the FIRST test window per learned method (ICIR/shrinkage/ridge each had 1 IC window instead of 30) | `research/multi_factor.py` synthesis loop | H4 method comparison (icir/shrink/ridge RankIC windows) | same replay |
| R3 | Coverage was max(n)/max(n) ≡ 1.0 by construction — the 80% coverage gate had no effect | `research/multi_factor.py::summarize_factor_ics` | H4 screening coverage claims | same replay |
| R4 | Nonfinite screening statistics could pass the gates (NaN comparisons are False) | `research/multi_factor.py::select_factors` | H4 screening | same replay |
| R5 | The "equal-weight universe benchmark" applied TOP_N=30 to a constant score — an arbitrary ≤30-name subset, not the universe | `research/multi_factor.py::run_top_n_sim` + benchmark call | H4 benchmark comparison (criterion a) | same replay |
| R6 | Target weights were keyed at signal+1 while the simulator fills at the open AFTER that close — active entries executed at signal+2 open, inconsistent with +1-open labels; the lag option shifted both signal age and delay | `cross_section.py::_quintile_sims`, `multi_factor.py::run_top_n_sim` | H2/H3 quintile net returns, H4 active returns and lag stress | all replays |
| R7 | Rows whose forward outcome was incomplete were dropped from the SIGNAL cross-section — label availability changed historical eligibility | same locations | H2/H3/H4 coverage and eligibility | all replays |

Additional corrections in the same replay round: single orientation convention
(aligned z at construction), label purge at the train/test boundary
(training rows maturing after the first test decision excluded), technical
features computed once per symbol.

## E-002 (2026-11-17): accounting and portfolio defects N1-N5 (metrics,
optimizer, ledger, broker)

| ID | Defect | Location (pre-fix) | Affected claims | Fix status |
|---|---|---|---|---|
| N1 | Risk-penalty flooring at 1e-6 kept holdings whose penalized edge was negative; cash was unreachable when the penalty eliminated all edges | `portfolio/optimizer.py::target_weights` | All runs' allocation behavior (weights could exceed the intended risk-adjusted policy) | Fixed; regression tests in `tests/unit/test_w2_fixes.py` |
| N2 | Drawdown was computed without anchoring the high-water mark at initial equity — a first-period loss was invisible | `research/metrics.py::summarize` | All runs' max_drawdown metric | Fixed; regression test |
| N3 | Completed-trade statistics paired BUY/SELL by position, ignoring quantities and fees — a 1-share partial sale of a 100-share lot reported a 10% win | `research/metrics.py::completed_trade_stats` | All runs' trade win rate / avg trade pnl | Fixed (fee-aware FIFO; partial exits vs completed positions distinguished); regression tests |
| N4 | Missing marks reverted to entry price, erasing unrealized P&L during suspensions | `ledger/account.py::equity` | Simulations with suspended symbols | Fixed (last-validated-mark chain + staleness; stale marks block new BUYs, de-risking SELLs permitted); regression tests |
| N5 | Full-day volume was used as an opening-liquidity cap (an ex-post capacity assumption, not an observable limit); stops assume threshold liquidity | `execution/broker.py` | All simulations' fill modeling | **Disclosure** (documented in ARCHITECTURE §6 and broker docstring); resolution requires event-level data — deferred per plan §4.5 |

## Replay ledger

| Study | Pre-fix run | Corrected run | Outcome delta |
|---|---|---|---|
| H2 cross-sectional screening | `xs_cn_h2-2caccc9508e3f8f2-2026-11-15T105527Z` | `xs_cn_h2-2caccc9508e3f8f2-2026-11-17T153112Z` | Conclusions substantially unchanged: size remains the only IC-screen pass and is slightly STRONGER corrected (NW t −3.30 vs −2.82; FM −3.25; net spread +39.6%, Sharpe 1.24) but still non-monotone → not a screen pass. EP/BP ICs rose (1.78/1.77) but stay < 2; turnover dropped to insignificant (−1.17). |
| H3 technical attribution | `xs_ta_h3-2caccc9508e3f8f2-2026-11-15T110412Z` | `xs_ta_h3-2caccc9508e3f8f2-2026-11-17T153427Z` | Conclusions unchanged: none of the 17 candidates passes; ret_1 still shows 1-day momentum (direction-rejected); FM-consistent bb_pos20/vol_5 legs still fail the net-cost portfolios. |
| H4 multi-factor synthesis | `xs_mf_h4-2caccc9508e3f8f2-2026-11-15T114232Z` | `xs_mf_h4-2caccc9508e3f8f2-2026-11-17T154618Z` | Materially changed by the fixes: with the universe benchmark corrected (R5) and PIT/score/timing fixes applied, the equal-weight synthesis BEATS the benchmark (+31.1% vs +17.9%) and survives costs ×4 (+23.1%) and lag (+30.5%) — criteria a/c/d now pass — but criterion (b) test-window RankIC significance still fails (t 0.58 < 2) → NOT PASSED per the pre-registered criteria; promotion remains refused. The flawed run's "all learned weights negative" claim did not survive the corrections. |

Interim ruling: until the replays complete, the H2/H3/H4 verdicts in
docs/DECISIONS.md §2.5-§2.7 and docs/VALIDATION.md §2.2-§2.4 are
`invalid_protocol_pending_replay` and must not be cited as evidence.

## E-003 (2026-11-17): two-sided factor orientation multiplied by zero
(found during the H4 corrective replay)

`research/multi_factor.py::preprocess` applied the direction multiplier
literally: a two-sided factor (direction 0, e.g. turnover) was transformed as
z × 0, fabricating a zero-variance z column. Its IC became undefined and the
R4 gate then rejected it — turnover, one of the strongest train-window factors
in the flawed run (t −4.83), silently vanished from the corrected screening.
Fixed: two-sided factors receive NO orientation transform (sign forced to 1).
Regression covered by the corrected replay (turnover restored, t −4.34,
selected). Recorded as a defect in the orientation-convention implementation,
not in the pre-registered factor definitions.
