# US and A-share strategy validation — November 17, 2026

## 1. Scope and data

This study tests whether three simple, long-only rules add value after modeled costs: a 200-session trend filter, 252-session momentum rotation, and multi-horizon momentum rotation. Each rule was tested in both markets under baseline execution, double costs, and one extra session of delay: 18 active configurations plus benchmarks. This is a narrow test of timing and ETF selection. It does not test single-stock factors, the project's regression model, or a live fallback trigger.

| Market | ETF universe | Evaluation | Latest 252 sessions | Cost per buy or sell |
|---|---|---|---|---:|
| US | SPY, QQQ, IWM, DIA | 2020-01-02 to 2026-11-16 (1,728 sessions) | 2025-11-14 to 2026-11-16 | 5 bp |
| A-shares | 510300, 510500, 510050, 159915 | 2020-01-02 to 2026-11-17 (1,665 sessions) | 2025-11-04 to 2026-11-17 | 10 bp |

US data come from [Sina Finance](https://finance.sina.com.cn/stock/usstock/sector.shtml), combining daily opens and closes with the vendor's adjustment factors. A-share data come from [Eastmoney](https://fund.eastmoney.com/159915.html), using back-adjusted daily opens and closes. The prices include warm-up history from 2018. The snapshot was retrieved on November 17 at 12:01 UTC, before the US open and after the mainland China close. Exact endpoints and checksums are in `manifest.json`. The source has one missing 159915 bar on 2021-02-08. The simulator marks it at its prior close and defers an affected rebalance until all four opens are available. Input checks cover duplicate dates, nonpositive or nonfinite prices, extreme price jumps and missing month-end observations; they do not independently validate every bar or corporate action.

## 2. Rules and execution

| Rule | Signal at month-end close | Benchmark |
|---|---|---|
| 200-session trend | Hold the broad ETF (SPY or 510300) if its close exceeds its trailing 200-session moving average. Otherwise hold cash. | Buy and hold the same broad ETF |
| 252-session rotation | Rank the four ETFs by adjusted close return over 252 sessions. Give the top two 50% each. | Monthly rebalanced 25%-per-ETF basket |
| Multi-horizon rotation | Compute percentile ranks for the 63-, 126-, and 252-session returns separately, average the three ranks, then give the top two 50% each. Break ties in fixed universe order. | Monthly rebalanced 25%-per-ETF basket |

Each trailing return is adjusted close today divided by adjusted close at the specified lag, minus one. Baseline fills occur at the next session's open. The delay test moves the active fill one session later and keeps the passive benchmark's schedule unchanged. The simulator marks positions daily, lets weights drift, and charges costs on actual traded notional, including entry. Cash earns zero. Each market is evaluated in its local currency, using 252 (US) or 244 (A-shares) sessions for annualization. The latest 252 sessions are measured from the existing positions and form a subset of the full history.

The paired comparison uses daily net strategy return minus benchmark return. It reports the annualized arithmetic mean and a nominal 95% confidence interval from 4,000 circular block bootstrap draws with 21-session blocks and seed 42. A Holm correction covers the six primary one-sided rule-versus-benchmark tests. These are exploratory intervals and tests on this sample; they do not control for retrospective choices such as which ETFs entered the study.

## 3. Results after modeled costs

| Market | Rule or benchmark | CAGR | Sharpe, zero cash rate | Maximum drawdown | Latest 252 sessions |
|---|---|---:|---:|---:|---:|
| US | SPY buy and hold | 15.26% | 0.75 | −38.76% | +11.36% |
| US | Equal-weight basket | 14.07% | 0.69 | −39.10% | +12.20% |
| US | 200-session trend | 9.08% | 0.69 | −27.52% | +0.81% |
| US | 252-session rotation | 13.96% | 0.69 | −34.37% | +10.20% |
| US | Multi-horizon rotation | 15.26% | 0.74 | −34.37% | +7.77% |
| A-shares | 510300 buy and hold | 1.28% | 0.16 | −38.47% | −10.83% |
| A-shares | Equal-weight basket | 3.24% | 0.26 | −38.80% | −13.05% |
| A-shares | 200-session trend | −1.83% | −0.08 | −43.32% | −0.39% |
| A-shares | 252-session rotation | 1.06% | 0.16 | −44.60% | −17.43% |
| A-shares | Multi-horizon rotation | 4.87% | 0.33 | −40.60% | −4.56% |

| Multi-horizon minus equal-weight | US | A-shares |
|---|---:|---:|
| Baseline CAGR difference | +1.18 pp | +1.63 pp |
| CAGR difference with double costs | +0.95 pp | +1.27 pp |
| CAGR difference with one extra session of delay | +1.18 pp | +1.19 pp |
| Latest 252-session total-return difference | −4.43 pp | +8.49 pp |
| Annualized arithmetic active return (nominal 95% CI) | +0.96 pp [−3.26, +5.21] | +2.02 pp [−4.88, +8.34] |

The US rotation result was concentrated in the earlier period. It returned **49.29% versus 23.91%** for the basket in 2020–2022, then **77.34% versus 99.07%** from 2023 to the cutoff. Over the whole study it roughly matched SPY on CAGR, with a smaller drawdown and slightly lower Sharpe. The US trend filter cut maximum drawdown by 11.24 percentage points versus SPY, while spending about 20% of sessions in cash and giving up 6.18 percentage points of CAGR. An exposure-matched benchmark would be needed to isolate the value of that timing rule from the effect of holding cash.

The A-share multi-horizon rule led the basket in both subperiods: **20.52% versus 10.25%** in 2020–2022 and **14.75% versus 12.76%** from 2023 to the cutoff. It also lost less than the basket over the latest 252 sessions, but suffered a larger full-period drawdown. The two speed horizons added to the 252-session rule produced positive estimated excess return in both markets; the confidence interval versus single-horizon rotation still included zero in each. None of the six primary tests showed statistically reliable outperformance after adjustment (all Holm-adjusted p-values were 1.00).

## 4. Limits and next validation gate

The four-ETF universes are fixed retrospectively, overlap in underlying holdings, and lack an independent holdout. The bootstrap cannot establish that an edge survives a change of regime. Adjusted-price accounting approximates reinvestment; a point-in-time rebuild from raw prices, distributions and splits is needed before making stronger claims. In particular, later vendor revisions and additive adjustments can change past return ranks. The fill simulation assumes continuous position sizes and executable adjusted opens when quotes exist. It does not model actual open-auction depth, temporary price limits, minimum lots, broker-specific fees, account taxes, or market impact. The study measures relative performance against simple tradable proxies without isolating exposures to specific equity factors.

1. Freeze the rules and benchmarks before new data arrive. Keep the A-share multi-horizon candidate as a separate research baseline. Preserve an untouched forward window and report net excess return, drawdown, turnover, and confidence intervals on a fixed review date.
2. Rebuild the inputs from point-in-time raw bars and corporate actions, match calendars to exchange sessions, and reconcile representative days against a second source. Add tradability checks, actual fees, liquidity constraints, and an exposure-matched comparator.
3. To evaluate stock selection, build a point-in-time stock universe with historical membership, delistings, filings and trade eligibility. Compare multi-factor portfolios with simple baselines using walk-forward validation and run the regression model and fallback jointly. Exercise the switch under predefined model degradation, shared data failures, and doubled costs.
4. Keep the fallback disabled for live orders until it passes a fresh, out-of-sample test with positive net benchmark-relative results and a predefined drawdown budget. A shared data or execution failure must block the fallback too.

Replay from the project root with `./.venv/Scripts/python.exe scripts/market_backtest_20261117.py artifacts/market_backtest_20261117`. The replay uses local snapshots and checks their hashes. The run used Python 3.13.9, NumPy 2.4.4 and pandas 2.3.3. `results.json` contains all metrics, `US_daily.csv` and `CN_daily.csv` contain the daily portfolios, and `run_manifest.json` records the script and results hashes. Add `--fetch` with a new, empty output path for a fresh snapshot at the configured cutoff. Fetching also requires Requests and AkShare; a fresh download may differ if providers revise history.

## 5. Conclusion

No tested rule is validated as a robust current-market strategy. The A-share multi-horizon ETF rotation is the best candidate for a forward test: it led its basket after both execution stresses and lost less than the basket recently. It still lost money recently, and its estimated edge is too uncertain to justify deploying it as an automatic fallback. The US multi-horizon rule showed no recent advantage against either passive baseline.
