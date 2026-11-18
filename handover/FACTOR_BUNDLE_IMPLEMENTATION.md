# Factor strategy bundle implementation — 2026-11-17

## Scope and current state

The package implements five frozen research specifications in each of the US and A-share markets, with separate results and a ten-hypothesis family correction. These are research experiments, not approved trading strategies. The first real-data readiness reports are under `artifacts/factor_bundles/`; both markets are blocked because canonical snapshots have not been supplied. Tushare access has not been assumed or purchased.

The implementation includes offline feature calculation, financial scoring, portfolio construction, raw-share execution, cost/delay stresses, inference, file hashes, trial records, and CLI commands. Synthetic end-to-end tests exercise the pipeline. No new claim of market profitability follows from these tests.

## Strategy specifications

Each market has MOM, ATO, QV, FAR and VPA. Defaults live in `research/factor_strategies.py`; version-1 constants cannot be changed through YAML. A change requires a new experiment version.

| ID | Signal and selection | Primary comparator |
|---|---|---|
| MOM | Mean z-score of trailing 63, 126 and 252 session total returns; top quintile | Top quintile of 252-session momentum |
| ATO | Negative z-score of mean 21-session turnover divided by mean 252-session turnover; top quintile | All feature-complete eligible securities |
| QV | Top-third book-to-market intersected with F-Score >=7 | Top-third book-to-market on the same complete-case population |
| FAR | Bottom-third trailing 21-session return intersected with F-Score >=7 | Bottom-third return on the same complete-case population |
| VPA | Mean of z(BM), z(annual ROE), and -z(ATO); top quintile | All feature-complete eligible securities |

Standardization clips at median ± 3 scaled MAD, with sample-standard-deviation fallback if MAD is zero. Constant signals are invalid. Direction is applied exactly once. Rank ties use stable security IDs.

F-Score uses three consecutive annual statements so both current and prior ROA can use beginning assets. Its nine binary tests cover profitability, operating cash flow, improving ROA, cash flow greater than income, declining long-term debt/average assets, improving current ratio, no equity issuance, improving gross margin, and improving sales/beginning assets. All required fields must be finite; equity, relevant assets, sales and current-liability denominators must be positive. Annual periods must be 300–430 days apart. Statements older than 550 days from fiscal period end are unavailable. This age cutoff is an explicit implementation default.

BM = latest available common book equity / market capitalization. ROE = annual common net income / average beginning and ending common equity. Financial-company exclusions belong in historical universe eligibility. Tushare accounting fields must be mapped to these definitions rather than treated as interchangeable vendor ratios.

## Commands

Run from the project root using the project virtual environment:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m quant_agent.cli factor-bundle --market cn --stage readiness --config configs/factor_bundles/cn_v1.yaml
.\.venv\Scripts\python.exe -m quant_agent.cli factor-bundle --market us --stage readiness --config configs/factor_bundles/us_v1.yaml
.\.venv\Scripts\python.exe -m quant_agent.cli factor-bundle --market cn --stage evaluate --config configs/factor_bundles/cn_v1.yaml
.\.venv\Scripts\python.exe -m quant_agent.cli factor-family --reports <cn-run>/report.json <us-run>/report.json --output <new-family-report>.json
```

Evaluation is offline. Each invocation creates a new output directory; existing runs cannot be overwritten. `--output` selects a new directory. Missing inputs generate readiness reports, while corrupt inputs raise validation errors.

For Tushare, install the optional dependency with `pip install -e ".[tushare]"` and configure `TUSHARE_TOKEN` locally. Do not place credentials in request files or handovers.

```powershell
.\.venv\Scripts\python.exe -m quant_agent.cli tushare-export --request configs/factor_bundles/tushare_cn_example.json --output <new-raw-directory>
.\.venv\Scripts\python.exe -m quant_agent.cli factor-normalize --raw <raw-directory> --recipe configs/factor_bundles/normalize_cn_example.json --output <new-snapshot-directory>
```

The examples demonstrate calendar filtering and share/capitalization unit conversion only. They intentionally do not produce a complete research snapshot. Verify endpoint permissions and row limits against the subscribed Tushare service. Any response reaching its configured row limit is rejected as possibly truncated. Failed requests are recorded without provider exception messages that might expose credentials.

## Canonical snapshot contract

A snapshot contains `manifest.json` with `market`, `files`, and optional provenance/evidence. Each file entry is `{path, sha256}`. All paths must remain within the snapshot directory. The normalizer adds schema version, unit conventions, recipe hash and raw-manifest hash. SHA256 proves byte integrity, not historical authenticity.

| Table | Required columns and semantics |
|---|---|
| sessions | `date`: observed exchange sessions, not weekdays or future calendar rows |
| bars | `symbol,date,open,high,low,close,volume,amount,tri,tri_open,buyable,sellable,buy_lot,sell_lot`; raw local-currency prices, shares, traded currency, consistent opening/closing total-return wealth indices, side-specific trading flags and lot sizes |
| universe | `symbol,effective_from,effective_to,available_at,listed_at,eligible,industry`; nonoverlapping membership/status intervals; end exclusive; empty end means continuing |
| actions | `event_id,symbol,date,sequence,kind,value,pay_date`; kinds split/dividend/delisting; explicit ordering within security/date; split value is new shares per old share, cash values are per share at that event's sequence |
| costs | `date,commission_bps,min_commission,slippage_bps,stamp_tax_sell_bps`; effective-from schedule covering evaluation |
| float | `symbol,date,available_at,float_shares,market_cap`; free-float shares and common-equity capitalization in units, not ten-thousands |
| financials | `symbol,period,available_at,revision,book_equity,net_income,assets,operating_cashflow,long_debt,current_assets,current_liabilities,revenue,cogs,equity_issued`; annual common-equity accounting definitions; issuance is monetary amount, zero only for confirmed no issuance |

`available_at` is the first usable local exchange session, normalized to a date. Publication dates without time are moved to the next exchange session by normalization. Daily observations documented as available at close use `availability_is_publication_date: false`. Financial vintages must retain original publication/revision identities; latest restated data alone do not establish point-in-time validity. Explicit recipe mappings and exact-key joins avoid guessing financial-field equivalence. Missing required fields remain blocked. Joining statements requires exact symbol, period and vintage keys, with no forward filling across revisions.

The normalizer supports filters, column renaming, copying existing columns, numeric unit scaling, concatenation, and exact one-to-one joins. It does not infer missing corporate actions, fabricate historical membership, infer industry eligibility from current membership, or convert a latest-restated financial feed into original vintages. Those inputs require documented source records. US free-float and delisted financial coverage must be demonstrated separately.

## Portfolio and execution conventions

Request history from 2010. Require 252-session warm-up, then 36 monthly development observations. Rank the top 1,000 eligible securities by trailing 60-session mean traded value with at least 48 observations; require 300 names per market and 80% feature coverage. Eligibility must already exclude financials, ADRs, ETFs, preferred shares and duplicate classes using historical evidence.

Targets allocate 80% gross with a 5% per-name cap, retaining unused cash. Smaller intersections therefore invest less. Benchmarks use the same weighting rule and complete-case population. Exposure differences are reported and must be considered when interpreting interaction comparisons.

The raw mode shares the existing order lifecycle and account/broker infrastructure. Orders are sized after decision close and fill at the next eligible open with costs and slippage. Delay stresses postpone submission while preserving the original target selection; quantities are sized at the delayed close. Sells are processed before buys. Orders have five fill attempts, expire if unfinished, and are cancelled when replaced or affected by a corporate action. No intraday stops run in this research mode.

Raw accounting adjusts quantities and bases on splits, records dividend receivables before payment, and applies explicit delisting settlements. Unknown delisting proceeds cannot be invented. Missing quotes carry the last validated mark and expose stale-position days. A-share sellability uses T+1. Prices are never treated as guaranteed fills merely because a target exists.

Capacity is limited to 1% of the prior 20-session mean volume, requiring 16 observations. This is a declared capacity approximation, not opening-auction evidence. Side restrictions and lot sizes come from dated inputs. Fees are applied per fill and multiplied in stress scenarios. Cash earns zero. Dividend withholding and account-specific taxes require a suitable input policy before stronger operational claims.

## Evaluation and artifacts

Version 1 uses no weight fitting or parameter search. Twelve-month chronological blocks organize reporting after development. Already inspected history remains historical diagnostics. `promotion_allowed` is always false.

Primary evidence is monthly net active return against the assigned comparator. Initial uninvested signal-month observations are removed. Report bootstrap intervals using 10,000 circular samples, three-month blocks and seed 42. The family command applies Holm across all ten slots; missing/blocked tests retain p=1. Research gates require 36 monthly observations, positive net active return, adjusted p<0.05, and positive active returns under costs x2 and delay +1. Costs x4 and delay +2 remain diagnostics. A passing numerical gate does not certify data authenticity or authorize trading.

Artifacts include registration and source hashes, readiness, independent populations, features, open-to-open labels, score coverage, targets, complete-comparator targets, RankIC/ICIR/Newey–West diagnostics, gross quintile outcomes, target industry/size exposures, daily equity/cash/exposure/staleness, attempted orders, fills, rejected sizing intents, account events, active monthly returns and trial logs. VPA also runs fixed leave-one-component-out baseline-cost portfolios. These diagnostics are not replacement primary hypotheses. No winner is selected for live deployment.

## Verification and remaining external requirements

Final verification on November 17 local: **154 tests passed**, with one existing
pandas date-parsing deprecation warning. `git diff --check` passed. The additional
H4 corrective/methodology replay completed and remains NOT PASSED; see E-004.

Current readiness artifacts:

- `artifacts/factor_bundles/cn-20261118T044150260368Z/report.md`
- `artifacts/factor_bundles/us-20261118T044151341303Z/report.md`
- `artifacts/factor_bundles/family_readiness_20261117.json`

Directory identifiers use UTC, while this document uses the user's local date.

Tests cover immutable input checks, missing financial dependencies, financial revision isolation, score direction, label-independent coverage, next-open timing, delayed fills, fees, split/dividend accounting, delisting, T+1, partial fills, suspensions, export truncation, publication lags, family correction and deterministic offline replay. Test data are synthetic and do not support empirical profitability claims.

Before empirical evaluation, obtain Tushare access and prepare documented canonical snapshots. The provider-specific completeness of original financial vintages, historical eligibility, raw corporate actions, open total-return indices and US free float remains unverified. Both initial readiness reports therefore honestly report `blocked_data` for all five strategies. Engineering tests can run without a subscription.
