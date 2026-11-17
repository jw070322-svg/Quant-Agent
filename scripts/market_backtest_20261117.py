"""Standalone exploratory ETF backtest. Run with --fetch to snapshot public data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MARKETS = {
    'US': (['SPY', 'QQQ', 'IWM', 'DIA'], '2026-11-16', 5, 252),
    'CN': (['sh510300', 'sh510500', 'sh510050', 'sz159915'], '2026-11-17', 10, 244),
}
START = '2020-01-01'
METHODS = ['buy_hold', 'equal_weight', 'trend_200', 'momentum_252', 'momentum_63_126_252']
SCENARIOS = {'base': (1, 0), 'double_cost': (2, 0), 'delay_one_session': (1, 1)}


def download(symbol, market, cutoff):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        session.headers['User-Agent'] = 'Mozilla/5.0'

        def get(url, params=None):
            response = session.get(url, params=params, timeout=18)
            response.raise_for_status()
            return response

        if market == 'US':
            from akshare.stock.stock_us_sina import py_mini_racer, zh_js_decode

            base = 'https://finance.sina.com.cn'
            prices_url = f'{base}/staticdata/us/{symbol}'
            factors_url = f'{base}/us_stock/company/reinstatement/{symbol}_qfq.js'
            encoded = get(prices_url).text.split('=', 1)[1].split(';', 1)[0].strip().strip('"')
            js = py_mini_racer.MiniRacer()
            js.eval(zh_js_decode)
            prices = pd.DataFrame(js.call('d', encoded))
            prices['date'] = pd.to_datetime(prices['date']).dt.tz_localize(None).dt.normalize()
            factors = json.JSONDecoder().raw_decode(get(factors_url).text.split('=', 1)[1].lstrip())[0]
            factors = pd.DataFrame(factors['data']).rename(columns={'d': 'date'})
            factors['date'] = pd.to_datetime(factors['date'])
            prices = pd.merge_asof(prices.sort_values('date'), factors.sort_values('date'), on='date')
            for field in ['open', 'close']:
                prices[field] = prices[field].astype(float) * prices['f'].astype(float) + prices['c'].astype(float)
            sources = [prices_url, factors_url]
        else:
            url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
            secid = ('1.' if symbol.startswith('sh') else '0.') + symbol[2:]
            params = {'secid': secid, 'klt': 101, 'fqt': 2, 'beg': '20180101',
                      'end': cutoff.replace('-', ''), 'lmt': 3000, 'fields1': 'f1,f2,f3,f4,f5,f6',
                      'fields2': 'f51,f52,f53,f54,f55,f56'}
            response = get(url, params)
            rows = [row.split(',')[:3] for row in response.json()['data']['klines']]
            prices = pd.DataFrame(rows, columns=['date', 'open', 'close'])
            sources = [response.url]

    prices = prices[['date', 'open', 'close']].copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.loc[prices['date'].between('2018-01-01', cutoff)].set_index('date').sort_index()
    prices = prices.astype(float)
    if prices.index.has_duplicates or len(prices) < 1000 or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError(f'{symbol}: failed daily bar validation')
    jumps = pd.concat([prices['close'].pct_change(),
                       prices['open'] / prices['close'].shift() - 1], axis=1).abs().max(axis=1)
    if jumps.max() > 0.30:
        raise ValueError(f'{symbol}: discontinuities {jumps[jumps > .30].to_dict()}')
    return prices, sources


def targets(close, method, evaluation_start, delay):
    dates = close.index
    schedule = {}
    equal = np.full(close.shape[1], 1 / close.shape[1])
    start_pos = dates.searchsorted(evaluation_start)
    if method == 'buy_hold':
        schedule[dates[start_pos]] = np.eye(close.shape[1])[0]
        return schedule
    if method == 'equal_weight':
        schedule[dates[start_pos]] = equal
    p = dates.to_period('M')
    month_ends = np.flatnonzero(p[:-1] != p[1:])
    if method == 'trend_200':
        ma = close.rolling(200, min_periods=200).mean().iloc[:, 0]
    elif method.startswith('momentum'):
        horizons = [252] if method == 'momentum_252' else [63, 126, 252]
        scores = [close.pct_change(h, fill_method=None).rank(axis=1, pct=True) for h in horizons]
        score = sum(scores) / len(scores)
    for i in month_ends:
        execution = i + 1 + (0 if method == 'equal_weight' else delay)
        if not start_pos <= execution < len(dates):
            continue
        if method == 'equal_weight':
            weight = equal
        elif method == 'trend_200':
            weight = np.zeros(close.shape[1])
            weight[0] = float(close.iloc[i, 0] > ma.iloc[i])
        else:
            weight = np.zeros(close.shape[1])
            order = score.iloc[i].sort_values(ascending=False, kind='stable').dropna()
            if len(order) != close.shape[1]:
                raise ValueError(f'missing ranking inputs on {dates[i]}')
            weight[close.columns.get_indexer(order.index[:2])] = 0.5
        schedule[dates[execution]] = weight
    return schedule


def simulate(open_, close, schedule, cost_bps, start=START):
    fee_rate = cost_bps / 10_000
    positions = np.zeros(close.shape[1])
    cash, previous_nav = 1.0, 1.0
    rows = []
    pending = None
    for date, op, cl in zip(close.index, open_.to_numpy(), close.to_numpy()):
        if date < pd.Timestamp(start):
            continue
        if date in schedule:
            pending = schedule[date]
        can_rebalance = np.isfinite(op).all()
        op = np.where(np.isfinite(op), op, cl)
        values = positions * op
        before = float(values.sum() + cash)
        fee = turnover = 0.0
        if pending is not None and can_rebalance:
            weight, pending = pending, None
            after = before
            for _ in range(30):
                amount = float(np.abs(after * weight - values).sum())
                updated = before - fee_rate * amount
                if abs(updated - after) < 1e-13:
                    after = updated
                    break
                after = updated
            dollars = after * weight
            fee = fee_rate * float(np.abs(dollars - values).sum())
            turnover = float(np.abs(dollars - values).sum()) / before
            cash = before - fee - float(dollars.sum())
            assert cash >= -1e-11
            positions = dollars / op
        equity = float(positions @ cl + cash)
        rows.append((date, equity, equity / previous_nav - 1, turnover, fee / before, 1 - cash / equity))
        previous_nav = equity
    return pd.DataFrame(rows, columns=['date', 'nav', 'return', 'turnover', 'cost_fraction', 'exposure']).set_index('date')


def metrics(frame, annual):
    returns = frame['return'].to_numpy()
    nav = np.r_[1., np.cumprod(1 + returns)]
    sd = returns.std(ddof=1)
    return {'sessions': len(returns), 'total': nav[-1] - 1,
            'annualized': nav[-1] ** (annual / len(returns)) - 1,
            'sharpe_zero_cash': returns.mean() / sd * np.sqrt(annual) if sd else None,
            'drawdown': (nav / np.maximum.accumulate(nav) - 1).min(),
            'avg_exposure': frame['exposure'].mean(),
            'annualized_turnover': frame['turnover'].mean() * annual}


def paired(frame, benchmark, annual, seed=42):
    v = (frame['return'] - benchmark['return']).to_numpy()
    blocks = 21
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(v), (4000, int(np.ceil(len(v) / blocks))))
    idx = (starts[:, :, None] + np.arange(blocks)) % len(v)
    samples = v[idx.reshape(len(starts), -1)[:, :len(v)]].mean(axis=1)
    ci = np.quantile(samples * annual, [.025, .975])
    p = (1 + np.sum(samples - v.mean() >= v.mean())) / (1 + len(samples))
    return {'annualized_arithmetic_excess': v.mean() * annual, 'ci95': ci.tolist(), 'p_one_sided': float(p)}


def run(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    results = {}
    for market, (symbols, cutoff, cost, annual) in MARKETS.items():
        for symbol in symbols:
            digest = hashlib.sha256((root / f'{symbol}.csv').read_bytes()).hexdigest()
            if digest != manifest['series'][symbol]['sha256']:
                raise ValueError(f'{symbol}: snapshot hash mismatch')
        frames = {s: pd.read_csv(root / f'{s}.csv', index_col='date', parse_dates=True) for s in symbols}
        by_field = {field: pd.concat({s: f[field] for s, f in frames.items()}, axis=1)
                    for field in ['open', 'close']}
        open_, close = by_field.values()
        first_common = max(frame.index.min() for frame in frames.values())
        close = close.loc[first_common:]
        open_ = open_.reindex(close.index)
        missing = close.loc[START:].isna()
        missing_bars = [(str(date.date()), symbol) for date in missing.index
                        for symbol in missing.columns if missing.at[date, symbol]]
        if len(missing_bars) > 5 or close.iloc[-1].isna().any():
            raise ValueError(f'{market}: incomplete evaluation history: {missing_bars}')
        month_end = close.index.to_period('M')[:-1] != close.index.to_period('M')[1:]
        if close.iloc[:-1].loc[month_end].loc[START:].isna().any().any():
            raise ValueError(f'{market}: missing month-end signal inputs')
        close = close.ffill()
        simulations = {}
        for scenario, (multiple, delay) in SCENARIOS.items():
            for method in METHODS:
                schedule = targets(close, method, START, delay)
                simulations[scenario, method] = simulate(open_, close, schedule, cost * multiple)
        pd.concat(simulations, names=['scenario', 'method']).to_csv(root / f'{market}_daily.csv')
        market_results = {'first': str(simulations['base', 'buy_hold'].index.min().date()),
                          'last': str(close.index.max().date()), 'missing_bars': missing_bars,
                          'recent_first': str(simulations['base', 'buy_hold'].tail(252).index[0].date()),
                          'methods': {}}
        for method in METHODS:
            base = simulations['base', method]
            benchmark = 'buy_hold' if method == 'trend_200' else 'equal_weight'
            record = {scenario: metrics(simulations[scenario, method], annual) for scenario in SCENARIOS}
            record['recent252'] = metrics(base.tail(252), annual)
            record['annual_returns'] = {str(year): metrics(group, annual)['total']
                                        for year, group in base.groupby(base.index.year)}
            if method not in ['buy_hold', 'equal_weight']:
                record['vs_benchmark'] = paired(base, simulations['base', benchmark], annual)
                record['benchmark'] = benchmark
                record['stress_excess_total'] = {
                    scenario: record[scenario]['total'] - metrics(simulations[scenario, benchmark], annual)['total']
                    for scenario in SCENARIOS if scenario != 'base'}
                record['recent_excess_total'] = record['recent252']['total'] - metrics(
                    simulations['base', benchmark].tail(252), annual)['total']
            market_results['methods'][method] = record
        market_results['composite_vs_single_horizon'] = paired(
            simulations['base', 'momentum_63_126_252'], simulations['base', 'momentum_252'], annual)
        results[market] = market_results
    tests = sorted((record['vs_benchmark'] for market in results.values()
                    for record in market['methods'].values() if 'vs_benchmark' in record),
                   key=lambda item: item['p_one_sided'])
    adjusted = 0.0
    for i, test in enumerate(tests):
        adjusted = max(adjusted, min(1.0, (len(tests) - i) * test['p_one_sided']))
        test['holm_p'] = adjusted
    (root / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    (root / 'run_manifest.json').write_text(json.dumps({
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        'results_sha256': hashlib.sha256((root / 'results.json').read_bytes()).hexdigest(),
        'markets': MARKETS, 'scenarios': SCENARIOS, 'evaluation_start': START,
        'numpy': np.__version__, 'pandas': pd.__version__,
    }, indent=2), encoding='utf-8')
    for market, report in results.items():
        print(market, report['first'], report['last'])
        for method, record in report['methods'].items():
            print(method, *(round(record['base'][key], 4) for key in ['annualized', 'sharpe_zero_cash', 'drawdown']),
                  round(record['recent252']['total'], 4), record.get('vs_benchmark', {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    root = args.output
    if args.fetch:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f'{root} already contains artifacts')
        jobs = [(s, m, cutoff) for m, (symbols, cutoff, _, _) in MARKETS.items() for s in symbols]
        with ThreadPoolExecutor(max_workers=6) as executor:
            downloads = list(executor.map(lambda job: download(*job), jobs))
        items = {}
        for (symbol, market, cutoff), (frame, sources) in zip(jobs, downloads):
            path = root / f'{symbol}.csv'
            frame.to_csv(path)
            items[symbol] = {'market': market, 'requested_cutoff': cutoff, 'sources': sources,
                             'rows': len(frame), 'first': str(frame.index.min().date()),
                             'last': str(frame.index.max().date()),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (root / 'manifest.json').write_text(json.dumps({'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'script_sha256': hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(), 'series': items}, indent=2), encoding='utf-8')
    run(root)


if __name__ == '__main__':
    main()
