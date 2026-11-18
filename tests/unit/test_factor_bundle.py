"""Hand-worked financial, execution, causality and integrity cases."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_agent.data.factor_snapshot import TABLES, sha, readiness, read_snapshot
from quant_agent.research.factor_strategies import DEFAULTS, financial_scores, construct, feature_panel
from quant_agent.research.factor_bundle import bootstrap, family_summary, run_bundle
from quant_agent.research.multi_factor import factor_ic_stats, summarize_factor_ics, preprocess
from quant_agent.ledger.account import RawAccount
from quant_agent.execution.simulator import run_raw_simulation


def tables(n=300, names=20):
    days = pd.bdate_range('2010-01-01', periods=n)
    bars = []
    for k in range(names):
        p = 20*np.exp((.0001+k*.00004)*np.arange(n))
        bars.append(pd.DataFrame({'date': days, 'symbol': f'S{k:03}', 'open': p, 'close': p,
            'high': p, 'low': p, 'volume': 1000000., 'amount': p*1000000., 'tri': p,
            'tri_open': p, 'buyable': 1, 'sellable': 1, 'buy_lot': 1, 'sell_lot': 1}))
    universe = pd.DataFrame([{'symbol': f'S{k:03}', 'effective_from': days[0],
          'effective_to': pd.NaT, 'available_at': days[0], 'listed_at': days[0],
          'eligible': 1, 'industry': 'industrial'} for k in range(names)])
    return {'sessions': pd.DataFrame({'date': days}), 'bars': pd.concat(bars, ignore_index=True),
       'universe': universe, 'actions': pd.DataFrame(columns=sorted(TABLES['actions'])),
       'costs': pd.DataFrame([{'date': days[0], 'commission_bps': 0., 'min_commission': 0.,
                              'slippage_bps': 0., 'stamp_tax_sell_bps': 0.}])}


def snapshot(root, ts, market='us'):
    root.mkdir()
    files = {}
    for name, frame in ts.items():
        path = root/f'{name}.csv'
        frame.to_csv(path, index=False)
        files[name] = {'path': path.name, 'sha256': sha(path)}
    (root/'manifest.json').write_text(json.dumps({'market': market, 'files': files}))


def test_snapshot_missing_financials_does_not_block_momentum(tmp_path):
    ts = tables()
    snapshot(tmp_path/'snapshot', ts)
    _, _, r = readiness(tmp_path/'snapshot', 'us')
    assert r['strategies']['MOM']['status'] == 'ready_for_diagnostics'
    assert r['strategies']['QV']['status'] == 'blocked_data'
    with (tmp_path/'snapshot'/'bars.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='integrity'):
        read_snapshot(tmp_path/'snapshot', 'us')


def test_overlapping_universe_rejected(tmp_path):
    ts = tables()
    extra = ts['universe'].iloc[[0]].copy()
    extra['effective_from'] += pd.Timedelta(days=1)
    ts['universe'] = pd.concat([ts['universe'], extra])
    snapshot(tmp_path/'snapshot', ts)
    with pytest.raises(ValueError, match='overlapping'):
        read_snapshot(tmp_path/'snapshot', 'us')


def test_financial_score_and_future_revision():
    records = []
    for year, ni, debt, revenue, cogs, ca in [(2017, 4, 30, 100, 70, 40), (2018, 8, 25, 110, 70, 45), (2019, 15, 20, 140, 75, 60)]:
        records.append({'symbol': 'A', 'period': pd.Timestamp(f'{year}-12-31'),
             'available_at': pd.Timestamp(f'{year+1}-04-01'), 'revision': '0',
             'book_equity': 50, 'net_income': ni, 'assets': 100, 'operating_cashflow': 20,
             'long_debt': debt, 'current_assets': ca, 'current_liabilities': 20,
             'revenue': revenue, 'cogs': cogs, 'equity_issued': 0})
    f = pd.DataFrame(records)
    cutoff = pd.Timestamp('2020-05-01')
    score = financial_scores(f, cutoff)
    assert score.loc['A', 'fscore'] == 9
    assert score.loc['A', 'roe'] == .3
    revision = f.iloc[[-1]].copy()
    revision['available_at'] = pd.Timestamp('2021-01-01')
    revision['net_income'] = -900
    revision['revision'] = '1'
    pd.testing.assert_frame_equal(score, financial_scores(pd.concat([f, revision]), cutoff))


def test_raw_split_dividend_and_payment_conserve_nav():
    a = RawAccount(1000., currency='USD', t_plus_1=False)
    a.apply_fill(symbol='A', side='BUY', qty=10, price=100, fees=0, session='2020-01-01', order_id='x')
    a.apply_mark('A', 100, session='2020-01-01')
    a.action({'symbol': 'A', 'date': '2020-01-02', 'kind': 'split', 'value': 2, 'pay_date': None})
    assert a.positions()['A'] == 20
    assert a.equity({}) == 1000
    a.action({'symbol': 'A', 'date': '2020-01-03', 'kind': 'dividend', 'value': 1, 'pay_date': '2020-01-06'})
    a.apply_mark('A', 49, session='2020-01-03')
    assert a.equity({}) == 1000
    assert a.cash == 0
    a.settle('2020-01-06')
    assert a.cash == 20
    assert a.equity({}) == 1000


def test_raw_next_open_delay_and_fees():
    ts = tables(30, 1)
    days = pd.DatetimeIndex(ts['sessions'].date)
    ts['bars'][['open', 'high', 'low', 'close']] = 100.
    ts['costs']['commission_bps'] = 100.
    target = {days[21]: {'S000': .05}}
    a = run_raw_simulation(ts, days[21:], target, market='us', initial_equity=10000.)
    assert a['daily'].iloc[0]['equity'] == 10000.
    assert a['fills'].iloc[0]['session'] == str(days[22].date())
    assert a['fills'].iloc[0]['qty'] == 5
    assert a['daily'].iloc[1]['equity'] == 9995.
    b = run_raw_simulation(ts, days[21:], target, market='us', initial_equity=10000., delay=1)
    assert b['fills'].iloc[0]['session'] == str(days[23].date())


def test_raw_suspension_retains_last_mark_and_expires_orders():
    ts = tables(36, 1)
    days = pd.DatetimeIndex(ts['sessions'].date)
    ts['bars'][['open', 'high', 'low', 'close']] = 100.
    ts['bars'] = ts['bars'].loc[~ts['bars'].date.isin(days[24:])]
    a = run_raw_simulation(ts, days[21:], {days[21]: {'S000': .05}, days[23]: {}}, market='cn', initial_equity=10000.)
    assert a['daily'].equity.iloc[-1] == 10000.
    assert a['daily'].stale_positions.iloc[-1] == 1
    assert 'EXPIRED' in list(a['orders'].state)


def test_feature_coverage_independent_of_labels():
    x = pd.DataFrame({'date': pd.Timestamp('2020-01-01'), 'z_BP': np.arange(20.), 'fwd_ret': np.arange(20.)})
    a = factor_ic_stats(x, ['BP'])
    x['fwd_ret'] = np.nan
    b = factor_ic_stats(x, ['BP'])
    assert a.n_features.iloc[0] == b.n_features.iloc[0] == 20
    assert b.n.iloc[0] == 0
    assert summarize_factor_ics(b, ['BP'], {pd.Timestamp('2020-01-01'): 40})['BP']['coverage'] == .5


def test_low_ato_selection_and_universe_benchmark():
    t = pd.Timestamp('2020-01-31')
    x = pd.DataFrame({'date': t, 'symbol': [f'S{k:03}' for k in range(20)], 'ato': np.arange(1., 21.)})
    cfg = {**DEFAULTS, 'minimum_universe': 20}
    a, b, _, _ = construct(x, pd.DataFrame([{'date': t, 'eligible': 20}]), 'ATO', cfg)
    assert list(a[t]) == ['S000', 'S001', 'S002', 'S003']
    assert len(b[t]) == 20
    assert sum(a[t].values()) == pytest.approx(.2)


def test_future_price_perturbation_does_not_change_past_features():
    ts = tables(330)
    a, _ = feature_panel(ts, DEFAULTS)
    cutoff = a.date.min()
    ts['bars'].loc[ts['bars'].date > cutoff, 'tri'] *= 12
    b, _ = feature_panel(ts, DEFAULTS)
    pd.testing.assert_frame_equal(a.loc[a.date <= cutoff], b.loc[b.date <= cutoff])


def test_bootstrap_and_family_reserve_missing_tests(tmp_path):
    r = bootstrap(np.full(40, .01))
    assert r['mean_annual_active'] == pytest.approx(.12)
    path = tmp_path/'us.json'
    path.write_text(json.dumps({'market': 'us', 'strategies': {'MOM': {'p': .004, 'months': 40, 'economic_pass': True}}}))
    output = family_summary([path], tmp_path/'family.json')
    f = json.loads(output.read_text())
    assert len(f) == 10
    assert f['us_MOM']['holm_p'] == pytest.approx(.04)
    assert f['us_MOM']['promotion_allowed'] is False


def test_bundle_readiness_without_subscription(tmp_path):
    config = tmp_path/'cfg.yaml'
    config.write_text('version: 1\nmarket: us\nsnapshot: missing\n')
    out = run_bundle('us', 'readiness', config, tmp_path/'run')
    r = json.loads((out/'report.json').read_text())
    assert all(x['status'] == 'blocked_data' for x in r['strategies'].values())
    assert (out/'registration.json').exists()


def test_offline_evaluation_pipeline(tmp_path, monkeypatch):
    # Reduced engineering fixture, not an empirical v1 claim.
    import quant_agent.research.factor_bundle as module
    ts = tables(360)
    snapshot(tmp_path/'snapshot', ts)
    monkeypatch.setitem(module.DEFAULTS, 'development_months', 1)
    monkeypatch.setitem(module.DEFAULTS, 'minimum_universe', 20)
    config = tmp_path/'cfg.yaml'
    config.write_text('version: 1\nmarket: us\nsnapshot: snapshot\n')
    out = run_bundle('us', 'evaluate', config, tmp_path/'run')
    report = json.loads((out/'report.json').read_text())
    assert report['strategies']['MOM']['status'] == 'inconclusive'
    assert (out/'MOM'/'base'/'strategy'/'fills.csv').exists()
    assert report['strategies']['QV']['status'] == 'blocked_data'
    other = run_bundle('us', 'evaluate', config, tmp_path/'repeat')
    pd.testing.assert_frame_equal(pd.read_csv(out/'MOM'/'base'/'strategy'/'daily.csv'),
                                  pd.read_csv(other/'MOM'/'base'/'strategy'/'daily.csv'))


def test_low_volatility_direction_and_wrong_direction_screen():
    from quant_agent.research.multi_factor import select_factors
    rng = np.random.default_rng(7)
    x = pd.DataFrame({'date': pd.Timestamp('2020-01-01'), 'vol_5': np.arange(1., 101.), 'size': rng.normal(size=100)})
    z = preprocess(x, ['vol_5'], {'vol_5': -1})
    assert z.z_vol_5.corr(x.vol_5) < -.9
    stats = {'vol_5': {'mean_ic': -.1, 'icir': -.5, 'nw_t': -3., 'coverage': 1.}}
    assert 'vol_5' in select_factors(stats, z, ['vol_5'])['rejected']


def test_unknown_coverage_denominator_fails():
    x = pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'factor': ['BP'],
                      'ic': [.1], 'n': [20], 'n_features': [20]})
    result = summarize_factor_ics(x, ['BP'], {pd.Timestamp('2020-02-01'): 20})
    assert np.isnan(result['BP']['coverage'])


def test_t_plus_one_and_delisting():
    a = RawAccount(1000., currency='CNY', t_plus_1=True)
    a.apply_fill(symbol='A', side='BUY', qty=10, price=10, fees=1, session='2020-01-01', order_id='a')
    with pytest.raises(ValueError, match='sellable'):
        a.apply_fill(symbol='A', side='SELL', qty=1, price=10, fees=1, session='2020-01-01', order_id='b')
    a.action({'symbol': 'A', 'date': '2020-01-02', 'kind': 'delisting', 'value': 3, 'pay_date': None})
    assert a.cash == 929
    assert not a.positions()


def test_raw_partial_fill_and_trading_restriction():
    ts = tables(35, 1)
    days = pd.DatetimeIndex(ts['sessions'].date)
    ts['bars'][['open', 'high', 'low', 'close']] = 100.
    ts['bars']['volume'] = 100.
    ts['bars'].loc[ts['bars'].date.eq(days[22]), 'buyable'] = 0
    result = run_raw_simulation(ts, days[21:], {days[21]: {'S000': .05}}, market='us', initial_equity=10000.)
    assert result['fills'].iloc[0]['session'] == str(days[23].date())
    assert len(result['fills']) == 4  # first of five attempts was blocked
    assert result['orders'].iloc[0].state == 'EXPIRED'
    assert result['orders'].iloc[0].filled_qty == 4


def test_normalizer_explicit_units_and_publication_lag(tmp_path):
    from quant_agent.data.factor_snapshot import normalize_export
    raw = tmp_path/'raw'
    raw.mkdir()
    pd.DataFrame({'cal_date': ['20200102', '20200103', '20200106']}).to_csv(raw/'calendar.csv', index=False)
    pd.DataFrame({'ts_code': ['A'], 'trade_date': ['20200102'], 'ann_date': ['20200102'],
                  'float_share': [2], 'total_mv': [3]}).to_csv(raw/'float.csv', index=False)
    requests = [{'path': name, 'sha256': sha(raw/name), 'status': 'retrieved'} for name in ['calendar.csv', 'float.csv']]
    (raw/'raw_manifest.json').write_text(json.dumps({'market': 'cn', 'requests': requests}))
    recipe = {'market': 'cn', 'tables': {
      'sessions': {'sources': [{'path': 'calendar.csv', 'columns': {'cal_date': 'date'}}]},
      'float': {'sources': [{'path': 'float.csv', 'columns': {'ts_code': 'symbol', 'trade_date': 'date',
                     'ann_date': 'available_at', 'float_share': 'float_shares', 'total_mv': 'market_cap'},
                     'scales': {'float_shares': 10000, 'market_cap': 10000}}]}}}
    rp = tmp_path/'recipe.json'
    rp.write_text(json.dumps(recipe))
    result = normalize_export(raw, rp, tmp_path/'normalized')
    frame = pd.read_csv(result/'float.csv')
    assert frame.float_shares.iloc[0] == 20000
    assert frame.available_at.iloc[0] == '2020-01-03'


def test_tushare_export_truncation_is_recorded(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from quant_agent.data.factor_snapshot import fetch_tushare_request
    monkeypatch.setenv('TUSHARE_TOKEN', 'test-only')
    client = SimpleNamespace(query=lambda *args, **kwargs: pd.DataFrame({'x': [1, 2]}))
    monkeypatch.setitem(sys.modules, 'tushare', SimpleNamespace(pro_api=lambda token: client))
    path = tmp_path/'request.json'
    path.write_text(json.dumps({'market': 'cn', 'requests': [{'endpoint': 'daily', 'row_limit': 2}]}))
    r = fetch_tushare_request(path, tmp_path/'export')
    assert r['requests'][0]['status'] == 'failed'
    assert r['requests'][0]['reason'] == 'partition_required'
    assert 'test-only' not in (tmp_path/'export'/'raw_manifest.json').read_text()


def test_all_five_strategies_compute_on_complete_pit_inputs():
    ts = tables(330)
    shares, financials = [], []
    for k, (symbol, b) in enumerate(ts['bars'].groupby('symbol')):
        shares.append(pd.DataFrame({'symbol': symbol, 'date': b.date, 'available_at': b.date,
              'float_shares': 100000000.*(1 + k/20 + np.arange(len(b))*k/10000),
              'market_cap': 1000000000.*(1+k/20)}))
        for year, ni, debt, revenue, cogs, ca in [(2007, 4, 30, 100, 70, 40), (2008, 8, 25, 110, 70, 45), (2009, 15+k, 20, 140, 75, 60)]:
            financials.append({'symbol': symbol, 'period': pd.Timestamp(f'{year}-12-31'),
                 'available_at': pd.Timestamp(f'{year+1}-04-01'), 'revision': '0',
                 'book_equity': 50, 'net_income': ni, 'assets': 100, 'operating_cashflow': 50,
                 'long_debt': debt, 'current_assets': ca, 'current_liabilities': 20,
                 'revenue': revenue, 'cogs': cogs, 'equity_issued': 0})
    ts['float'] = pd.concat(shares, ignore_index=True)
    ts['financials'] = pd.DataFrame(financials)
    cfg = {**DEFAULTS, 'minimum_universe': 20}
    panel, populations = feature_panel(ts, cfg)
    for strategy in ['MOM', 'ATO', 'QV', 'FAR', 'VPA']:
        targets, references, scores, coverage = construct(panel, populations, strategy, cfg)
        assert len(targets) == len(populations), strategy
        assert np.isfinite(scores.score).all()
        assert coverage.coverage.eq(1).all()
        assert all(sum(w.values()) <= .8+1e-9 for w in targets.values())
