"""Offline, immutable bundle evaluation and ten-hypothesis family inference."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ..config import PROJECT_ROOT
from ..data.factor_snapshot import readiness, sha
from ..execution.simulator import run_raw_simulation
from .factor_strategies import CATALOG, DEFAULTS, feature_panel, construct, ranked, weights
from .multi_factor import mad_winsorize_zscore
from .cross_section import newey_west_t
from .trials import TrialLedger

SCENARIOS = {'base': (1, 0), 'cost2': (2, 0), 'cost4': (4, 0), 'delay1': (1, 1), 'delay2': (1, 2)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def bootstrap(active, samples=10000, block=3, seed=42):
    x = np.asarray(active, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        return {'mean_annual_active': None, 'ci95': [None, None], 'p': 1.}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (samples, int(np.ceil(len(x)/block))))
    idx = ((starts[:, :, None]+np.arange(block)) % len(x)).reshape(samples, -1)[:, :len(x)]
    means = x[idx].mean(axis=1)
    p = (1+np.sum(means-x.mean() >= x.mean()))/(samples+1)
    return {'mean_annual_active': float(12*x.mean()), 'ci95': (12*np.quantile(means, [.025, .975])).tolist(), 'p': float(p)}


def metrics(daily):
    r = daily['return'].to_numpy()
    nav = np.r_[1., np.cumprod(1+r)]
    years = max((daily.index[-1]-daily.index[0]).days/365.25, 1/365.25)
    annual = len(r)/years
    return {'total_return': float(nav[-1]-1), 'cagr': float(nav[-1]**(1/years)-1),
            'max_drawdown': float((nav/np.maximum.accumulate(nav)-1).min()),
            'sharpe_zero_cash': finite(r.mean()/r.std(ddof=1)*np.sqrt(annual)) if r.std(ddof=1) > 0 else None,
            'mean_exposure': float(daily.exposure.mean()), 'annual_turnover': float(daily.turnover.sum()/years),
            'stale_position_days': int(daily.stale_positions.sum())}


def labels(tables, decisions):
    """Scheduled open-to-open labels, kept separate from features and selection."""
    sessions = pd.DatetimeIndex(tables['sessions'].date.sort_values())
    b = tables['bars'].set_index(['date', 'symbol'])
    rows = []
    for entry, end in zip(decisions[:-1], decisions[1:]):
        i, j = sessions.get_loc(entry)+1, sessions.get_loc(end)+1
        if j >= len(sessions):
            continue
        for s in tables['bars'].symbol.unique():
            value = np.nan
            if 'tri_open' in b and (sessions[i], s) in b.index and (sessions[j], s) in b.index:
                value = b.loc[(sessions[j], s), 'tri_open']/b.loc[(sessions[i], s), 'tri_open']-1
            rows.append({'date': entry, 'symbol': s, 'entry_at': sessions[i],
                         'label_end_at': sessions[j], 'return': value})
    return pd.DataFrame(rows, columns=['date', 'symbol', 'entry_at', 'label_end_at', 'return'])


def run_bundle(market, stage, config_path, output=None):
    config_path = Path(config_path).resolve()
    supplied = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if supplied.get('market') != market or supplied.get('version') != 1:
        raise ValueError('configuration market/version mismatch')
    cfg = {**DEFAULTS, **supplied}
    # Changing registered constants creates a different experiment, not v1.
    for k, v in DEFAULTS.items():
        if cfg[k] != v:
            raise ValueError(f'frozen v1 parameter changed: {k}')
    root = (config_path.parent / cfg['snapshot']).resolve()
    out = Path(output) if output else PROJECT_ROOT/'artifacts'/'factor_bundles'/f'{market}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
    out.mkdir(parents=True, exist_ok=False)
    trials = TrialLedger(out, out.name)
    trials.log('registration', market=market, primary_hypotheses=10, catalog=CATALOG)
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((PROJECT_ROOT/'src'/'quant_agent').rglob('*.py')))).hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
        patch = subprocess.check_output(['git', 'diff', '--', 'src', 'configs', 'pyproject.toml'], cwd=PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError):
        commit, patch = 'unavailable', b''
    tables, manifest, ready = readiness(root, market)
    dump(out/'registration.json', {'market': market, 'catalog': CATALOG, 'config': cfg,
         'config_sha256': sha(config_path), 'source_sha256': source_hash, 'code_commit': commit,
         'dirty_patch_sha256': hashlib.sha256(patch).hexdigest(), 'snapshot': manifest,
         'registered_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
         'numpy': np.__version__, 'pandas': pd.__version__, 'family_size': 10,
         'parent_studies': ['H2', 'H3', 'H4', 'market_backtest_20261117'],
         'evidence_kind': 'historical_diagnostic', 'promotion_allowed': False})
    dump(out/'readiness.json', ready)
    trials.log('readiness', strategies=ready['strategies'])
    report = {'market': market, 'stage': stage, 'strategies': ready['strategies'], 'promotion_allowed': False}
    if stage == 'evaluate' and all(k in tables for k in ['bars', 'sessions', 'universe']):
        panel, population = feature_panel(tables, cfg)
        panel.to_csv(out/'features.csv', index=False)
        population.to_csv(out/'population.csv', index=False)
        decisions = list(population.date)
        lab = labels(tables, decisions)
        lab.to_csv(out/'labels.csv', index=False)
        if len(decisions) > cfg['development_months']:
            first = decisions[cfg['development_months']]
            test_dates = decisions[cfg['development_months']:]
            sessions = pd.DatetimeIndex(tables['sessions'].date)
            sessions = sessions[(sessions >= first) & (sessions <= decisions[-1])]
            dump(out/'folds.json', [{'start': test_dates[i], 'end': test_dates[min(i+11, len(test_dates)-1)]}
                                  for i in range(0, len(test_dates), 12)])
            for strategy in CATALOG:
                if ready['strategies'][strategy]['status'] == 'blocked_data':
                    continue
                targets, reference, scores, coverage = construct(panel, population, strategy, cfg)
                d = out/strategy
                d.mkdir()
                coverage.to_csv(d/'coverage.csv', index=False)
                scores.to_csv(d/'scores.csv', index=False)
                if any(t not in targets for t in test_dates):
                    report['strategies'][strategy] = {'status': 'blocked_data', 'reasons': ['inadequate universe, coverage or dispersion in evaluation months']}
                    continue
                targets = {t: w for t, w in targets.items() if t >= first}
                reference = {t: w for t, w in reference.items() if t >= first}
                dump(d/'targets.json', {str(t): w for t, w in targets.items()})
                dump(d/'benchmark_targets.json', {str(t): w for t, w in reference.items()})
                joined = scores.merge(lab, on=['date', 'symbol'], how='left')
                # Exposure and sorted-return diagnostics are descriptive, not extra primary tests.
                exposure_rows, group_rows = [], []
                for t, target in targets.items():
                    xs = panel.loc[panel.date.eq(t)].set_index('symbol')
                    for symbol, weight in target.items():
                        exposure_rows.append({'date': t, 'symbol': symbol, 'weight': weight,
                              'industry': xs.loc[symbol, 'industry'], 'market_cap': xs.loc[symbol, 'market_cap']})
                pd.DataFrame(exposure_rows).to_csv(d/'exposures.csv', index=False)
                for t, group in joined.groupby('date'):
                    ordered = group.sort_values(['score', 'symbol']).copy()
                    ordered['quintile'] = np.minimum(5, np.arange(len(ordered))*5//max(len(ordered), 1)+1)
                    for q, g in ordered.groupby('quintile'):
                        group_rows.append({'date': t, 'quintile': q, 'gross_forward_return': g['return'].mean(),
                                           'features': len(g), 'outcomes': g['return'].count()})
                pd.DataFrame(group_rows).to_csv(d/'quintiles.csv', index=False)
                ics = []
                for t, g in joined.loc[joined.date >= first].groupby('date'):
                    valid = g[['score', 'return']].replace([np.inf, -np.inf], np.nan).dropna()
                    ic = valid.score.corr(valid['return'], method='spearman') if len(valid) >= 10 and valid.score.nunique() > 1 else np.nan
                    ics.append({'date': t, 'ic': ic, 'outcomes': len(valid)})
                icframe = pd.DataFrame(ics)
                icframe.to_csv(d/'ic.csv', index=False)
                record = {'status': 'inconclusive', 'scenarios': {}, 'rank_ic': finite(icframe.ic.mean()),
                          'icir': finite(icframe.ic.mean()/icframe.ic.std()) if icframe.ic.std() > 0 else None,
                          'nw_t': finite(newey_west_t(icframe.ic.dropna()))}
                for scenario, (cost, delay) in SCENARIOS.items():
                    runs = []
                    for name, schedule in [('strategy', targets), ('benchmark', reference)]:
                        trials.log('simulation_started', strategy=strategy, scenario=scenario, portfolio=name)
                        sim = run_raw_simulation(tables, sessions, schedule, market=market,
                               initial_equity=cfg['initial_equity'], cost_multiple=cost,
                               delay=delay if name == 'strategy' else 0,
                               run_id=f'{market}-{strategy}-{scenario}-{name}')
                        sd = d/scenario/name
                        sd.mkdir(parents=True)
                        for key, frame in sim.items():
                            frame.to_csv(sd/f'{key}.csv', index=(key == 'daily'))
                        runs.append(sim['daily'])
                        trials.log('simulation_completed', strategy=strategy, scenario=scenario, portfolio=name)
                    monthly = [(1+r['return']).groupby(r.index.to_period('M')).prod()-1 for r in runs]
                    # Initial signal-month observation contains no invested horizon.
                    active = (monthly[0]-monthly[1]).iloc[1:]
                    active.to_csv(d/f'{scenario}_active_monthly.csv', header=['active_return'])
                    record['scenarios'][scenario] = {'strategy': metrics(runs[0]), 'benchmark': metrics(runs[1]),
                        'mean_active': finite(active.mean()), 'months': len(active)}
                    if scenario == 'base':
                        record.update(bootstrap(active, cfg['bootstrap_samples'], cfg['block_months'], cfg['seed']))
                        record['months'] = len(active)
                record['economic_pass'] = all(record['scenarios'][s]['mean_active'] is not None and
                      record['scenarios'][s]['mean_active'] > 0 for s in ['base', 'cost2', 'delay1'])
                record['status'] = 'family_review_pending' if record['months'] >= 36 else 'inconclusive'
                report['strategies'][strategy] = record
                if strategy == 'VPA':
                    # Fixed leave-one-component-out diagnostics; never select a winner here.
                    for omitted in ['bm', 'roe', 'ato']:
                        trials.log('diagnostic_ablation', strategy='VPA', omitted=omitted)
                        ablation = {}
                        for t in targets:
                            x = panel.loc[panel.date.eq(t)].set_index('symbol')
                            x = x.loc[np.isfinite(x[['bm', 'roe', 'ato']]).all(axis=1)]
                            parts = [mad_winsorize_zscore(x[c])*(-1 if c == 'ato' else 1)
                                     for c in ['bm', 'roe', 'ato'] if c != omitted]
                            score = (parts[0]+parts[1])/2
                            ablation[t] = weights(ranked(score.dropna(), .2), cfg)
                        sim = run_raw_simulation(tables, sessions, ablation, market=market,
                            initial_equity=cfg['initial_equity'], run_id=f'{market}-VPA-without-{omitted}')
                        ad = d/f'diagnostic_without_{omitted}'
                        ad.mkdir()
                        for key, frame in sim.items():
                            frame.to_csv(ad/f'{key}.csv', index=(key == 'daily'))
                        dump(ad/'metrics.json', metrics(sim['daily']))
        else:
            for key, value in report['strategies'].items():
                if value['status'] != 'blocked_data':
                    report['strategies'][key] = {'status': 'blocked_data', 'reasons': ['insufficient post-warmup development history']}
    dump(out/'report.json', report)
    (out/'report.md').write_text('# Factor bundle research\n\n' +
        f'Market: {market}. Stage: {stage}. Historical diagnostics only.\n\n' +
        '\n'.join(f"- {k}: **{v['status']}**. " + '; '.join(v.get('reasons', [])) for k, v in report['strategies'].items())+
        '\n\nNo strategy is approved for trading. Tushare permissions and historical coverage require separate evidence.\n', encoding='utf-8')
    dump(out/'artifacts.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return out


def family_summary(paths, output):
    family = {(market, name): {'status': 'blocked_data', 'p': 1.} for market in ['cn', 'us'] for name in CATALOG}
    seen = set()
    for path in paths:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        market = report['market']
        if market not in ['cn', 'us'] or market in seen:
            raise ValueError('require at most one report per market')
        seen.add(market)
        for name, r in report['strategies'].items():
            family[market, name] = r
    ordered = sorted(family, key=lambda k: family[k].get('p', 1.))
    previous = 0.
    for i, key in enumerate(ordered):
        r = family[key]
        p = r.get('p', 1.)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid p value')
        previous = max(previous, min(1., (10-i)*p))
        r['holm_p'] = previous
        r['research_gate_pass'] = bool(previous < .05 and r.get('months', 0) >= 36 and r.get('economic_pass', False))
        # Historical diagnostics cannot establish untouched evidence or execution validity.
        r['promotion_allowed'] = False
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    dump(out, {f'{m}_{n}': r for (m, n), r in family.items()})
    return out
