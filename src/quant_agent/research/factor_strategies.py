"""Frozen v1 hypotheses. Scores never depend on forward outcomes."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .multi_factor import mad_winsorize_zscore

CATALOG = {
    'MOM': {'formula': 'mean(z(TRI/lag(TRI,L)-1)), L=63,126,252', 'benchmark': 'MOM252', 'stage': 'A'},
    'ATO': {'formula': '-z(mean(turnover,21)/mean(turnover,252))', 'benchmark': 'UNIVERSE', 'stage': 'B'},
    'QV': {'formula': 'top_third(BM) intersect F_SCORE>=7', 'benchmark': 'VALUE', 'stage': 'C'},
    'FAR': {'formula': 'bottom_third(return21) intersect F_SCORE>=7', 'benchmark': 'REVERSAL', 'stage': 'C'},
    'VPA': {'formula': '(z(BM)+z(ROE)-z(ATO))/3', 'benchmark': 'UNIVERSE', 'stage': 'C'},
}
REQUIRED = {'MOM': ['mom63', 'mom126', 'mom252'], 'ATO': ['ato'],
            'QV': ['bm', 'fscore'], 'FAR': ['return21', 'fscore'], 'VPA': ['bm', 'roe', 'ato']}
DEFAULTS = {'version': 1, 'start': '2010-01-01', 'warmup_sessions': 252,
            'development_months': 36, 'minimum_evaluation_months': 36,
            'minimum_universe': 300, 'maximum_universe': 1000, 'minimum_coverage': .8,
            'gross': .8, 'name_cap': .05, 'initial_equity': 1_000_000.,
            'bootstrap_samples': 10000, 'block_months': 3, 'seed': 42}


def asof_rows(frame, decision, effective='date'):
    """Latest effective observation available at this local session's close.

    Canonical available_at is a normalized first-usable exchange SESSION, not
    an unparsed publication date. Adapters apply next-session disclosure lag.
    """
    f = frame.loc[(frame.available_at <= decision) & (frame[effective] <= decision)]
    return f.sort_values([effective, 'available_at']).drop_duplicates('symbol', keep='last').set_index('symbol')


def financial_scores(frame, decision):
    result = {}
    f = frame.loc[(frame.available_at <= decision) & (frame.period <= decision)]
    f = f.sort_values(['period', 'available_at', 'revision']).drop_duplicates(['symbol', 'period'], keep='last')
    fields = ['book_equity', 'net_income', 'assets', 'operating_cashflow', 'long_debt',
              'current_assets', 'current_liabilities', 'revenue', 'cogs', 'equity_issued']
    for symbol, group in f.groupby('symbol'):
        rows = group.sort_values('period').tail(3)
        if len(rows) < 3 or not np.isfinite(rows[fields]).all().all():
            continue
        a, b, c = [row for _, row in rows.iterrows()]
        if not (300 <= (b.period-a.period).days <= 430 and 300 <= (c.period-b.period).days <= 430):
            continue
        if (decision-c.period).days > 550:
            continue
        if min(a.assets, b.assets, c.assets, b.current_liabilities, c.current_liabilities,
               b.revenue, c.revenue, b.book_equity, c.book_equity) <= 0:
            continue
        roa, old_roa = c.net_income/b.assets, b.net_income/a.assets
        tests = [roa > 0, c.operating_cashflow > 0, roa > old_roa,
                 c.operating_cashflow > c.net_income,
                 c.long_debt/((b.assets+c.assets)/2) < b.long_debt/((a.assets+b.assets)/2),
                 c.current_assets/c.current_liabilities > b.current_assets/b.current_liabilities,
                 c.equity_issued == 0,
                 (c.revenue-c.cogs)/c.revenue > (b.revenue-b.cogs)/b.revenue,
                 c.revenue/b.assets > b.revenue/a.assets]
        result[symbol] = {'fscore': sum(int(v) for v in tests), 'book_equity': c.book_equity,
                          'roe': c.net_income/((b.book_equity+c.book_equity)/2)}
    return pd.DataFrame.from_dict(result, orient='index', columns=['fscore', 'book_equity', 'roe'])


def feature_panel(tables, cfg):
    sessions = pd.DatetimeIndex(tables['sessions'].date.sort_values())
    sessions = sessions[sessions >= pd.Timestamp(cfg['start'])]
    bars = {s: f.set_index('date').sort_index().reindex(sessions) for s, f in tables['bars'].groupby('symbol')}
    months = sessions.to_period('M')
    # Last observed session can be an incomplete month. Only closed months qualify.
    decisions = sessions[:-1][months[:-1] != months[1:]]
    rows, populations = [], []
    for t in decisions:
        loc = sessions.get_loc(t)
        if loc < cfg['warmup_sessions']:
            continue
        u = asof_rows(tables['universe'], t, 'effective_from')
        u = u.loc[u.eligible.eq(1) & (u.effective_to.isna() | (u.effective_to > t))]
        eligible = []
        for symbol, item in u.iterrows():
            if symbol not in bars or sessions.searchsorted(item.listed_at) > loc-252:
                continue
            history = bars[symbol].loc[:t]
            amount = history.amount.tail(60)
            if amount.notna().sum() < 48 or not np.isfinite(history.iloc[-1].close):
                continue
            eligible.append((symbol, float(amount.mean()), item.industry))
        eligible = sorted(eligible, key=lambda v: (-v[1], v[0]))[:cfg['maximum_universe']]
        populations.append({'date': t, 'eligible': len(eligible)})
        floats = asof_rows(tables['float'], t) if 'float' in tables else pd.DataFrame()
        financials = financial_scores(tables['financials'], t) if 'financials' in tables else pd.DataFrame()
        for symbol, liquidity, industry in eligible:
            h = bars[symbol].loc[:t]
            tri = h.tri
            row = {'date': t, 'symbol': symbol, 'industry': industry, 'liquidity': liquidity, 'market_cap': np.nan}
            for lag in [21, 63, 126, 252]:
                row['return21' if lag == 21 else f'mom{lag}'] = tri.iloc[-1]/tri.iloc[-lag-1]-1
            row.update({'ato': np.nan, 'bm': np.nan, 'roe': np.nan, 'fscore': np.nan})
            if symbol in financials.index:
                row.update(financials.loc[symbol].to_dict())
            if symbol in floats.index:
                fl = floats.loc[symbol]
                if (t-fl.date).days <= 7 and fl.market_cap > 0:
                    row['market_cap'] = fl.market_cap
                    row['bm'] = row.get('book_equity', np.nan)/fl.market_cap
            if 'float' in tables:
                # Each historical turnover uses shares known on that historical date.
                fs = tables['float'].loc[tables['float'].symbol.eq(symbol)].copy()
                fs['usable'] = fs[['date', 'available_at']].max(axis=1)
                fs = fs.loc[fs.usable <= t].sort_values(['usable', 'date']).drop_duplicates('usable', keep='last')
                if len(fs):
                    hist = h.tail(252).reset_index()
                    joined = pd.merge_asof(hist, fs[['usable', 'float_shares']], left_on='date', right_on='usable',
                                           tolerance=pd.Timedelta(days=7), direction='backward')
                    turnover = joined.volume/joined.float_shares.where(joined.float_shares > 0)
                    turnover = turnover.where(joined.volume > 0)
                    if turnover.tail(21).count() >= 18 and turnover.count() >= 202 and turnover.mean() > 0:
                        row['ato'] = turnover.tail(21).mean()/turnover.mean()
            rows.append(row)
    columns = ['date', 'symbol', 'industry', 'liquidity', 'market_cap', 'mom63', 'mom126', 'mom252',
               'return21', 'ato', 'bm', 'roe', 'fscore']
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(populations, columns=['date', 'eligible'])


def ranked(series, fraction, largest=True):
    frame = series.rename('score').reset_index()
    frame.columns = ['symbol', 'score']
    return frame.sort_values(['score', 'symbol'], ascending=[not largest, True]).head(math.ceil(len(frame)*fraction)).symbol.tolist()


def weights(symbols, cfg):
    return {s: min(cfg['gross']/len(symbols), cfg['name_cap']) for s in symbols} if len(symbols) else {}


def construct(panel, populations, strategy, cfg):
    schedules, comparators, scores, coverage = {}, {}, [], []
    for t, population in populations.set_index('date').eligible.items():
        xs = panel.loc[panel.date.eq(t)].set_index('symbol')
        valid = np.isfinite(xs[REQUIRED[strategy]]).all(axis=1)
        x = xs.loc[valid]
        ratio = len(x)/population if population else np.nan
        coverage.append({'date': t, 'eligible': population, 'features': len(x), 'coverage': ratio})
        if population < cfg['minimum_universe'] or not np.isfinite(ratio) or ratio < cfg['minimum_coverage']:
            continue
        z = lambda col: mad_winsorize_zscore(x[col]).reindex(x.index)
        if strategy == 'MOM':
            signal = (z('mom63')+z('mom126')+z('mom252'))/3
            selected = ranked(signal.dropna(), .2)
            reference = ranked(x.mom252, .2)
        elif strategy == 'ATO':
            signal = -z('ato')
            selected, reference = ranked(signal.dropna(), .2), list(x.index)
        elif strategy in ['QV', 'FAR']:
            field = 'bm' if strategy == 'QV' else 'return21'
            reference = ranked(x[field], 1/3, strategy == 'QV')
            selected = [s for s in reference if x.loc[s, 'fscore'] >= 7]
            signal = pd.Series([float(s in selected) for s in x.index], index=x.index)
        else:
            signal = (z('bm')+z('roe')-z('ato'))/3
            selected, reference = ranked(signal.dropna(), .2), list(x.index)
        if not np.isfinite(signal).all():
            continue
        schedules[t], comparators[t] = weights(selected, cfg), weights(reference, cfg)
        scores.extend({'date': t, 'symbol': s, 'score': float(v)} for s, v in signal.items())
    return schedules, comparators, pd.DataFrame(scores, columns=['date', 'symbol', 'score']), pd.DataFrame(coverage)
