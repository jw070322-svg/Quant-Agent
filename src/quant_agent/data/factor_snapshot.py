"""Frozen, versioned factor inputs. Network access is explicit and separate from evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

TABLES = {
    'sessions': {'date'},
    'bars': {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
             'tri', 'tri_open', 'buyable', 'sellable', 'buy_lot', 'sell_lot'},
    'universe': {'symbol', 'effective_from', 'effective_to', 'available_at',
                 'listed_at', 'eligible', 'industry'},
    'actions': {'event_id', 'symbol', 'date', 'sequence', 'kind', 'value', 'pay_date'},
    'costs': {'date', 'commission_bps', 'min_commission', 'slippage_bps', 'stamp_tax_sell_bps'},
    'float': {'symbol', 'date', 'available_at', 'float_shares', 'market_cap'},
    'financials': {'symbol', 'period', 'available_at', 'revision', 'book_equity',
                   'net_income', 'assets', 'operating_cashflow', 'long_debt',
                   'current_assets', 'current_liabilities', 'revenue', 'cogs', 'equity_issued'},
}
NEEDS = {
    'MOM': ['sessions', 'bars', 'universe', 'actions', 'costs'],
    'ATO': ['sessions', 'bars', 'universe', 'actions', 'costs', 'float'],
    'QV': ['sessions', 'bars', 'universe', 'actions', 'costs', 'float', 'financials'],
    'FAR': ['sessions', 'bars', 'universe', 'actions', 'costs', 'financials'],
    'VPA': ['sessions', 'bars', 'universe', 'actions', 'costs', 'float', 'financials'],
}
ENDPOINTS = {
    'cn': ['stock_basic', 'trade_cal', 'daily', 'adj_factor', 'daily_basic',
           'income', 'balancesheet', 'cashflow', 'dividend', 'suspend_d', 'stk_limit'],
    'us': ['us_basic', 'us_tradecal', 'us_daily', 'us_daily_adj', 'us_income',
           'us_balancesheet', 'us_cashflow', 'us_fina_indicator'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_snapshot(root: Path, market: str):
    """Read only hashed files inside root. Missing datasets are blocked, corruption is invalid."""
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        return {}, {'market': market}, ['snapshot manifest missing']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('market') != market:
        raise ValueError('snapshot market mismatch')
    tables, errors = {}, []
    for name, columns in TABLES.items():
        item = manifest.get('files', {}).get(name)
        if not item:
            continue
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != item['sha256']:
            raise ValueError(f'{name}: snapshot integrity failure')
        frame = pd.read_csv(path, dtype={'symbol': str, 'revision': str})
        missing = columns - set(frame.columns)
        if missing:
            errors.append(f'{name}: missing columns {sorted(missing)}')
            continue
        for col in columns & {'available_at', 'date', 'period', 'effective_from',
                              'effective_to', 'listed_at', 'pay_date'}:
            frame[col] = pd.to_datetime(frame[col], errors='raise')
        keys = {'sessions': ['date'], 'bars': ['symbol', 'date'],
                'universe': ['symbol', 'effective_from', 'available_at'],
                'actions': ['event_id'], 'costs': ['date'],
                'float': ['symbol', 'date', 'available_at'],
                'financials': ['symbol', 'period', 'available_at', 'revision']}[name]
        if frame.duplicated(keys).any():
            raise ValueError(f'{name}: duplicate keys')
        if name == 'financials' and frame.duplicated(['symbol', 'period', 'available_at']).any():
            raise ValueError('financials: ambiguous same-session revisions')
        if frame.empty and name != 'actions':
            errors.append(f'{name}: empty required table')
            continue
        for col in columns & {'available_at', 'date', 'period', 'effective_from', 'listed_at'}:
            if frame[col].isna().any() or frame[col].dt.tz is not None or not frame[col].eq(frame[col].dt.normalize()).all():
                raise ValueError(f'{name}: {col} must be non-null local session dates')
        if name == 'universe':
            if not frame.eligible.isin([0, 1]).all() or frame.industry.isna().any():
                raise ValueError('universe: missing eligibility/classification')
            for _, group in frame.groupby('symbol'):
                g = group.sort_values('effective_from')
                if ((g.effective_to.notna()) & (g.effective_to <= g.effective_from)).any():
                    raise ValueError('universe: invalid interval')
                for i in range(len(g)-1):
                    if pd.isna(g.effective_to.iloc[i]) or g.effective_to.iloc[i] > g.effective_from.iloc[i+1]:
                        raise ValueError('universe: overlapping intervals')
        if name == 'actions' and not frame.empty:
            if frame.duplicated(['symbol', 'date', 'sequence']).any():
                raise ValueError('actions: ambiguous event ordering')
            if not frame.kind.isin(['split', 'dividend', 'delisting']).all() or not np.isfinite(frame.value).all() or (frame.value < 0).any():
                raise ValueError('actions: invalid event')
            if (frame.loc[frame.kind.eq('split'), 'value'] <= 0).any():
                raise ValueError('actions: invalid split')
            div = frame.loc[frame.kind.eq('dividend')]
            if div.pay_date.isna().any() or (div.pay_date < div.date).any():
                raise ValueError('actions: payment precedes entitlement')
        tables[name] = frame
    if 'bars' in tables:
        b = tables['bars']
        nums = b[['open', 'high', 'low', 'close', 'volume', 'amount', 'tri', 'tri_open', 'buy_lot', 'sell_lot']]
        if not np.isfinite(nums).all().all() or (nums < 0).any().any():
            raise ValueError('bars: invalid numeric values')
        if (b[['open', 'high', 'low', 'close', 'tri', 'tri_open', 'buy_lot', 'sell_lot']] <= 0).any().any():
            raise ValueError('bars: prices and lots must be positive')
        if ((b.high < b[['open', 'close']].max(axis=1)) |
                (b.low > b[['open', 'close']].min(axis=1))).any():
            raise ValueError('bars: OHLC inconsistent')
        for col in ['buyable', 'sellable']:
            if not b[col].isin([0, 1, False, True]).all():
                raise ValueError(f'bars: {col} must be boolean')
        if not (b[['buy_lot', 'sell_lot']] % 1 == 0).all().all():
            raise ValueError('bars: lot sizes must be integers')
    if 'costs' in tables:
        values = tables['costs'][sorted(TABLES['costs'] - {'date'})]
        if not np.isfinite(values).all().all() or (values < 0).any().any():
            raise ValueError('costs: invalid rates')
    if 'sessions' in tables:
        calendar = tables['sessions'].date
        if 'bars' in tables and not tables['bars'].date.isin(calendar).all():
            raise ValueError('bars: date outside exchange calendar')
        if 'costs' in tables and tables['costs'].date.min() > calendar.min():
            errors.append('costs: schedule starts after calendar')
    return tables, manifest, errors


def readiness(root, market):
    tables, manifest, errors = read_snapshot(Path(root), market)
    strategies = {}
    for name, needs in NEEDS.items():
        missing = [table for table in needs if table not in tables]
        reasons = [e for e in errors if ':' not in e or e.split(':')[0] in needs] + [f'missing {table}' for table in missing]
        strategies[name] = {'status': 'blocked_data' if reasons else 'ready_for_diagnostics',
                            'reasons': reasons}
    return tables, manifest, {'market': market, 'strategies': strategies,
        'tushare_token_present': bool(os.environ.get('TUSHARE_TOKEN')),
        'required_endpoints': ENDPOINTS[market],
        'permissions': manifest.get('permissions', 'not_verified'),
        'limitations': ['Daily bars cannot establish opening-auction capacity.',
                       'US financial coverage and historical free-float require verification.'],
        'evidence': manifest.get('evidence', {})}


def fetch_tushare_request(request_path, output):
    """Explicit immutable raw export; never emits a false PIT-ready snapshot.

    Requests enumerate finite endpoint/parameter/field batches. Row-limit responses
    fail closed: the caller must partition the request rather than accept truncation.
    """
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    if request.get('market') not in ENDPOINTS or not request.get('requests'):
        raise ValueError('market and finite requests required')
    for batch in request['requests']:
        if batch.get('endpoint') not in ENDPOINTS[request['market']] or not isinstance(batch.get('row_limit'), int) or batch['row_limit'] <= 0:
            raise ValueError('invalid endpoint or documented row limit')
        if any(k.lower() in ['token', 'password', 'secret'] for k in batch.get('params', {})):
            raise ValueError('credentials must not appear in export requests')
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        raise ValueError('TUSHARE_TOKEN is not configured')
    import tushare as ts
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    client = ts.pro_api(token)
    records = []
    for index, batch in enumerate(request['requests']):
        endpoint = batch['endpoint']
        if endpoint not in ENDPOINTS[request['market']]:
            raise ValueError(f'unsupported endpoint {endpoint}')
        if not isinstance(batch.get('row_limit'), int) or batch['row_limit'] <= 0:
            raise ValueError('each request requires a positive documented row_limit')
        try:
            frame = client.query(endpoint, **batch.get('params', {}), fields=batch.get('fields', ''))
            if len(frame) >= batch['row_limit']:
                raise ValueError('possible truncation; split the request into smaller periods')
            path = root / f'{index:04d}_{endpoint}.csv'
            frame.to_csv(path, index=False)
            records.append({'request': batch, 'path': path.name, 'sha256': sha(path),
                            'rows': len(frame), 'status': 'retrieved'})
        except Exception as exc:
            # Provider messages may contain credentials. Persist only the class.
            records.append({'request': batch, 'status': 'failed', 'error_type': type(exc).__name__,
                            'reason': 'partition_required' if isinstance(exc, ValueError) else 'provider_request_failed'})
    result = {'market': request['market'], 'retrieved_at': datetime.now(timezone.utc).isoformat(),
              'status': 'raw_export_only', 'requests': records,
              'normalization_required': True}
    (root / 'raw_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def normalize_export(raw_root, recipe_path, output):
    """Declarative, auditable normalization of exports into canonical tables.

    Each source supplies explicit column mappings/scales. Multiple sources join
    ONLY on the recipe's exact vintage keys. No synthetic missing history.
    Unsupported tables remain absent and are reported blocked by readiness.
    """
    root = Path(raw_root).resolve()
    raw = json.loads((root/'raw_manifest.json').read_text(encoding='utf-8'))
    recipe = json.loads(Path(recipe_path).read_text(encoding='utf-8'))
    if recipe['market'] != raw['market']:
        raise ValueError('normalization market mismatch')
    permitted = {r['path']: r['sha256'] for r in raw['requests'] if r['status'] == 'retrieved'}
    frames = {}
    for name, spec in recipe['tables'].items():
        if name not in TABLES:
            raise ValueError(f'unknown canonical table {name}')
        parts = []
        for source in spec['sources']:
            path = (root/source['path']).resolve()
            if not path.is_relative_to(root) or source['path'] not in permitted or sha(path) != permitted[source['path']]:
                raise ValueError('raw export integrity failure')
            frame = pd.read_csv(path, dtype=str)
            for column, allowed in source.get('filters', {}).items():
                frame = frame.loc[frame[column].isin([str(v) for v in allowed])]
            frame = frame.rename(columns=source['columns'])
            frame = frame[list(source['columns'].values())]
            for target, origin in source.get('copies', {}).items():
                frame[target] = frame[origin]
            for col, scale in source.get('scales', {}).items():
                frame[col] = pd.to_numeric(frame[col], errors='raise')*float(scale)
            for col in set(frame) & {'date', 'period', 'available_at', 'listed_at', 'effective_from', 'effective_to', 'pay_date'}:
                frame[col] = pd.to_datetime(frame[col], errors='raise')
            parts.append(frame)
        if spec.get('join_on'):
            frame = parts[0]
            for other in parts[1:]:
                frame = frame.merge(other, on=spec['join_on'], how='outer', validate='one_to_one')
        else:
            frame = pd.concat(parts, ignore_index=True)
        frames[name] = frame
    if 'sessions' not in frames:
        raise ValueError('exchange calendar required for normalization')
    calendar = pd.DatetimeIndex(frames['sessions'].date.sort_values().unique())
    for name, frame in frames.items():
        spec = recipe['tables'][name]
        if 'available_at' in frame and spec.get('availability_is_publication_date', True):
            indices = calendar.searchsorted(frame.available_at, side='right')
            frame['available_at'] = [calendar[i] if i < len(calendar) else pd.NaT for i in indices]
            if frame.available_at.isna().any():
                raise ValueError('calendar does not extend beyond all publication dates')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, frame in frames.items():
        path = out/f'{name}.csv'
        frame.to_csv(path, index=False)
        files[name] = {'path': path.name, 'sha256': sha(path)}
    manifest = {'schema_version': 1, 'market': raw['market'], 'files': files,
                'raw_manifest_sha256': sha(root/'raw_manifest.json'), 'recipe_sha256': sha(recipe_path),
                'price_basis': 'raw', 'quantity_unit': 'shares', 'amount_unit': 'local_currency',
                'permissions': 'raw_requests_only', 'evidence': {'historical_vintages': 'unverified'}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    _, _, report = readiness(out, raw['market'])
    (out/'normalization_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return out
