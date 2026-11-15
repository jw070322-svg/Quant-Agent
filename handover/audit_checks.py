"""Read-only source/data audit and minimal arithmetic reproductions; stdlib only.
Run: python handover/audit_checks.py
Does not import project modules, load pickle files, call providers, or place trades.
"""
from pathlib import Path
import ast, csv, hashlib, json, math
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
files = sorted(p for p in ROOT.rglob('*') if p.is_file() and 'handover' not in p.relative_to(ROOT).parts)
report = {'scope': str(ROOT), 'counts': dict(Counter(p.suffix for p in files)), 'files': [], 'csv': [], 'syntax': [], 'checks': {}}
for p in files:
    raw = p.read_bytes()
    report['files'].append({'path': p.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if p.suffix == '.py':
        ast.parse(raw.decode('utf-8-sig'), filename=str(p))
        report['syntax'].append({'path': p.relative_to(ROOT).as_posix(), 'lines': len(raw.splitlines()), 'parse': 'PASS'})
    if p.suffix == '.csv':
        rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        cols = list(rows[0]) if rows else []
        dc = next((x for x in ['Date','date','日期','trade_date'] if x in cols), None)
        dates = [r[dc] for r in rows] if dc else []
        names = {k: next((c for c in v if c in cols), None) for k,v in {'o':['Open','open','开盘'],'h':['High','high','最高'],'l':['Low','low','最低'],'c':['Close','close','收盘'],'v':['Volume','volume','vol','成交量']}.items()}
        bad, missing = 0, 0
        for r in rows:
            try:
                o,h,l,c,v = [float(r[names[k]]) for k in ['o','h','l','c','v']]
                missing += int(not all(map(math.isfinite,[o,h,l,c,v])))
                bad += int(min(o,h,l,c)<=0 or v<0 or h<max(o,c,l) or l>min(o,c,h))
            except (ValueError,TypeError,KeyError): missing += 1
        report['csv'].append({'path':p.relative_to(ROOT).as_posix(),'rows':len(rows),'columns':cols,'min_date':min(dates,default=None),'max_date':max(dates,default=None),'duplicates':len(dates)-len(set(dates)),'ascending':dates==sorted(dates),'bad_ohlcv_rows':bad,'missing_or_nonfinite_rows':missing})
checks = report['checks']
# Exact arithmetic of the post-solver normalization for feasible optima in these cases.
checks['optimizer_normalization'] = []
for n in [1,2,3,4]:
    x=[.2]*n
    weights=[v/sum(x)*.8 for v in x]
    checks['optimizer_normalization'].append({'candidates':n,'feasible_solver_weights':x,'returned_weights':weights,'violates_20pct_cap':max(weights)>.2+1e-12})
checks['optimizer_risk']={'solver_weights':[.05]*4,'vols':[.1]*4,'solver_linear_risk':.02,'normalized_linear_risk':.08,'limit':.02}
checks['unknown_label']={'future_return':'NaN','comparison_gt_1pct':int(float('nan')>.01)}
checks['saturated_scale']={str(s):[max(-1,min(1,x*s)) for x in [-1,0,1]] for s in [1,1.5,2,2.5,3,3.5]}
windows=[list(range(s+750+125,s+750+125+250)) for s in [0,125]]
checks['short_walk_forward']={'two_windows_rows':sum(map(len,windows)),'unique_dates':len(set(windows[0]+windows[1])),'overlap':len(set(windows[0])&set(windows[1]))}
checks['index_alignment']={'A_rows':[0,2],'B_rows':[1,3],'effect':'concat(axis=1) aligns global row indices, not trading dates; each row has one stock, so mean is not a daily equal-weight portfolio'}
sel=json.loads((ROOT/'outputs/selected_stocks.json').read_text(encoding='utf-8'))
checks['selected_without_local_data']=[s for s in sel['selected'] if not (ROOT/'data/cn'/f'{s}.csv').exists()]
checks['downloader_resolved_output']=str((ROOT/'data'/'..'/'..'/'data'/'us').resolve())
groups=defaultdict(list)
for f in report['files']:
    if f['path'].endswith(('.pkl','.png','.txt','.csv')): groups[f['sha256']].append(f['path'])
report['identical_artifacts']=[v for v in groups.values() if len(v)>1]
print(json.dumps(report,indent=2,ensure_ascii=False))
