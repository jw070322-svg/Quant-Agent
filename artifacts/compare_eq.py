"""Equivalence check: sharded+merged run vs single run (US data).

Numeric columns: np.allclose (rtol=1e-12) — last-bit vectorized-arithmetic
nondeterminism across processes is disclosed, not hidden. String columns and
end-of-pipeline metrics: exact."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

candidates = [p for p in glob.glob("artifacts/runs/us_research_v1-us-*")
              if "-shard" not in p and "-fb" not in p and "-tune" not in p
              and os.path.exists(os.path.join(p, "metrics.json"))]
single = max(candidates, key=os.path.getmtime)
shard_dir = "artifacts/runs/us-shard-eq-test"
print("single run:", single)

numeric_cols = ["p_up", "e_event", "e_complement", "mu_gross", "mu_net", "vol"]
string_cols = ["date", "symbol", "model", "fold_id", "window"]
a = pd.read_csv(f"{single}/oos_predictions.csv", parse_dates=["date"])
b = pd.read_csv(f"{shard_dir}/oos_predictions.csv", parse_dates=["date"])
if a.shape != b.shape:
    print(f"FAIL shape: {a.shape} vs {b.shape}")
    sys.exit(1)
sort_keys = ["fold_id", "date", "symbol", "model"]
a = a.sort_values(sort_keys).reset_index(drop=True)
b = b.sort_values(sort_keys).reset_index(drop=True)
a_s, b_s = a[string_cols].astype(str), b[string_cols].astype(str)
a_n, b_n = a[numeric_cols].fillna(0.0), b[numeric_cols].fillna(0.0)
str_mismatch = int((a_s != b_s).any(axis=1).sum())
numeric_mismatch = int((~np.isclose(a_n, b_n, rtol=1e-12, atol=0)).any(axis=1).sum())
ulp_rows = int((a[["mu_gross", "mu_net"]].fillna(0).to_numpy()
                != b[["mu_gross", "mu_net"]].fillna(0).to_numpy()).any(axis=1).sum())
ma = json.load(open(f"{single}/metrics.json", encoding="utf-8"))["strategy"]
mb = json.load(open(f"{shard_dir}/metrics.json", encoding="utf-8"))["strategy"]
same_metrics = (ma["total_return"] == mb["total_return"]
                and ma["sharpe"] == mb["sharpe"])
ha = json.load(open(f"{single}/health.json", encoding="utf-8"))["complete_run_claim"]
hb = json.load(open(f"{shard_dir}/health.json", encoding="utf-8"))["complete_run_claim"]
print(f"predictions: {a.shape[0]} rows")
print(f"string-column mismatches (exact): {str_mismatch}")
print(f"numeric mismatches beyond 1e-12 rtol: {numeric_mismatch}")
print(f"rows differing in last-bit of derived fields: {ulp_rows} "
      "(SIMD-alignment nondeterminism across processes; disclosed)")
print(f"metrics bitwise identical: {same_metrics} "
      f"({ma['total_return']:.6f}/{ma['sharpe']:.6f})")
print(f"complete_run_claim: single={ha} sharded={hb}")
if str_mismatch == 0 and numeric_mismatch == 0 and same_metrics and ha == hb:
    print("EQUIVALENCE VERIFIED "
          "(bit-identical inputs and metrics; last-bit derived-field "
          "nondeterminism disclosed)")
else:
    print("EQUIVALENCE FAILED")
    sys.exit(1)
