"""Chronological, date-grouped walk-forward folds with label-maturity separation.

Fixes handover 4.3/5.3:
- Splits are cut on session DATES, never row counts that can split a date.
- Fitting labels must have matured before the fit cutoff (enforced by the
  runner via labels.training_eligible).
- OOS prediction windows are disjoint -> every OOS (symbol, date) is produced
  exactly once; no overlapping-window double counting.
- Policy (threshold) tuning happens on the first half of each OOS window and
  the selected policy is applied only afterwards (activation schedule),
  so reported dates are never used for selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    fit_cutoff: pd.Timestamp          # labels must mature <= this to be fitted
    predict_start: pd.Timestamp       # first OOS decision session (inclusive)
    predict_end: pd.Timestamp         # last OOS decision session (inclusive)
    tune_start: pd.Timestamp
    tune_end: pd.Timestamp
    report_start: pd.Timestamp
    report_end: pd.Timestamp

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            d[k] = str(pd.Timestamp(v).date())
        return d


def build_folds(sessions: pd.DatetimeIndex, *, initial_train_fraction: float,
                test_sessions: int, step_sessions: int,
                policy_tune_fraction: float, min_fit_sessions: int = 150) -> list[Fold]:
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be in (0, 1)")
    if test_sessions < 4 or step_sessions < 1:
        raise ValueError("test_sessions must be >= 4")
    if not 0.0 < policy_tune_fraction < 1.0:
        raise ValueError("policy_tune_fraction must be in (0, 1)")

    n = len(sessions)
    start_idx = max(int(n * initial_train_fraction), min_fit_sessions)
    folds: list[Fold] = []
    i = 0
    while True:
        fit_idx = start_idx + i * step_sessions
        end_idx = fit_idx + test_sessions
        if fit_idx >= n or end_idx > n:
            break
        tune_len = max(int(test_sessions * policy_tune_fraction), 1)
        folds.append(Fold(
            fold_id=i,
            fit_cutoff=sessions[fit_idx - 1],
            predict_start=sessions[fit_idx],
            predict_end=sessions[end_idx - 1],
            tune_start=sessions[fit_idx],
            tune_end=sessions[fit_idx + tune_len - 1],
            report_start=sessions[fit_idx + tune_len],
            report_end=sessions[end_idx - 1],
        ))
        i += 1
    if not folds:
        raise ValueError("no walk-forward folds fit this history; reduce initial_train_fraction")
    # disjointness guard: OOS windows never overlap
    for a, b in zip(folds, folds[1:]):
        if b.predict_start <= a.predict_end:
            raise AssertionError("overlapping OOS windows generated")
    return folds
