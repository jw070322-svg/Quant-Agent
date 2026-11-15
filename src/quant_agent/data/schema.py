"""Canonical bar schema and validation.

Principles (handover 8.3 `data` row):
- Invalid bars are QUARANTINED with reasons, never silently clipped/repaired.
- Every normalized dataset records adjustment status and volume units.
- A "complete run claim" is impossible while any universe symbol is unusable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANONICAL_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_factor"]
ADJ_STATUS_UNKNOWN = "unknown_no_factor"
ADJ_STATUS_FACTOR = "raw_with_adj_factor"

QUARANTINE_OHLC_INCONSISTENT = "ohlc_inconsistent"
QUARANTINE_NONFINITE = "nonfinite_or_nonpositive"
QUARANTINE_BAD_VOLUME = "bad_volume"
QUARANTINE_DUPLICATE_DATE = "duplicate_date"
QUARANTINE_EXTREME_MOVE = "extreme_daily_move_review"


def validate_bars(df: pd.DataFrame, *, extreme_move_threshold: float = 0.35) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Return (clean_df, quarantine_records, review_flags).

    clean_df is sorted ascending by date with canonical columns; quarantined
    rows are excluded from the clean frame but reported for reconciliation;
    review-only rows (extreme single-session moves) STAY in the clean frame
    and are reported separately — flagged, never silently dropped or kept
    silently.
    """
    if df.empty:
        return df.copy(), [], []
    df = df.sort_values("date").reset_index(drop=True)
    reasons: dict[int, list[str]] = {}

    price_cols = ["open", "high", "low", "close"]
    required_ok = df[price_cols + ["volume"]].apply(np.isfinite)
    positive = (df[price_cols] > 0).all(axis=1)
    vol_ok = np.isfinite(df["volume"]) & (df["volume"] >= 0)

    bad_finite = ~required_ok.all(axis=1) | ~positive | ~vol_ok
    for i in np.flatnonzero(bad_finite.values):
        reasons.setdefault(i, []).append(
            QUARANTINE_NONFINITE if not positive.iloc[i] or not vol_ok.iloc[i] else QUARANTINE_NONFINITE
        )

    hi_ok = df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12
    for i in np.flatnonzero((~hi_ok | ~lo_ok).values):
        reasons.setdefault(i, []).append(QUARANTINE_OHLC_INCONSISTENT)

    # duplicate dates: keep the last occurrence (matches vendor correction),
    # quarantine the earlier ones.
    dup_mask = df.duplicated(subset="date", keep="last")
    for i in np.flatnonzero(dup_mask.values):
        reasons.setdefault(i, []).append(QUARANTINE_DUPLICATE_DATE)

    # extreme one-session moves: do NOT drop (may be a genuine move), but flag
    # for review as a possible corporate action / adjustment discontinuity.
    ret = df["close"].pct_change().abs()
    for i in np.flatnonzero((ret > extreme_move_threshold).values):
        reasons.setdefault(i, []).append(QUARANTINE_EXTREME_MOVE)

    # Drop rows with a hard violation; the review-only flag (extreme move)
    # keeps the row in the clean frame but is reported for reconciliation.
    hard_reasons = {QUARANTINE_OHLC_INCONSISTENT, QUARANTINE_NONFINITE,
                    QUARANTINE_BAD_VOLUME, QUARANTINE_DUPLICATE_DATE}
    drop_idx = sorted(i for i, rs in reasons.items() if any(r in hard_reasons for r in rs))

    quarantine = []
    for i in drop_idx:
        row = df.iloc[i]
        quarantine.append({
            "row_index": int(i),
            "date": str(row["date"].date()),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
            "reasons": sorted(set(reasons[i])),
        })

    clean = df.drop(index=drop_idx).reset_index(drop=True)

    # review-only records: rows kept in the clean frame but flagged for
    # reconciliation (possible corporate action / adjustment discontinuity)
    review_flags = []
    for i in sorted({j for j, rs in reasons.items()
                     if any(r == QUARANTINE_EXTREME_MOVE for r in rs)}):
        if i in drop_idx:
            continue  # already quarantined for a hard violation
        row = df.iloc[i]
        review_flags.append({
            "row_index": int(i),
            "date": str(row["date"].date()),
            "close": float(row["close"]),
            "reasons": sorted(set(reasons[i])),
        })
    return clean, quarantine, review_flags
