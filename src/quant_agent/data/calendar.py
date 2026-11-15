"""Session calendars inferred from ingested data (v1).

Limitations documented in docs/DECISIONS.md: suspensions mean a symbol may not
have a bar on every market session; portfolio logic must handle missing bars
per symbol instead of assuming the union calendar applies to each name.
"""
from __future__ import annotations

import pandas as pd


def market_sessions(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all symbols' session dates, ascending."""
    all_dates = []
    for sym, df in frames.items():
        if not df.empty:
            all_dates.append(pd.DatetimeIndex(df["date"]))
    if not all_dates:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(pd.concat([pd.Series(d) for d in all_dates]).unique()))


def next_session(sessions: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    idx = sessions.searchsorted(d, side="right")
    return sessions[idx] if idx < len(sessions) else None


def prev_session(sessions: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    idx = sessions.searchsorted(d, side="left") - 1
    return sessions[idx] if idx >= 0 else None
