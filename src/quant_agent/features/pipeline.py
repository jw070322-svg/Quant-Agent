"""Feature pipeline: normalized frames -> aligned feature/label panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import WARMUP_SESSIONS, compute_features


def build_symbol_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """date + features + raw tradable prices for one symbol, warm-up rows
    retained with NaN features (they remain valid inference rows later)."""
    feats = compute_features(df, feature_names)
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    for col in ("open_adj", "high_adj", "low_adj", "close_adj", "adj_factor"):
        out[col] = df[col]
    for name in feature_names:
        out[name] = feats[name].values
    return out


def build_panel(frames: dict[str, pd.DataFrame], feature_names: list[str]) -> dict[str, pd.DataFrame]:
    return {sym: build_symbol_frame(df, feature_names) for sym, df in frames.items()}


def assert_prefix_causality(df: pd.DataFrame, feature_names: list[str],
                            probe_offsets: tuple[int, ...] = (1, 7, 30),
                            min_tail: int = WARMUP_SESSIONS + 30) -> None:
    """features(prefix) must equal features(full) restricted to the prefix.

    Guards handover 8.1 principle 1: no feature may look into the future.
    Compares the last common rows before the tail so warm-up transients and
    the appended block cannot mask a violation.
    """
    n = len(df)
    if n < min_tail + 1:
        raise ValueError(f"need >= {min_tail + 1} rows, got {n}")
    cut = n - probe_offsets[-1]
    full = compute_features(df, feature_names)
    for k in probe_offsets:
        prefix = compute_features(df.iloc[: n - k], feature_names)
        cmp_rows = slice(max(WARMUP_SESSIONS, 0), n - k)
        a = prefix.iloc[cmp_rows]
        b = full.iloc[cmp_rows]
        if a.notna().sum().sum() == 0:
            continue
        same_nan = a.isna().equals(b.isna())
        vals_close = np.allclose(
            a.fillna(0.0).to_numpy(dtype=float),
            b.fillna(0.0).to_numpy(dtype=float),
            rtol=1e-9, atol=1e-12, equal_nan=True,
        )
        if not (same_nan and vals_close):
            raise AssertionError(
                f"prefix causality violated for prefix length {n - k}: "
                "features changed when future rows were present"
            )
