"""Executable forward returns with explicit maturity (handover 4.2, 8.4).

Primary label (strategy configs): decision after close on session t;
enter at next session's open (t+entry_offset), exit at the open of session
t+exit_offset. For CN this satisfies T+1 sellability by construction.

Invariants:
- A row whose label is not yet known (future bars missing) has label NaN and
  is EXCLUDED from fitting/evaluation but RETAINED for inference. Unknown is
  never cast to a negative class (this was the legacy fabricated-label bug).
- Every label carries its maturity session: fitting requires
  maturity_date <= fit_cutoff, so no fitting label can depend on prices
  outside the fit window (handover 4.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    entry_offset: int = 1
    entry_field: str = "open"
    exit_offset: int = 2
    exit_field: str = "open"

    def __post_init__(self):
        if self.exit_offset <= self.entry_offset:
            raise ValueError("exit_offset must be after entry_offset")
        if self.entry_offset < 1:
            raise ValueError("entry must be strictly after the decision session")
        for f in (self.entry_field, self.exit_field):
            if f not in ("open", "close", "open_adj", "close_adj"):
                raise ValueError(f"unsupported label field: {f}")

    @property
    def horizon_sessions(self) -> int:
        return self.exit_offset


def add_labels(frame: pd.DataFrame, spec: LabelSpec) -> pd.DataFrame:
    """Return frame with columns: label, label_entry_date, label_maturity_date.

    Prices are the adjusted tradable series (open_adj/close_adj by field),
    consistent with simulated fills.
    """
    out = frame.copy()
    entry_px = out[spec.entry_field if spec.entry_field.endswith("_adj") else f"{spec.entry_field}_adj"]
    exit_px = out[spec.exit_field if spec.exit_field.endswith("_adj") else f"{spec.exit_field}_adj"]

    entry = entry_px.shift(-spec.entry_offset)
    exit_ = exit_px.shift(-spec.exit_offset)
    out["label"] = exit_ / entry - 1.0
    out["label_entry_date"] = out["date"].shift(-spec.entry_offset)
    out["label_maturity_date"] = out["date"].shift(-spec.exit_offset)
    return out


def training_eligible(frame: pd.DataFrame, *, fit_cutoff: pd.Timestamp) -> pd.Series:
    """Rows fit-eligible at `fit_cutoff`: label known AND matured by cutoff."""
    known = frame["label"].notna()
    matured = frame["label_maturity_date"].notna() & (frame["label_maturity_date"] <= fit_cutoff)
    return known & matured


def inference_rows(frame: pd.DataFrame) -> pd.Series:
    """All rows (any session) — inference only needs causal features."""
    return pd.Series(True, index=frame.index)
