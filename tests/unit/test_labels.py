"""Label invariants: unknown future is NaN (never a negative class) and
every fitting label has matured before the fit cutoff (handover 4.2/4.3)."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.labels.forward import LabelSpec, add_labels, training_eligible


def make_frame(n=30):
    dates = pd.bdate_range("2025-01-01", periods=n)
    px = np.linspace(10, 12, n)
    return pd.DataFrame({
        "date": dates,
        "open_adj": px, "close_adj": px,
        "open": px, "close": px,
    })


def test_tail_labels_are_nan_not_zero():
    df = add_labels(make_frame(), LabelSpec())
    assert df["label"].tail(2).isna().all()
    # legacy bug produced 0.0 for unknown future returns
    assert not (df["label"].tail(2) == 0.0).any()


def test_maturity_strictly_after_decision_date():
    df = add_labels(make_frame(), LabelSpec())
    ok = df["label_maturity_date"].dropna()
    assert (ok > df.loc[ok.index, "date"]).all()


def test_training_eligible_excludes_unmatured_rows():
    df = add_labels(make_frame(), LabelSpec())
    cutoff_last = df["date"].iloc[-1]
    elig = training_eligible(df, fit_cutoff=cutoff_last)
    # last two rows cannot have matured labels at the final session
    assert not elig.iloc[-2:].any()
    assert elig.iloc[:-2].all()
    # cutting earlier excludes anything whose exit bar lies beyond the cutoff
    mid = df["date"].iloc[15]
    elig_mid = training_eligible(df, fit_cutoff=mid)
    assert (df.loc[elig_mid, "label_maturity_date"] <= mid).all()


def test_label_uses_tradable_entry_exit_prices():
    spec = LabelSpec(entry_offset=1, entry_field="open", exit_offset=2, exit_field="open")
    df = add_labels(make_frame(), spec)
    expected = df["open_adj"].shift(-2) / df["open_adj"].shift(-1) - 1.0
    assert np.allclose(df["label"].dropna(), expected.dropna())


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):
        LabelSpec(entry_offset=2, exit_offset=2)
    with pytest.raises(ValueError):
        LabelSpec(entry_offset=0, exit_offset=2)
