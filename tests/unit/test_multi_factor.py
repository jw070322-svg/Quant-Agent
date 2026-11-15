"""H4 multi-factor pure functions (NO network, NO sims)."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.research.multi_factor import (icir_weights, mad_winsorize_zscore,
                                               neutralize_size, shrink_weights)


def test_mad_winsorize_clips_extreme_outlier():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 1, 200))
    s.iloc[5] = 100.0
    z = mad_winsorize_zscore(s, k=3.0)
    assert len(z) == len(s)
    assert z.abs().max() < 4.0  # MAD clip bounds the influence of one outlier


def test_mad_winsorize_insufficient_or_degenerate_is_empty():
    assert mad_winsorize_zscore(pd.Series([1.0, 2.0])).empty
    assert mad_winsorize_zscore(pd.Series([7.0] * 60)).empty  # zero MAD & zero std


def test_size_neutralization_removes_size_exposure():
    rng = np.random.default_rng(5)
    n = 200
    log_mv = pd.Series(rng.normal(0, 1, n))
    # factor = size effect + noise: raw correlation with size is strong
    factor = 0.8 * log_mv + pd.Series(rng.normal(0, 0.5, n))
    raw_corr = factor.corr(log_mv)
    resid = neutralize_size(factor, log_mv)
    assert abs(raw_corr) > 0.5
    assert abs(resid.corr(log_mv)) < 0.05  # exposure removed by construction
    assert resid.std(ddof=1) > 0


def test_icir_weights_nonnegative_and_normalized():
    w = icir_weights({"a": 0.5, "b": -0.2, "c": 0.1})
    assert w["b"] == 0.0  # negative ICIR gets zero weight
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["a"] > w["c"] > 0.0


def test_icir_weights_all_nonpositive_falls_back_to_equal():
    w = icir_weights({"a": -0.5, "b": -0.1})
    assert w == {"a": 0.5, "b": 0.5}


def test_shrink_weights_blends_toward_equal():
    w = icir_weights({"a": 0.9, "b": 0.1})
    s = shrink_weights(w, lam=0.5)
    assert s["a"] < w["a"] and s["b"] > w["b"]
    assert sum(s.values()) == pytest.approx(1.0)
