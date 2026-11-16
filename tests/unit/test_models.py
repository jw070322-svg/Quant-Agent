"""Model construction and device handling."""
import numpy as np
import pytest

from quant_agent.research.models import (_device_of, fit_predict_proba,
                                         make_model)


def test_gbm_default_device_is_cpu():
    model = make_model("gbm", {}, seed=42)
    assert _device_of(model) == "cpu"


def test_gbm_device_param_is_passed_through():
    model = make_model("gbm", {"device": "cuda"}, seed=42)
    assert _device_of(model) == "cuda"


def test_unusable_device_falls_back_to_cpu():
    """A bad device config must degrade to the validated CPU path, not fail
    the run (GPU fallback contract)."""
    model = make_model("gbm", {"device": "notarealdevice"}, seed=42)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    p = fit_predict_proba(model, X, y, X[:10])
    assert len(p) == 10
    assert np.isfinite(p).all()
    assert _device_of(model) == "cpu"  # retried on the CPU path


def test_cuda_fit_when_gpu_available():
    """Real CUDA training test: skipped automatically when no usable CUDA
    device is present (e.g., CI without a GPU)."""
    pytest.importorskip("xgboost")
    model = make_model("gbm", {"device": "cuda", "n_estimators": 20}, seed=42)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    try:
        p = fit_predict_proba(model, X, y, X[:10])
    except Exception as exc:  # no CUDA runtime / unsupported GPU
        pytest.skip(f"no usable CUDA device: {type(exc).__name__}")
    assert np.isfinite(p).all()
    assert _device_of(model) == "cuda"


def test_degenerate_labels_return_nan():
    model = make_model("gbm", {}, seed=42)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    p = fit_predict_proba(model, X, np.ones(50, dtype=int), X[:5])
    assert np.isnan(p).all()
