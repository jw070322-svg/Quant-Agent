"""Model factory: shallow boosted trees and regularized logistic regression.

Both estimate P(label > 0) for the SAME executable event definition; the
conditional payoff sizes are estimated separately in quant_agent.alpha.
"""
from __future__ import annotations

import numpy as np

MODEL_KINDS = ("gbm", "logistic")


def make_model(kind: str, params: dict, seed: int):
    if kind == "gbm":
        from xgboost import XGBClassifier
        device = str(params.get("device", "cpu")).lower()
        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 5)),
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )
        if device != "cpu":
            # GPU readiness: opt-in via strategy config (model.gbm.device).
            # XGBoost >= 2.0 selects CUDA kernels with device="cuda".
            model.set_params(device=device)
        return model
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=float(params.get("C", 0.5)),
                max_iter=int(params.get("max_iter", 1000)),
                random_state=seed,
            )),
        ])
    raise ValueError(f"unknown model kind: {kind}; expected one of {MODEL_KINDS}")


def _device_of(model) -> str:
    """Read the effective device of a (possibly pipeline-wrapped) tree model.
    xgboost 3.x defaults the sklearn `device` param to None (= CPU)."""
    if hasattr(model, "get_xgb_params"):
        value = model.get_xgb_params().get("device", "cpu")
        if value is None or str(value).strip().lower() in ("", "none", "cpu"):
            return "cpu"
        return str(value).strip().lower()
    return "cpu"


def fit_predict_proba(model, X_train: np.ndarray, y_train: np.ndarray,
                      X_pred: np.ndarray) -> np.ndarray:
    """P(event) for X_pred. Requires both classes present in y_train.

    GPU fallback: if the model was built with a non-CPU device and the fit
    fails (no CUDA runtime, unsupported GPU, out of memory), retry once on
    the CPU path so a bad device configuration degrades to the validated
    CPU behavior instead of failing the run."""
    y = np.asarray(y_train, dtype=int)
    if len(np.unique(y)) < 2:
        # Degenerate fold: no learnable event. Refuse to fabricate — return NaN
        # so the runner records a coverage gap instead of fake probabilities.
        return np.full(len(X_pred), np.nan)
    try:
        model.fit(X_train, y)
    except Exception:
        if _device_of(model) != "cpu":
            model.set_params(device="cpu")
            model.fit(X_train, y)
        else:
            raise
    return model.predict_proba(X_pred)[:, 1]
