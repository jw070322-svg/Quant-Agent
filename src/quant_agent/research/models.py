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
        return XGBClassifier(
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
        )
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


def fit_predict_proba(model, X_train: np.ndarray, y_train: np.ndarray,
                      X_pred: np.ndarray) -> np.ndarray:
    """P(event) for X_pred. Requires both classes present in y_train."""
    y = np.asarray(y_train, dtype=int)
    if len(np.unique(y)) < 2:
        # Degenerate fold: no learnable event. Refuse to fabricate — return NaN
        # so the runner records a coverage gap instead of fake probabilities.
        return np.full(len(X_pred), np.nan)
    model.fit(X_train, y)
    return model.predict_proba(X_pred)[:, 1]
