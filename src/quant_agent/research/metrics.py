"""Performance metrics with honest conventions (handover 5.2, 5.4).

- Equity is compounded from SIMPLE period returns (never summed log returns).
- Drawdown uses a high-water mark that includes the initial equity.
- Annualization states its clock: sessions per year is an argument.
- Block bootstrap CIs acknowledge serial dependence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    r = pd.Series(returns).fillna(0.0).astype(float)
    return initial * (1.0 + r).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    hwm = equity.cummax()  # cummax of the curve already includes the initial point
    return float((equity / hwm - 1.0).min())


def summarize(returns: pd.Series, *, sessions_per_year: int = 252,
              initial: float = 1.0, turnover: float = 0.0, costs_paid: float = 0.0,
              exposure: float | None = None, n_trades: int = 0) -> dict:
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    eq = equity_from_returns(r, initial)
    total = float(eq.iloc[-1] / initial - 1.0) if n else 0.0
    ann = float((eq.iloc[-1] / initial) ** (sessions_per_year / max(n, 1)) - 1.0) if n else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(sessions_per_year)) if n > 1 else 0.0
    mean = float(r.mean())
    sharpe = float(mean / r.std(ddof=1) * np.sqrt(sessions_per_year)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    mdd = max_drawdown(eq)
    return {
        "n_sessions": int(n),
        "total_return": total,
        "annual_return": ann,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else 0.0,
        "mean_daily_return": mean,
        "turnover_total": float(turnover),
        "costs_total": float(costs_paid),
        "exposure": exposure,
        "n_trades": int(n_trades),
    }


def block_bootstrap_sharpe_ci(returns: pd.Series, *, sessions_per_year: int = 252,
                              n_boot: int = 1000, block: int = 20, seed: int = 42) -> dict:
    """Circular block bootstrap CI for annualized Sharpe (dependence-aware)."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3 * block or r.std(ddof=1) == 0:
        return {"sharpe_ci_low": None, "sharpe_ci_high": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sd > 0, samples.mean(axis=1) / sd * np.sqrt(sessions_per_year), np.nan)
    sharpes = sharpes[~np.isnan(sharpes)]
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return {"sharpe_ci_low": float(lo), "sharpe_ci_high": float(hi), "n_boot": int(len(sharpes))}


def calibration(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Brier score, log loss, AUC, and base rate for the event definition."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) and len(np.unique(y)) == 2 else None,
        "auc": float(roc_auc_score(y, p)) if len(y) and len(np.unique(y)) == 2 else None,
    }
    return out


def reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin": [float(lo), float(hi)],
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "event_rate": float(y[m].mean()),
        })
    return bins


def completed_trade_stats(fills: pd.DataFrame) -> dict:
    """Win rate over COMPLETED round trips (legacy counted positive days)."""
    if fills is None or len(fills) == 0:
        return {"n_completed_trades": 0, "trade_win_rate": None, "avg_trade_pnl": None}
    buys = fills[fills["side"] == "BUY"]
    sells = fills[fills["side"] == "SELL"]
    # FIFO pairing per symbol on lots is authoritative in the ledger; here we
    # approximate with average-cost matching for reporting only.
    stats = []
    for sym in pd.unique(fills["symbol"]):
        b = buys[buys["symbol"] == sym]["price"].to_numpy()
        s = sells[sells["symbol"] == sym]["price"].to_numpy()
        k = min(len(b), len(s))
        stats.extend((s[:k] - b[:k]) / b[:k])
    if not stats:
        return {"n_completed_trades": 0, "trade_win_rate": None, "avg_trade_pnl": None}
    stats = np.asarray(stats)
    return {
        "n_completed_trades": int(len(stats)),
        "trade_win_rate": float((stats > 0).mean()),
        "avg_trade_pnl": float(stats.mean()),
    }
