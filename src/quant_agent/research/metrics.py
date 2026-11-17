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
    # N2 fix: anchor the high-water mark at the initial equity so a first-
    # period loss is visible in the drawdown (equity alone starts at the
    # post-first-return value and hides it).
    eq_anchored = pd.concat([pd.Series([float(initial)]), eq], ignore_index=True)
    total = float(eq.iloc[-1] / initial - 1.0) if n else 0.0
    ann = float((eq.iloc[-1] / initial) ** (sessions_per_year / max(n, 1)) - 1.0) if n else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(sessions_per_year)) if n > 1 else 0.0
    mean = float(r.mean())
    sharpe = float(mean / r.std(ddof=1) * np.sqrt(sessions_per_year)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    mdd = max_drawdown(eq_anchored)
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
    """Fee-aware FIFO completed-position statistics (W2/N3).

    Each SELL fill is matched FIFO against open BUY lots; the realized pnl of
    the matched quantity = sell notional − sell fee − allocated buy cost
    (buy notional + allocated buy fees). Partial exits and fully closed
    positions are counted separately; remaining open quantity is reported.
    Quantity- and fee-blind position pairing (the legacy defect) is retired.
    """
    empty = {"n_completed_trades": 0, "trade_win_rate": None,
             "avg_trade_pnl": None, "realized_pnl": 0.0,
             "n_completed_positions": 0, "n_partial_exits": 0, "open_qty": 0.0}
    if fills is None or len(fills) == 0:
        return empty
    need = {"symbol", "side", "qty", "price", "fees"}
    if not need.issubset(fills.columns):
        return empty
    realized: list[float] = []
    n_completed_positions = 0
    n_partial_exits = 0
    open_qty: dict[str, float] = {}
    for sym in pd.unique(fills["symbol"]):
        f = fills[fills["symbol"] == sym].sort_values("session")
        lots: list[list[float]] = []  # [qty_remaining, unit_cost_incl_fee]
        for _, r in f.iterrows():
            qty, price, fee = float(r["qty"]), float(r["price"]), float(r["fees"])
            if qty <= 0 or price <= 0:
                continue
            if r["side"] == "BUY":
                lots.append([qty, (qty * price + fee) / qty])
                open_qty[sym] = open_qty.get(sym, 0.0) + qty
                continue
            if fee / qty < 0:
                continue
            sell_proceeds_ps = price - fee / qty
            remaining = qty
            open_before = open_qty.get(sym, 0.0)
            while remaining > 1e-9 and lots:
                take = min(lots[0][0], remaining)
                realized.append(take * (sell_proceeds_ps - lots[0][1]))
                lots[0][0] -= take
                remaining -= take
                if lots[0][0] <= 1e-9:
                    lots.pop(0)
            open_after = open_qty.get(sym, 0.0) - (qty - remaining)
            open_qty[sym] = max(open_after, 0.0)
            if open_after <= 1e-9 and open_before > 0:
                n_completed_positions += 1
            elif open_before > 0:
                n_partial_exits += 1
    if not realized:
        return {**empty, "open_qty": float(sum(open_qty.values()))}
    wins = sum(1 for p in realized if p > 0)
    return {
        "n_completed_trades": int(len(realized)),
        "trade_win_rate": float(wins / len(realized)),
        "avg_trade_pnl": float(np.mean(realized)),
        "realized_pnl": float(sum(realized)),
        "n_completed_positions": int(n_completed_positions),
        "n_partial_exits": int(n_partial_exits),
        "open_qty": float(sum(open_qty.values())),
    }
