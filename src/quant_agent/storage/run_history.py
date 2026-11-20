"""Read-only access to recorded research runs and their evidence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..util import read_json

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


class RunNotFoundError(FileNotFoundError):
    pass


class SymbolNotFoundError(KeyError):
    pass


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg = d / "config.json"
        if cfg.exists():
            meta = read_json(cfg)
            runs.append({
                "run_id": meta["run_id"],
                "strategy": meta.get("strategy"),
                "market": meta.get("market"),
                "created_utc": meta.get("created_utc"),
            })
    return runs


def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    if not (d / "config.json").exists():
        raise RunNotFoundError(
            f"run '{run_id}' not found under {RUNS_DIR}; known runs: "
            f"{[r['run_id'] for r in list_runs()]}")
    return d


def get_run(run_id: str) -> dict:
    d = _run_dir(run_id)
    return {
        "run_id": run_id,
        "provenance": read_json(d / "config.json"),
        "metrics": read_json(d / "metrics.json") if (d / "metrics.json").exists() else None,
        "health": read_json(d / "health.json") if (d / "health.json").exists() else None,
    }


def get_signals(run_id: str, *, symbol: str | None = None, date: str | None = None) -> pd.DataFrame:
    """OOS predictions (as-of estimates, not realized outcomes)."""
    d = _run_dir(run_id)
    df = pd.read_csv(d / "oos_predictions.csv", parse_dates=["date"])
    if symbol is not None and symbol not in set(df["symbol"]):
        raise SymbolNotFoundError(
            f"symbol '{symbol}' has no predictions in run '{run_id}'; "
            f"available: {sorted(set(df['symbol']))}")
    if date is not None:
        ts = pd.Timestamp(date)
        if ts not in set(df["date"]):
            raise SymbolNotFoundError(
                f"date '{date}' has no predictions in run '{run_id}'")
        df = df[df["date"] == ts]
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    return df.sort_values(["date", "symbol", "model"])


def get_orders(run_id: str, *, symbol: str | None = None) -> pd.DataFrame:
    d = _run_dir(run_id)
    df = pd.read_csv(d / "orders.csv")
    if symbol is not None:
        if symbol not in set(df["symbol"]):
            raise SymbolNotFoundError(f"no orders for '{symbol}' in run '{run_id}'")
        df = df[df["symbol"] == symbol]
    return df


def get_health(run_id: str) -> dict:
    d = _run_dir(run_id)
    p = d / "health.json"
    if not p.exists():
        raise FileNotFoundError(f"run '{run_id}' has no health record")
    return read_json(p)
