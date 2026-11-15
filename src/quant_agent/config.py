"""Configuration loading with explicit validation.

Every run records a fingerprint of the exact config used so results link to
data/code/configuration versions (handover 8.1 principle 5).
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

REQUIRED_MARKET_KEYS = {
    "market", "venue", "currency", "calendar", "source_dir", "lots",
    "allow_fractional", "t_plus_1", "universe", "costs", "participation_cap",
    "price_limit_pct", "stale_session_days",
}
REQUIRED_COST_KEYS = {"commission_bps", "min_commission", "slippage_bps", "stamp_tax_sell_bps"}
REQUIRED_STRATEGY_KEYS = {
    "strategy", "market", "label", "features", "walk_forward", "model",
    "portfolio", "benchmark", "seed",
}
REQUIRED_LABEL_KEYS = {"entry_offset", "entry_field", "exit_offset", "exit_field"}
REQUIRED_WF_KEYS = {"initial_train_fraction", "test_sessions", "step_sessions", "policy_tune_fraction"}
REQUIRED_PORTFOLIO_KEYS = {"edge_threshold", "lambda_vol", "max_names", "shrink_conditional_returns"}
REQUIRED_RISK_KEYS = {
    "max_weight_per_name", "max_gross_exposure", "cash_reserve", "max_turnover",
    "min_order_value", "max_open_orders_per_symbol", "allowed_sides",
    "stop_loss_pct", "trail_stop_pct",
}


class ConfigError(ValueError):
    pass


def _load(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    return cfg


def _require(cfg: dict, keys: set[str], what: str) -> None:
    missing = keys - set(cfg)
    if missing:
        raise ConfigError(f"{what} missing required keys: {sorted(missing)}")


def load_market_config(market: str) -> dict:
    cfg = _load(CONFIG_DIR / "markets" / f"{market}.yaml")
    _require(cfg, REQUIRED_MARKET_KEYS, f"market config '{market}'")
    _require(cfg["costs"], REQUIRED_COST_KEYS, f"market config '{market}'.costs")
    if not cfg["universe"]:
        raise ConfigError(f"market '{market}' has empty universe")
    return cfg


def load_strategy_config(name: str) -> dict:
    cfg = _load(CONFIG_DIR / "strategies" / f"{name}.yaml")
    _require(cfg, REQUIRED_STRATEGY_KEYS, f"strategy config '{name}'")
    _require(cfg["label"], REQUIRED_LABEL_KEYS, f"strategy '{name}'.label")
    _require(cfg["walk_forward"], REQUIRED_WF_KEYS, f"strategy '{name}'.walk_forward")
    _require(cfg["portfolio"], REQUIRED_PORTFOLIO_KEYS, f"strategy '{name}'.portfolio")
    if not cfg["features"]:
        raise ConfigError(f"strategy '{name}' has no features")
    return cfg


def load_risk_config(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG_DIR / "risk" / "default.yaml"
    cfg = _load(p)
    _require(cfg, REQUIRED_RISK_KEYS, f"risk config '{p.name}'")
    return cfg
