"""Paper decision pipeline (Phase 4/5 scaffolding, NO broker connection).

F-002 fix: decisions are generated ONLY from the frozen model artifact inside
an APPROVED registry bundle (model.pkl, SHA-256 verified against REGISTRY.json).
There is NO decision-time refitting — the artifact is fitted at registration
time by the research runner on all labels matured through the last session.

Additional gates:
- dataset drift: the current manifest's per-symbol source SHA-256 set must
  exactly match the bundle's recorded hashes, else the decision is refused
  (re-registration required);
- staleness: symbols whose data end too long before today are blocked from
  NEW risk (BUY) while de-risking (SELL) remains possible;
- intents are PAPER ONLY: nothing here reaches a broker.
"""
from __future__ import annotations

import pickle

import pandas as pd

from ..config import PROJECT_ROOT, load_market_config, load_risk_config
from ..data.ingest import load_normalized
from ..data.calendar import market_sessions
from ..features.pipeline import build_panel
from ..labels.forward import LabelSpec, add_labels
from ..portfolio.optimizer import PortfolioConfig, target_weights
from ..registry.store import MODELS_DIR, get_status
from ..risk.checks import RiskLimits, check_order_intents
from ..util import now_utc_iso, read_json, sha256_file, write_json


def paper_decide(bundle_id: str, *, initial_equity: float = 1_000_000.0,
                 cash: float | None = None, holdings: dict[str, float] | None = None,
                 pending_orders: list[dict] | None = None) -> dict:
    reg = get_status(bundle_id)
    if reg["status"] != "approved":
        raise PermissionError(
            f"bundle '{bundle_id}' is '{reg['status']}', not approved — "
            "paper decisions refuse unapproved bundles.")
    bundle_dir = MODELS_DIR / bundle_id
    cfg = read_json(bundle_dir / "config.json")
    scfg = cfg["strategy_config"]
    mcfg = load_market_config(cfg["market"])
    rcfg = load_risk_config()

    # ---- dataset drift gate (F-002): recorded hashes must match reality ----
    manifest = read_json(PROJECT_ROOT / "datasets" / f"manifest_{cfg['market']}.json")
    bundle_hashes = cfg.get("dataset_source_sha256") or {}
    drifted = {}
    for sym, entry in manifest["symbols"].items():
        live = entry.get("source_sha256")
        recorded = bundle_hashes.get(sym)
        if live and recorded and live != recorded:
            drifted[sym] = {"recorded": recorded, "live": live}
    if drifted:
        raise PermissionError(
            f"dataset drift for {sorted(drifted)}: bundle '{bundle_id}' was "
            "registered on different data — re-run research and re-approve "
            "before deciding on the new snapshot.")

    # ---- frozen model artifact (NO decision-time refitting) -----------------
    model_path = bundle_dir / "model.pkl"
    if not model_path.exists():
        raise PermissionError(
            f"bundle '{bundle_id}' has no frozen model artifact — decisions "
            "refuse bundles registered before the model-artifact contract "
            "(re-run research to register one).")
    model_sha = sha256_file(model_path)
    expected_sha = reg.get("model_sha256") or cfg.get("model_sha256")
    if not expected_sha:
        raise PermissionError("registry entry lacks model_sha256 — cannot verify artifact")
    if model_sha != expected_sha:
        raise PermissionError(
            f"model artifact hash mismatch for bundle '{bundle_id}' "
            f"({model_sha} != {expected_sha}) — refusing to decide on a "
            "tampered/unverified artifact.")
    artifact = pickle.loads(model_path.read_bytes())
    model, payoff = artifact["model"], artifact["payoff"]
    feat_cols = list(artifact["feature_names"])

    # ---- rebuild the inference panel from the current normalized data ------
    frames = load_normalized(cfg["market"], mcfg["universe"])
    panel = build_panel(frames, feat_cols)
    for sym, df in panel.items():
        df["symbol"] = sym
    panel = {sym: add_labels(df, LabelSpec(**scfg["label"])) for sym, df in panel.items()}
    sessions = market_sessions(frames)
    last_session = sessions[-1]

    pred_parts = []
    for sym, df in panel.items():
        last_row = df[df["date"] == last_session].dropna(subset=feat_cols)
        pred_parts.append(last_row)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    pred_df["p_up"] = model.predict_proba(pred_df[feat_cols].to_numpy(float))[:, 1]
    pred_df["mu_net"] = [
        p * payoff.e_event.get(s, payoff.global_event)
        + (1 - p) * payoff.e_complement.get(s, payoff.global_complement)
        for s, p in zip(pred_df["symbol"], pred_df["p_up"])
    ]
    pred_df = pred_df[pred_df["p_up"].notna() & (pred_df["vol_20"] > 0)]

    pcfg = PortfolioConfig(edge_threshold=scfg["portfolio"]["edge_threshold"],
                           lambda_vol=scfg["portfolio"]["lambda_vol"],
                           max_names=scfg["portfolio"]["max_names"],
                           max_weight_per_name=rcfg["max_weight_per_name"],
                           max_gross_exposure=rcfg["max_gross_exposure"])
    alloc = target_weights(pred_df.set_index("symbol")["mu_net"],
                           pred_df.set_index("symbol")["vol_20"], pcfg)

    limits = RiskLimits(**{k: rcfg[k] for k in
                           ("max_weight_per_name", "max_gross_exposure", "cash_reserve",
                            "max_turnover", "min_order_value", "max_open_orders_per_symbol",
                            "allowed_sides", "stop_loss_pct", "trail_stop_pct")})
    cash = float(cash if cash is not None else initial_equity)
    holdings = {s: float(q) for s, q in (holdings or {}).items() if float(q) > 0}
    pending_orders = list(pending_orders or [])

    # staleness: block NEW risk on symbols whose data ended too long ago
    as_of_ts = pd.Timestamp(now_utc_iso()[:10])
    stale: set[str] = set()
    for sym, entry in manifest["symbols"].items():
        if entry.get("usable") and entry.get("date_end"):
            if (as_of_ts - pd.Timestamp(entry["date_end"])).days > int(mcfg["stale_session_days"]):
                stale.add(sym)

    # frozen funds: pending BUY notional reduces the cash available to new buys
    pending_buy_value = sum(
        float(o.get("qty", 0.0)) * float(o.get("ref_price", 0.0))
        for o in pending_orders if o.get("side") == "BUY")

    last_prices: dict[str, float] = {}
    for sym in sorted(set(holdings) | set(mcfg["universe"])):
        rows = panel[sym].loc[panel[sym]["date"] == last_session, "close_adj"]
        if len(rows):
            last_prices[sym] = float(rows.iloc[0])
    equity = cash + sum(q * last_prices[s] for s, q in holdings.items()
                        if s in last_prices and last_prices[s] > 0)
    if equity <= 0:
        raise ValueError("nonpositive equity: check cash and holdings inputs")

    # two-sided rebalance intents from current weights; conservative: buys are
    # checked against cash NET of frozen pending-buy notional
    current_w = {s: q * last_prices[s] / equity
                 for s, q in holdings.items() if s in last_prices}
    intents = []
    for sym in sorted(set(alloc.weights.index) | set(holdings)):
        w_target = float(alloc.weights.get(sym, 0.0))
        w_cur = current_w.get(sym, 0.0)
        px = last_prices.get(sym, 0.0)
        delta_value = (w_target - w_cur) * equity
        if abs(delta_value) < limits.min_order_value or px <= 0:
            continue
        side = "BUY" if delta_value > 0 else "SELL"
        intents.append({"symbol": sym, "side": side,
                        "qty": abs(delta_value) / px, "ref_price": px})
    open_count: dict[str, int] = {}
    for o in pending_orders:
        open_count[o.get("symbol", "?")] = open_count.get(o.get("symbol", "?"), 0) + 1
    approved, rejected = check_order_intents(
        intents, limits=limits, equity=equity, cash=cash,
        pending_buy_value=pending_buy_value,
        sellable_qty={s: q for s, q in holdings.items()},
        stale_symbols=stale, eligible_symbols=set(mcfg["universe"]),
        open_orders_count=open_count)

    record = {
        "as_of_session": str(last_session.date()),
        "decision_for_session": "next market session after as_of (entry at its open)",
        "bundle_id": bundle_id,
        "bundle_status": reg["status"],
        "mode": "PAPER INTENTS ONLY — no broker connection, no orders submitted",
        "model_source": {
            "bundle_id": bundle_id,
            "kind": artifact.get("kind"),
            "sha256": model_sha,
            "trained_through": artifact.get("trained_through"),
            "refit_at_decision": False,
        },
        "inputs": {
            "cash": cash,
            "holdings": holdings,
            "pending_buy_value_frozen": pending_buy_value,
            "equity_estimate": equity,
            "stale_symbols_blocked": sorted(stale),
        },
        "target_weights": {k: float(v) for k, v in alloc.weights.items() if v > 0},
        "cash_weight": float(1.0 - alloc.weights.sum()),
        "diagnostics": alloc.diagnostics,
        "approved_intents": [{**i, "qty": round(i["qty"], 2)} for i in approved],
        "rejected_intents": rejected,
        "provenance": {
            "created_utc": now_utc_iso(),
            "config_fingerprint": cfg["config_fingerprint"],
            "dataset_sha256": cfg["dataset_source_sha256"],
            "dataset_drift": False,
        },
    }
    out_dir = PROJECT_ROOT / "artifacts" / "decisions"
    write_json(out_dir / f"decision_{bundle_id}_{now_utc_iso().replace(':', '')}.json", record)
    return record
