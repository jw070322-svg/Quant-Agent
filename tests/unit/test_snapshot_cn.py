"""CN snapshot assembler pure builders (NO network): PIT conventions, unit
conversions, action ordering/validation, manifest round-trip via read_snapshot."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quant_agent.data.factor_snapshot import NEEDS, read_snapshot
from quant_agent.util import write_json
from quant_agent.research.snapshot_cn import (build_actions_table,
                                              build_bars_table,
                                              build_costs_table,
                                              build_financials_table,
                                              build_float_table,
                                              build_universe_table)


def _raw_frame(dates, base=10.0):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": base * (1 + rng.normal(0, 0.002, len(dates))),
        "high": base * 1.01, "low": base * 0.99,
        "close": base * (1 + rng.normal(0, 0.002, len(dates))),
        "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        "amount": rng.uniform(1e7, 5e7, len(dates)),
        "outstanding_share": np.full(len(dates), 1_000_000_000.0),
    })


def test_build_bars_drops_ohlc_inconsistent_rows_and_discloses():
    dates = pd.bdate_range("2025-01-01", periods=10)
    raw = _raw_frame(dates).set_index("date")
    raw.loc[dates[3], "high"] = raw.loc[dates[3], "low"] - 1  # inconsistent
    raw = raw.reset_index()
    qfq = pd.DataFrame({"date": dates, "open": raw["open"], "close": raw["close"]})
    bars, dropped = build_bars_table({"A": raw}, {"A": qfq}, buy_lot=100, sell_lot=1)
    assert len(bars) == 9
    assert dropped["dropped_total"] == 1
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()


def test_build_bars_tri_from_qfq_series():
    dates = pd.bdate_range("2025-01-01", periods=10)
    raw = _raw_frame(dates)
    qfq = pd.DataFrame({"date": dates, "open": raw["open"] * 2.0,
                        "close": raw["close"] * 2.0})
    bars, _ = build_bars_table({"A": raw}, {"A": qfq}, buy_lot=100, sell_lot=1)
    assert np.allclose(bars["tri"], bars["close"] * 2.0)


def test_build_universe_fallback_to_first_bar_and_industry_fallback():
    listing = {"A": {"listing_date": pd.Timestamp("2024-01-05"), "industry": "白酒"},
               "B": {"listing_date": None, "industry": ""}}
    first_bar = {"A": pd.Timestamp("2024-01-02"), "B": pd.Timestamp("2024-01-10")}
    uni = build_universe_table(listing, first_bar)
    row_a = uni[uni["symbol"] == "A"].iloc[0]
    row_b = uni[uni["symbol"] == "B"].iloc[0]
    assert row_a["listed_at"] == pd.Timestamp("2024-01-05")
    assert row_b["listed_at"] == pd.Timestamp("2024-01-10")  # first-bar fallback
    assert row_b["industry"] == "UNKNOWN"
    assert (uni["eligible"] == 1).all()
    assert uni["effective_to"].isna().all()


def test_build_actions_per_share_conversion_and_dividend_before_split():
    batch = pd.DataFrame([{
        "symbol": "000001", "split_ratio_10": 5.0, "dividend_ratio_10": 20.0,
        "ex_date": "2025-06-20", "plan_notice": "2025-04-01",
    }])
    actions = build_actions_table([batch], {"000001.SZ"})
    assert set(actions["kind"]) == {"split", "dividend"}
    assert actions[actions["kind"] == "split"]["value"].iloc[0] == pytest.approx(0.5)
    assert actions[actions["kind"] == "dividend"]["value"].iloc[0] == pytest.approx(2.0)
    div_seq = actions[actions["kind"] == "dividend"]["sequence"].iloc[0]
    split_seq = actions[actions["kind"] == "split"]["sequence"].iloc[0]
    assert div_seq < split_seq
    assert actions["event_id"].is_unique


def test_build_actions_filters_to_universe_and_ignores_pending():
    batch = pd.DataFrame([
        {"symbol": "000001", "split_ratio_10": 5.0, "dividend_ratio_10": 0.0,
         "ex_date": "2025-06-20", "plan_notice": "2025-04-01"},
        {"symbol": "999999", "split_ratio_10": 5.0, "dividend_ratio_10": 0.0,
         "ex_date": "2025-06-20", "plan_notice": "2025-04-01"},
        {"symbol": "000001", "split_ratio_10": 5.0, "dividend_ratio_10": 0.0,
         "ex_date": None, "plan_notice": "2025-04-01"},
    ])
    actions = build_actions_table([batch], {"000001.SZ"})
    assert len(actions) == 1
    assert actions.iloc[0]["symbol"] == "000001.SZ"


def test_build_float_table_daily_availability_convention():
    dates = pd.bdate_range("2025-01-01", periods=5)
    raw = _raw_frame(dates)
    fl = build_float_table({"A": raw})
    assert (fl["float_shares"] == 1_000_000_000.0).all()
    # market_cap = close × outstanding shares
    assert np.allclose(fl["market_cap"], fl["float_shares"] * raw["close"].to_numpy())
    assert (fl["available_at"] == fl["date"]).all()


def test_build_costs_table_from_market_config():
    mcfg = {"costs": {"commission_bps": 2.5, "min_commission": 5.0,
                      "slippage_bps": 5.0, "stamp_tax_sell_bps": 5.0}}
    costs = build_costs_table(mcfg, pd.Timestamp("2025-01-02"))
    assert costs.iloc[0]["commission_bps"] == pytest.approx(2.5)
    assert costs.iloc[0]["stamp_tax_sell_bps"] == pytest.approx(5.0)


def _write_snapshot(tmp_path, tables):
    files = {}
    for name, df in tables.items():
        f = tmp_path / f"{name}.csv"
        df.to_csv(f, index=False)
        # contract: manifest["files"] is keyed by table name
        files[name] = {"path": f"{name}.csv",
                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return files


def _mini_tables(dates=None):
    dates = (dates if dates is not None
             else pd.bdate_range("2025-01-01", periods=300))
    raw = _raw_frame(dates)
    raw["date"] = pd.DatetimeIndex(dates)
    qfq = pd.DataFrame({"date": dates, "open": raw["open"], "close": raw["close"]})
    bars, _ = build_bars_table({"000001.SZ": raw}, {"000001.SZ": qfq},
                               buy_lot=100, sell_lot=1)
    uni = build_universe_table(
        {"000001.SZ": {"listing_date": pd.Timestamp("2024-12-01"), "industry": "银行"}},
        {"000001.SZ": dates[0]})
    costs = build_costs_table({"costs": {"commission_bps": 2.5, "min_commission": 5.0,
                                         "slippage_bps": 5.0, "stamp_tax_sell_bps": 5.0}},
                              dates[0])
    sessions = pd.DataFrame({"date": dates})
    fl = build_float_table({"000001.SZ": raw})
    # MOM requires an actions table; an empty implemented-events frame is the
    # valid no-actions state (the loader tolerates empty actions specifically)
    actions = build_actions_table([], set(uni["symbol"]))
    return {"sessions": sessions, "bars": bars, "universe": uni, "costs": costs,
            "float": fl, "actions": actions}


def test_snapshot_roundtrip_through_read_snapshot(tmp_path):
    """Assembled tables must pass the bundle's read_snapshot validation; MOM's
    dependencies (sessions/bars/universe/actions/costs) are all satisfied."""
    tables = _mini_tables()
    files = _write_snapshot(tmp_path, tables)
    write_json(tmp_path / "manifest.json", {
        "market": "cn", "schema_version": 1, "price_basis": "raw",
        "quantity_unit": "shares", "amount_unit": "local_currency",
        "files": files})
    tables_loaded, manifest, errors = read_snapshot(tmp_path, "cn")
    assert errors == [], f"unexpected snapshot errors: {errors}"
    for table in NEEDS["MOM"]:
        assert table in tables_loaded
    assert manifest["price_basis"] == "raw"


def test_snapshot_with_financials_reads_fscore_fields(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=300)
    tables = _mini_tables(dates)
    fin = pd.DataFrame([
        {"symbol": "000001.SZ", "period": pd.Timestamp("2022-12-31"),
         "available_at": pd.Timestamp("2023-04-20"), "revision": "20230420",
         "book_equity": 2e11, "net_income": 3e10, "assets": 3.3e12,
         "operating_cashflow": 4e10, "long_debt": 2e11,
         "current_assets": 5e11, "current_liabilities": 4e11,
         "revenue": 3e11, "cogs": 2e11, "equity_issued": 0.0},
        {"symbol": "000001.SZ", "period": pd.Timestamp("2023-12-31"),
         "available_at": pd.Timestamp("2024-04-20"), "revision": "20240420",
         "book_equity": 2.2e11, "net_income": 3.3e10, "assets": 3.4e12,
         "operating_cashflow": 4.2e10, "long_debt": 1.9e11,
         "current_assets": 5.2e11, "current_liabilities": 4.1e11,
         "revenue": 3.2e11, "cogs": 2.1e11, "equity_issued": 0.0},
    ])
    tables["financials"] = fin
    files = _write_snapshot(tmp_path, tables)
    write_json(tmp_path / "manifest.json", {
        "market": "cn", "schema_version": 1, "price_basis": "raw",
        "quantity_unit": "shares", "amount_unit": "local_currency",
        "files": files})
    tables_loaded, _, errors = read_snapshot(tmp_path, "cn")
    assert errors == []
    assert "financials" in tables_loaded
    assert set(tables_loaded["financials"]["book_equity"]) == {2e11, 2.2e11}
