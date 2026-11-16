"""Shard/merge machinery for distributed fold fitting (NO network, NO sims)."""
import numpy as np
import pandas as pd
import pytest

from quant_agent.research.runner import _merge_shard_outputs
from quant_agent.util import write_json

PRED_COLS = ["date", "symbol", "model", "fold_id", "p_up", "mu_net", "vol",
             "window"]


def _pred_row(fold_id, date="2025-03-05", symbol="A", model="gbm"):
    return {"date": date, "symbol": symbol, "model": model, "fold_id": fold_id,
            "p_up": 0.6, "mu_net": 0.01, "vol": 0.02, "window": "report"}


def _write_shards(tmp_path, shards: dict[int, list[int]]):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for shard, folds in shards.items():
        rows = [_pred_row(f) for f in folds]
        pd.DataFrame(rows, columns=PRED_COLS).to_csv(
            run_dir / f"oos_predictions_shard{shard}of2.csv", index=False)
        write_json(run_dir / f"chosen_thresholds_shard{shard}of2.csv".replace(".csv", ".json"),
                   {str(f): 0.55 for f in folds})
        write_json(run_dir / f"fold_records_shard{shard}of2.json",
                   [{"fold_id": f} for f in folds])
    return run_dir


def test_shard_partition_is_exact_and_disjoint():
    """Round-robin fold assignment must partition every fold exactly once —
    this is what makes distributed fitting safe."""
    n_folds, n_shards = 11, 3
    assigned = [{f for f in range(n_folds) if f % n_shards == i}
                for i in range(n_shards)]
    union = set().union(*assigned)
    assert union == set(range(n_folds))
    assert sum(len(a) for a in assigned) == n_folds  # no overlap


def test_merge_concatenates_and_sorts(tmp_path):
    run_dir = _write_shards(tmp_path, {0: [0, 2], 1: [1, 3]})
    preds, thr, folds = _merge_shard_outputs(run_dir, {0, 1, 2, 3})
    assert len(preds) == 4
    assert list(preds["fold_id"]) == [0, 1, 2, 3]  # sorted by fold_id
    assert thr == {0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55}
    assert [r["fold_id"] for r in folds] == [0, 1, 2, 3]


def test_merge_rejects_duplicate_fold_across_shards(tmp_path):
    run_dir = _write_shards(tmp_path, {0: [0, 1], 1: [1]})  # fold 1 twice
    with pytest.raises(RuntimeError, match="multiple shards"):
        _merge_shard_outputs(run_dir, {0, 1})


def test_merge_rejects_incomplete_shards(tmp_path):
    run_dir = _write_shards(tmp_path, {0: [0]})  # fold 1 never predicted
    with pytest.raises(RuntimeError, match="shards incomplete"):
        _merge_shard_outputs(run_dir, {0, 1})


def test_merge_rejects_no_shards(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no shard prediction files"):
        _merge_shard_outputs(run_dir, {0})
