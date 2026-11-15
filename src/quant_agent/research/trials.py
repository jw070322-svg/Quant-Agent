"""Trial ledger: every fit, every policy search, every decision is recorded.

Addresses handover 4.5 / 8.3 `research`: search history must survive so the
selected result can be interpreted with its multiple-comparisons burden.
"""
from __future__ import annotations

from ..util import append_jsonl, now_utc_iso


class TrialLedger:
    def __init__(self, run_dir, run_id: str):
        self.path = f"{run_dir}/trials.jsonl"
        self.run_id = run_id

    def log(self, stage: str, **fields) -> None:
        append_jsonl(self.path, {
            "run_id": self.run_id,
            "ts_utc": now_utc_iso(),
            "stage": stage,
            **fields,
        })
