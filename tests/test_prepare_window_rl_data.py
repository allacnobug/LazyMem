"""Regression tests for paper-aligned balanced RL data sampling."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline"
    / "prepare_window_rl_data.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_window_rl_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_rl = importlib.util.module_from_spec(SPEC)
sys.modules["prepare_window_rl_data"] = prepare_rl
SPEC.loader.exec_module(prepare_rl)


def row(index: int, gold: bool) -> dict:
    return {
        "extra_info": {
            "index": index,
            "gold_global_message_idxs": [index] if gold else [],
        }
    }


class PrepareWindowRlDataTest(unittest.TestCase):
    def test_paper_balance_is_half_gold_and_half_non_gold(self) -> None:
        rows = [row(0, True), row(1, True)]
        rows.extend(row(index, False) for index in range(2, 8))

        balanced = prepare_rl.balance_train_rows(
            rows,
            total_size=4,
            seed=42,
        )

        self.assertEqual(len(balanced), 4)
        self.assertEqual(
            sum(
                bool(item["extra_info"]["gold_global_message_idxs"])
                for item in balanced
            ),
            2,
        )
        self.assertEqual(
            [item["extra_info"]["index"] for item in balanced],
            [0, 1, 2, 3],
        )
        self.assertTrue(
            all("source_index" in item["extra_info"] for item in balanced)
        )


if __name__ == "__main__":
    unittest.main()
