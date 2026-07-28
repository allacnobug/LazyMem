"""Regression tests for bounded LongMemEval and LoCoMo history windows."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline"
    / "build_longmemeval_windows.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_longmemeval_windows",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
window_builder = importlib.util.module_from_spec(SPEC)
sys.modules["build_longmemeval_windows"] = window_builder
SPEC.loader.exec_module(window_builder)

LOCOMO_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline"
    / "build_locomo_windows.py"
)
LOCOMO_SPEC = importlib.util.spec_from_file_location(
    "build_locomo_windows",
    LOCOMO_SCRIPT_PATH,
)
assert LOCOMO_SPEC is not None and LOCOMO_SPEC.loader is not None
locomo_builder = importlib.util.module_from_spec(LOCOMO_SPEC)
sys.modules["build_locomo_windows"] = locomo_builder
LOCOMO_SPEC.loader.exec_module(locomo_builder)


class WindowBuildingTest(unittest.TestCase):
    def test_context_expansion_is_capped_at_eight_with_stride_seven(self) -> None:
        entry = {
            "question_id": "q-001",
            "question": "test",
            "question_date": "2025-01-01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024-01-01"],
            "haystack_sessions": [
                [
                    {
                        "role": "user" if index % 2 else "assistant",
                        "content": f"message {index}",
                        "has_answer": False,
                    }
                    for index in range(1, 21)
                ]
            ],
            "answer_session_ids": [],
        }
        args = argparse.Namespace(
            one_based_global_idx=False,
            context_window=2,
            max_window_messages=8,
            stride=7,
        )

        rows, _ = window_builder.build_windows_for_entry(
            entry,
            ["s1_03", "s1_08", "s1_13"],
            args,
        )
        indexes = []
        for public, _private in rows:
            window = public["window"]
            messages = (
                window["previous_bridge"]
                + window["core_messages"]
                + window["next_bridge"]
            )
            self.assertLessEqual(len(messages), 8)
            indexes.append(
                [message["global_message_index"] for message in messages]
            )

        self.assertEqual(len(indexes), 2)
        self.assertEqual(indexes[0][-1], indexes[1][0])

    def test_locomo_uses_the_same_total_window_cap(self) -> None:
        entry = {
            "question_id": "sample-1::qa_000",
            "sample_id": "sample-1",
            "qa_idx": 0,
            "question": "test",
            "answer": "answer",
            "category": 1,
            "gold_message_ids": [],
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {
                        "dia_id": f"D1:{index}",
                        "speaker": "Alice" if index % 2 else "Bob",
                        "text": f"message {index}",
                    }
                    for index in range(1, 21)
                ],
            },
        }
        args = argparse.Namespace(
            one_based_global_idx=False,
            context_window=2,
            max_window_messages=8,
            stride=7,
        )

        rows, _ = locomo_builder.build_windows_for_entry(
            entry,
            ["D1:3", "D1:8", "D1:13"],
            args,
        )
        for public, _private in rows:
            window = public["window"]
            messages = (
                window["previous_bridge"]
                + window["core_messages"]
                + window["next_bridge"]
            )
            self.assertLessEqual(len(messages), 8)


if __name__ == "__main__":
    unittest.main()
