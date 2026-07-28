"""Regression tests for LazyMem reward weighting and phase transitions."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REWARD_PATH = Path(__file__).resolve().parents[1] / "active_rl" / "reward_score.py"
SPEC = importlib.util.spec_from_file_location("reward_score", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


class RewardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.metrics_path = root / "eval_metrics.jsonl"
        self.latch_path = root / "phase2_latched.json"
        self.env = mock.patch.dict(
            os.environ,
            {
                "LAZYMEM_ACTIVE_RL_ACTION_ETA": "0.915",
                "LAZYMEM_ACTIVE_RL_LAMBDA_QUAL": "0.5",
                "LAZYMEM_ACTIVE_RL_PHASE2_FORMAT_THRESHOLD": "0.99",
                "LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH": str(self.metrics_path),
                "LAZYMEM_ACTIVE_RL_PHASE2_LATCH_PATH": str(self.latch_path),
                "LAZYMEM_ACTIVE_RL_JUDGE_MODE": "disabled",
                "LAZYMEM_ACTIVE_RL_AUTO_APPEND_EVAL_METRICS": "0",
            },
            clear=False,
        )
        self.env.start()
        self.reset_state()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def reset_state(self) -> None:
        reward.PHASE2_LATCH_STATE.clear()
        reward.EVAL_METRIC_STATE.clear()
        reward.EVAL_METRIC_FILE_CACHE.update(
            {"path": None, "mtime_ns": None, "size": None, "rows": []}
        )

    @staticmethod
    def extra_info() -> dict:
        return {
            "query": "Where is the meeting?",
            "gold_answer": "Room 2",
            "required_evidence": "The meeting is in Room 2.",
            "gold_global_message_idxs": [10],
            "split": "train",
            "source_messages": [
                {
                    "global_message_index": 10,
                    "role": "user",
                    "timestamp": "2026-01-01",
                    "content": "The meeting is in Room 2.",
                },
                {
                    "global_message_index": 11,
                    "role": "assistant",
                    "timestamp": "2026-01-01",
                    "content": "Unrelated weather message.",
                },
            ],
        }

    @staticmethod
    def valid_solution(other_op: str = "DROP") -> str:
        other_content = "" if other_op == "DROP" else "Unrelated weather message."
        return json.dumps(
            [
                {
                    "op": "KEEP",
                    "compressed_content": "Meeting: Room 2.",
                    "reason": "Required evidence",
                },
                {
                    "op": other_op,
                    "compressed_content": other_content,
                    "reason": "Not needed",
                },
            ]
        )

    def write_metric(self, value: float) -> None:
        self.metrics_path.write_text(
            json.dumps({"source": "verl_global_validation", "eval_g_format": value}) + "\n",
            encoding="utf-8",
        )
        reward.EVAL_METRIC_FILE_CACHE.update(
            {"path": None, "mtime_ns": None, "size": None, "rows": []}
        )

    def test_eta_is_fixed_to_0915(self) -> None:
        scored = reward.score_with_details(
            solution_str=self.valid_solution(other_op="KEEP"),
            extra_info=self.extra_info(),
        )
        self.assertEqual(scored["g_format"], 1)
        self.assertAlmostEqual(scored["R_action"], 0.915)
        self.assertAlmostEqual(scored["details"]["action"]["action_eta"], 0.915)

    def test_invalid_format_never_calls_judge(self) -> None:
        self.write_metric(1.0)
        with mock.patch.object(
            reward,
            "_judge_reward",
            side_effect=AssertionError("judge must not be called"),
        ):
            scored = reward.score_with_details(
                solution_str="not valid JSON",
                extra_info=self.extra_info(),
            )
        self.assertEqual(scored["g_format"], 0)
        self.assertEqual(scored["details"]["judge"]["skip_reason"], "skip_invalid_format")

    def test_phase2_requires_strictly_greater_than_threshold_and_stays_latched(self) -> None:
        self.write_metric(0.99)
        phase1 = reward.score_with_details(
            solution_str=self.valid_solution(),
            extra_info=self.extra_info(),
        )
        self.assertEqual(phase1["details"]["weights"]["phase2_enabled"], 0.0)
        self.assertFalse(self.latch_path.exists())

        self.write_metric(0.991)
        judge_info = {
            "R_judge": 1.0,
            "R_quality": 1.0,
            "judge_weighted_level": 2.0,
            "faithfulness": 2.0,
            "utility": 2.0,
            "weights": {},
            "keep_judged": 1,
            "judge_items": [],
            "judge_mode_counts": {"test": 1},
            "judge_cache_hit_rate": 0.0,
            "constructed_memory_chars": 10,
        }
        with mock.patch.object(reward, "_judge_reward", return_value=(1.0, judge_info)):
            triggered = reward.score_with_details(
                solution_str=self.valid_solution(),
                extra_info=self.extra_info(),
            )
        self.assertEqual(triggered["details"]["weights"]["phase2_triggered_now"], 1.0)
        self.assertTrue(self.latch_path.is_file())

        self.write_metric(0.0)
        self.reset_state()
        with mock.patch.object(reward, "_judge_reward", return_value=(1.0, judge_info)):
            restarted = reward.score_with_details(
                solution_str=self.valid_solution(),
                extra_info=self.extra_info(),
            )
        self.assertEqual(restarted["details"]["weights"]["phase2_enabled"], 1.0)
        self.assertEqual(restarted["details"]["weights"]["phase2_latched"], 1.0)


if __name__ == "__main__":
    unittest.main()
