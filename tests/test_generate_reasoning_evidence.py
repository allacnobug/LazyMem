"""Regression tests for question-level GRPO reasoning-evidence generation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline"
    / "generate_reasoning_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("generate_reasoning_evidence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_reasoning = importlib.util.module_from_spec(SPEC)
sys.modules["generate_reasoning_evidence"] = generate_reasoning
SPEC.loader.exec_module(generate_reasoning)


def dataset_entry(question_id: str = "q-1") -> dict:
    return {
        "question_id": question_id,
        "question": "Where did Alice move?",
        "question_date": "2025-02-01",
        "answer": "Paris",
        "answer_session_ids": ["session-1"],
        "haystack_session_ids": ["session-1", "session-2"],
        "haystack_dates": ["2025-01-01", "2025-01-02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I may move soon.", "has_answer": False},
                {"role": "user", "content": "I moved to Paris.", "has_answer": True},
            ],
            [{"role": "user", "content": "Unrelated.", "has_answer": True}],
        ],
    }


class GenerateReasoningEvidenceTest(unittest.TestCase):
    def test_regular_question_prefers_marked_answer_turns(self) -> None:
        turns = generate_reasoning.evidence_turns(dataset_entry())

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["content"], "I moved to Paris.")

    def test_abstention_question_uses_the_complete_answer_session(self) -> None:
        turns = generate_reasoning.evidence_turns(dataset_entry("q-1_abs"))

        self.assertEqual(
            [turn["content"] for turn in turns],
            ["I may move soon.", "I moved to Paris."],
        )

    def test_think_block_is_separated_from_visible_reasoning(self) -> None:
        thinking, visible = generate_reasoning.split_thinking(
            "<think>internal check</think>\nAlice explicitly said she moved to Paris."
        )

        self.assertEqual(thinking, "internal check")
        self.assertEqual(visible, "Alice explicitly said she moved to Paris.")

    def test_vllm_request_enables_qwen_thinking_through_chat_template(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "hidden check",
                        "content": "The evidence supports Paris.",
                    }
                }
            ]
        }
        with mock.patch.object(
            generate_reasoning.requests,
            "post",
            return_value=response,
        ) as post:
            thinking, visible = generate_reasoning.call_model(
                endpoint="http://localhost/v1/chat/completions",
                model="qwen-32b",
                api_key="",
                entry=dataset_entry(),
                turns=generate_reasoning.evidence_turns(dataset_entry()),
                timeout=1,
                retries=1,
                max_tokens=128,
                enable_thinking=True,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True},
        )
        self.assertEqual(thinking, "hidden check")
        self.assertEqual(visible, "The evidence supports Paris.")

    def test_split_writer_produces_the_file_consumed_by_grpo_prep(self) -> None:
        entry = dataset_entry()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with mock.patch.object(
                generate_reasoning,
                "call_model",
                return_value=("hidden", "The evidence states that Alice moved to Paris."),
            ):
                report = generate_reasoning.process_split(
                    split="train",
                    question_ids=["q-1"],
                    dataset_by_id={"q-1": entry},
                    output_dir=output_dir,
                    endpoint="http://localhost/v1/chat/completions",
                    model="qwen-32b",
                    api_key="",
                    workers=1,
                    timeout=1,
                    retries=1,
                    max_tokens=128,
                    enable_thinking=True,
                    overwrite=False,
                )

            output = output_dir / "train" / "reasoning_qwen32b.split.jsonl"
            rows = list(generate_reasoning.iter_jsonl(output))
            self.assertEqual(report["output_rows"], 1)
            self.assertEqual(rows[0]["question_id"], "q-1")
            self.assertEqual(
                rows[0]["reasoning_answer"],
                "The evidence states that Alice moved to Paris.",
            )
            self.assertEqual(rows[0]["gold_answer"], "Paris")


if __name__ == "__main__":
    unittest.main()
