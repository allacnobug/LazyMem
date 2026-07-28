"""Regression tests for metadata and prediction flow across evaluation stages."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EVALUATION_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_inference = load_module(
    "run_parquet_inference_io",
    "run_parquet_inference_io.py",
)
run_qa = load_module("run_qwen32b_qa", "run_qwen32b_qa.py")
llm_judge = load_module("module2_llm_judge", "module2_llm_judge.py")


class EvaluationPipelineTest(unittest.TestCase):
    def test_llm_judge_defaults_match_the_paper(self) -> None:
        with mock.patch.object(sys, "argv", ["module2_llm_judge.py"]):
            args = llm_judge.parse_args()
        self.assertEqual(args.judge_base_url, "https://api.deepseek.com/v1")
        self.assertEqual(args.judge_model, "deepseek-v4-pro")
        self.assertEqual(args.temperature, 0.0)

    def test_parquet_metadata_is_made_public(self) -> None:
        extra_info = {
            "query": "Where does Alice live?",
            "gold_answer": "Paris",
            "metadata": {
                "question_id": "q-001",
                "query_timestamp": "2025-02-01",
                "question_type": "single-session-user",
            },
        }
        self.assertEqual(
            run_inference.public_metadata(extra_info),
            {
                "question_id": "q-001",
                "query_id": "q-001",
                "query": "Where does Alice live?",
                "query_time": "2025-02-01",
                "question_type": "single-session-user",
            },
        )

    def test_qa_output_preserves_id_and_exposes_prediction(self) -> None:
        args = argparse.Namespace(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            model="qwen-32b",
            temperature=0.0,
            max_tokens=512,
            timeout=120,
            retries=0,
            retry_sleep=0.0,
            keep_raw_response=False,
        )
        row = {
            "question_id": "q-001",
            "query_id": "q-001",
            "query": "Where does Alice live?",
            "qa_prompt": "prompt",
        }
        response = {
            "model": "qwen-32b",
            "choices": [
                {
                    "message": {"content": "Paris."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        with mock.patch.object(
            run_qa,
            "post_chat_completion",
            return_value=response,
        ):
            result = run_qa.answer_one(0, row, args, {})

        self.assertEqual(result["question_id"], "q-001")
        self.assertEqual(result["prediction"], "Paris.")
        self.assertEqual(result["answer"], "Paris.")

    def test_judge_reads_qa_answer_field(self) -> None:
        self.assertEqual(llm_judge.prediction_text({"answer": "Paris."}), "Paris.")


if __name__ == "__main__":
    unittest.main()
