"""Regression tests for single-file chronological QA prompt construction."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "build_qa_prompts.py"
SPEC = importlib.util.spec_from_file_location("build_qa_prompts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_qa_prompts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_qa_prompts)


class BuildQaPromptsTest(unittest.TestCase):
    def test_default_output_is_one_qa_prompts_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_root = root / "validation"
            out_dir = root / "output"
            validation_root.mkdir()

            history = {
                "previous_bridge": [],
                "core_messages": [
                    {
                        "role": "user",
                        "timestamp": "2025-01-01",
                        "global_message_index": 1,
                        "content": "Alice moved to Paris.",
                    }
                ],
                "next_bridge": [],
            }
            record = {
                "question_id": "q-001",
                "query_id": "q-001",
                "question_type": "single-session-user",
                "input": (
                    "# Query\n"
                    "Query Time: 2025-02-01\n"
                    "Query: Where does Alice live?\n\n"
                    "# History Window\n"
                    f"{json.dumps(history)}"
                ),
                "output": json.dumps(
                    [
                        {
                            "op": "KEEP",
                            "compressed_content": "Alice moved to Paris.",
                            "reason": "Answers the query.",
                        }
                    ]
                ),
            }
            source = validation_root / "predictions.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            argv = [
                str(SCRIPT_PATH),
                "--validation-root",
                str(validation_root),
                "--out-dir",
                str(out_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                build_qa_prompts.main()

            self.assertEqual(
                sorted(path.name for path in out_dir.iterdir()),
                ["qa_prompts.jsonl"],
            )
            rows = [
                json.loads(line)
                for line in (out_dir / "qa_prompts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question_id"], "q-001")
            self.assertEqual(rows[0]["query_id"], "q-001")
            self.assertEqual(rows[0]["query"], "Where does Alice live?")
            self.assertEqual(
                rows[0]["memory"][0]["compressed_content"],
                "Alice moved to Paris.",
            )


if __name__ == "__main__":
    unittest.main()
