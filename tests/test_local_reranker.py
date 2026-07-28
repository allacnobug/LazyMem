"""Regression tests for the credential-free local experiment-15 reranker."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "retrieval" / "local_reranker.py"
SPEC = importlib.util.spec_from_file_location("local_reranker", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
local_reranker = importlib.util.module_from_spec(SPEC)
sys.modules["local_reranker"] = local_reranker
SPEC.loader.exec_module(local_reranker)


class LocalRerankerTest(unittest.TestCase):
    def test_local_scores_reorder_the_candidate_head_and_preserve_the_tail(self) -> None:
        model = mock.Mock()
        model.predict.return_value = [0.1, 0.9, 0.5]
        documents = {
            "a": {"raw_text": "first"},
            "b": {"raw_text": "second"},
            "c": {"raw_text": "third"},
            "d": {"raw_text": "tail"},
        }

        with mock.patch.object(
            local_reranker,
            "get_cross_encoder",
            return_value=model,
        ):
            ranking = local_reranker.local_cross_encoder_rerank(
                query="query",
                ranked_ids=["a", "b", "c", "d"],
                documents_by_id=documents,
                model_name="/models/bge-reranker-v2-m3",
                device="cuda",
                batch_size=8,
                candidate_count=3,
            )

        self.assertEqual(ranking, ["b", "c", "a", "d"])
        model.predict.assert_called_once_with(
            [["query", "first"], ["query", "second"], ["query", "third"]],
            batch_size=8,
            show_progress_bar=False,
        )


if __name__ == "__main__":
    unittest.main()
