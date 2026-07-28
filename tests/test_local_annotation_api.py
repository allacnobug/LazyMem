"""Regression tests for credential-free local teacher annotation requests."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "data_pipeline" / "annotate_windows.py"
SPEC = importlib.util.spec_from_file_location("annotate_windows", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
annotate_windows = importlib.util.module_from_spec(SPEC)
sys.modules["annotate_windows"] = annotate_windows
SPEC.loader.exec_module(annotate_windows)


class LocalAnnotationApiTest(unittest.TestCase):
    def test_local_request_needs_no_authorization_header(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "[]"}}]}
        ).encode("utf-8")

        with mock.patch.object(
            annotate_windows.urllib.request,
            "urlopen",
            return_value=response,
        ) as urlopen:
            annotate_windows.post_chat_completion(
                base_url="http://127.0.0.1:8022/v1",
                api_key="",
                model="qwen-32b",
                messages=[{"role": "user", "content": "annotate"}],
                temperature=0.0,
                max_tokens=128,
                timeout=1,
                thinking=True,
                extra_body=None,
            )

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(
            request_body["chat_template_kwargs"],
            {"enable_thinking": True},
        )


if __name__ == "__main__":
    unittest.main()
