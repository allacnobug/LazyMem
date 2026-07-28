#!/usr/bin/env python3
"""Send QA prompts to a Qwen32B OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "evaluation" / "qa_prompts" / "qa_prompts.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "evaluation" / "qwen32b_qa_answers.jsonl"
PASSTHROUGH_FIELDS = (
    "question_id",
    "query_id",
    "sample_id",
    "qa_idx",
    "question",
    "question_type",
    "category",
    "query_time",
    "query",
)


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def endpoint_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer the user question concisely and faithfully from the provided memory.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url(base_url),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def answer_one(
    index: int,
    row: Dict[str, Any],
    args: argparse.Namespace,
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = row.get("qa_prompt")
    if not prompt:
        raise ValueError(f"row {index} missing qa_prompt")
    metadata = {key: row.get(key) for key in PASSTHROUGH_FIELDS if key in row}

    last_error = None
    for attempt in range(1, args.retries + 2):
        try:
            response = post_chat_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body=extra_body,
            )
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            answer = message.get("content", "")
            return {
                **metadata,
                "index": index,
                "prediction": answer,
                "answer": answer,
                "finish_reason": choice.get("finish_reason", ""),
                "model": response.get("model", args.model),
                "usage": response.get("usage", {}),
                "source_records": row.get("source_records", []),
                "raw_response": response if args.keep_raw_response else None,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt <= args.retries:
                time.sleep(args.retry_sleep * attempt)

    return {
        **metadata,
        "index": index,
        "prediction": "",
        "answer": "",
        "error": repr(last_error),
        "source_records": row.get("source_records", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA over qa_prompts.jsonl using a Qwen32B OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"default: {DEFAULT_OUTPUT}")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN32B_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://localhost:8001/v1",
        help="OpenAI-compatible base URL. Can also set QWEN32B_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN32B_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY",
    )
    parser.add_argument("--model", default=os.environ.get("QWEN32B_MODEL", "qwen-32b"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N prompts.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help='Extra JSON fields merged into the request body, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'.',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output instead of appending.",
    )
    parser.add_argument(
        "--keep-raw-response",
        action="store_true",
        help="Store full raw API response in each output row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_body = json.loads(args.extra_body_json)
    rows = read_jsonl(args.input, limit=args.limit)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    print(f"Loaded {len(rows)} prompts from {args.input}")
    print(f"Sending to {endpoint_url(args.base_url)} model={args.model}")

    if args.workers <= 1:
        for index, row in enumerate(rows):
            result = answer_one(index, row, args, extra_body)
            append_jsonl(args.output, [result])
            status = "error" if result.get("error") else "ok"
            print(f"[{index + 1}/{len(rows)}] {status}")
        return

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(answer_one, index, row, args, extra_body): index
            for index, row in enumerate(rows)
        }
        buffered: Dict[int, Dict[str, Any]] = {}
        next_to_write = 0
        for future in futures.as_completed(future_to_index):
            index = future_to_index[future]
            buffered[index] = future.result()
            while next_to_write in buffered:
                result = buffered.pop(next_to_write)
                append_jsonl(args.output, [result])
                status = "error" if result.get("error") else "ok"
                print(f"[{next_to_write + 1}/{len(rows)}] {status}")
                next_to_write += 1


if __name__ == "__main__":
    main()
