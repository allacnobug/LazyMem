#!/usr/bin/env python3
"""Build and run local-window annotation jobs for LazyMem windows.

The script has two steps:
1. build-inputs: select windows whose question_id is in a split file and render
   the system/user prompts into OpenAI-compatible chat messages.
2. annotate: call an OpenAI-compatible chat completion endpoint and save one
   annotation record per input line.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PRE_DIR = Path(__file__).resolve().parent
ROOT = PRE_DIR
DEFAULT_WINDOWS = PRE_DIR / "window" / "top50_ctx2_windows.jsonl"
DEFAULT_IDS = PRE_DIR / "data_splits" / "train_100.txt"
DEFAULT_SYSTEM_PROMPT = PRE_DIR / "prompt" / "annotate_window_system.txt"
DEFAULT_USER_PROMPT = PRE_DIR / "prompt" / "annotate_window_user.txt"
DEFAULT_INPUTS = PRE_DIR / "window_annotation_inputs_train100.jsonl"
DEFAULT_OUTPUT = PRE_DIR / "window_annotations_train100.jsonl"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_ids(path: Path) -> List[str]:
    ids: List[str] = []
    seen = set()
    for line in read_text(path).splitlines():
        qid = line.strip()
        if not qid or qid.startswith("#"):
            continue
        if qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{lineno}: {exc}") from exc


def compact_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = msg.get("content")
    role = msg.get("role")
    if not role or content is None or str(content).strip() == "":
        return None
    # Keep metadata that helps identify the original chronological message, while
    # making the model-facing window concise and stable.
    return {
        "position": msg.get("position"),
        "role": role,
        "timestamp": msg.get("timestamp"),
        "global_message_index": msg.get("global_message_index"),
        "retrieval_rank": msg.get("retrieval_rank"),
        "content": content,
    }


def model_window(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_window = record.get("window") or {}
    rendered: Dict[str, Any] = {}
    for key in ("previous_bridge", "core_messages", "next_bridge"):
        messages = []
        for msg in raw_window.get(key) or []:
            item = compact_message(msg)
            if item is not None:
                messages.append(item)
        rendered[key] = messages
    return rendered


def render_user_prompt(template: str, record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    query = record.get("query", "")
    query_time = record.get("query_timestamp") or record.get("query_time") or ""
    query_block = f"Query Time: {query_time}\nQuery: {query}"
    window = model_window(record)
    prompt = template.replace("{query_time and query}", query_block)
    prompt = prompt.replace("{window}", json.dumps(window, ensure_ascii=False, indent=2))
    return prompt, window


def build_inputs(args: argparse.Namespace) -> None:
    ids = read_ids(args.ids)
    id_set = set(ids)
    system_prompt = read_text(args.system_prompt).strip()
    user_template = read_text(args.user_prompt).strip()

    by_id: Dict[str, List[Dict[str, Any]]] = {qid: [] for qid in ids}
    total_seen = 0
    total_selected = 0
    for record in iter_jsonl(args.windows):
        total_seen += 1
        qid = record.get("question_id")
        if qid in id_set:
            by_id[qid].append(record)
            total_selected += 1

    missing = [qid for qid in ids if not by_id.get(qid)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        sample_index = 0
        for qid in ids:
            for rank_in_question, record in enumerate(by_id.get(qid, [])):
                user_prompt, window = render_user_prompt(user_template, record)
                raw_window = record.get("window") or {}
                item = {
                    "sample_index": sample_index,
                    "question_id": qid,
                    "query": record.get("query"),
                    "query_timestamp": record.get("query_timestamp") or record.get("query_time"),
                    "domain": record.get("domain"),
                    "dataset": record.get("dataset"),
                    "question_type": record.get("question_type"),
                    "rank_in_question": rank_in_question,
                    "segment_index": raw_window.get("segment_index"),
                    "window_index": raw_window.get("window_index"),
                    "segment_type": raw_window.get("segment_type"),
                    "context_window": raw_window.get("context_window"),
                    "model_input_window": window,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                sample_index += 1

    print(
        json.dumps(
            {
                "windows_file": str(args.windows),
                "ids_file": str(args.ids),
                "output": str(args.output),
                "ids": len(ids),
                "windows_seen": total_seen,
                "windows_selected": total_selected,
                "missing_ids": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def check_connectivity(base_url: str, timeout: int) -> None:
    if any(os.environ.get(name) for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")):
        return
    url = normalize_base_url(base_url)
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        raise SystemExit(f"Invalid base URL for connectivity check: {base_url}")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except Exception as exc:
        raise SystemExit(
            f"Connectivity check failed for {host}:{port}: {exc}. "
            "Configure HTTPS proxy/network access, or rerun with --no-connectivity-check if you are sure."
        ) from exc


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    thinking: bool,
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking:
        body.setdefault("chat_template_kwargs", {"enable_thinking": True})
    if extra_body:
        body.update(extra_body)

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        normalize_base_url(base_url),
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_reasoning_and_answer(message: Dict[str, Any]) -> Tuple[str, str]:
    answer = message.get("content") or ""
    reasoning_parts: List[str] = []
    for key in (
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "thinking",
        "thinking_content",
    ):
        value = message.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            reasoning_parts.append(value)
        else:
            reasoning_parts.append(json.dumps(value, ensure_ascii=False))
    return "\n".join(reasoning_parts), answer


def parse_json_array(text: str) -> List[Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            value = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(value, list):
        raise ValueError("model output is not a JSON array")
    return value


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def existing_keys(path: Path) -> set[Tuple[str, Any, Any]]:
    keys = set()
    if not path.exists():
        return keys
    for row in iter_jsonl(path):
        keys.add((row.get("question_id"), row.get("window_index"), row.get("sample_index")))
    return keys


def annotate_one(item: Dict[str, Any], args: argparse.Namespace, api_key: str, extra_body: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    key = (item.get("question_id"), item.get("window_index"), item.get("sample_index"))
    last_error: Optional[BaseException] = None
    for attempt in range(1, args.retries + 2):
        try:
            response = post_chat_completion(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                messages=item["messages"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                thinking=args.thinking,
                extra_body=extra_body,
            )
            message = response["choices"][0]["message"]
            reasoning, answer = extract_reasoning_and_answer(message)
            annotation = parse_json_array(answer)
            result = {
                "sample_index": item.get("sample_index"),
                "question_id": item.get("question_id"),
                "query": item.get("query"),
                "query_timestamp": item.get("query_timestamp"),
                "rank_in_question": item.get("rank_in_question"),
                "segment_index": item.get("segment_index"),
                "window_index": item.get("window_index"),
                "model_input_window": item.get("model_input_window"),
                "messages": item.get("messages"),
                "model_reasoning": reasoning,
                "model_answer": answer,
                "annotation": annotation,
                "raw_model_message": message,
                "model": args.model,
                "request_params": {
                    "base_url": args.base_url,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "thinking": args.thinking,
                    "thinking_budget": 1024 if args.thinking else None,
                    "reasoning_effort": "high" if args.thinking else None,
                    "thinking_body": {"type": "enabled"} if args.thinking else None,
                    "extra_body": extra_body,
                    "concurrency": args.concurrency,
                },
                "raw_response": response,
                "usage": response.get("usage"),
            }
            if args.sleep > 0:
                time.sleep(args.sleep)
            return "success", result, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt <= args.retries:
                wait = args.retry_sleep * (2 ** (attempt - 1)) + random.random()
                print(f"retry {attempt}/{args.retries} for {key}: {exc}; sleep {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                err_item = {
                    "sample_index": item.get("sample_index"),
                    "question_id": item.get("question_id"),
                    "window_index": item.get("window_index"),
                    "error": repr(last_error),
                    "model": args.model,
                }
                return "failed", err_item, repr(last_error)
    raise RuntimeError("unreachable")


def annotate(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = args.api_key or os.environ.get(args.api_key_env, "")

    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    if args.connectivity_check:
        check_connectivity(args.base_url, args.connectivity_timeout)
    done = existing_keys(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pending: List[Dict[str, Any]] = []
    skipped = 0
    for item in iter_jsonl(args.input):
        key = (item.get("question_id"), item.get("window_index"), item.get("sample_index"))
        if key in done:
            skipped += 1
            continue
        if args.limit is not None and len(pending) >= args.limit:
            break
        pending.append(item)

    total = skipped + len(pending)
    processed = 0
    failed = 0
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise SystemExit(
                "tqdm is required for --progress. Install it with: python3 -m pip install --user tqdm"
            ) from exc
        pbar = tqdm(total=total, desc="annotate windows", unit="win", dynamic_ncols=True)
        if skipped:
            pbar.update(skipped)
            pbar.set_postfix(processed=processed, skipped=skipped, failed=failed, refresh=False)

    def update_progress() -> None:
        if pbar is None:
            return
        pbar.update(1)
        pbar.set_postfix(processed=processed, skipped=skipped, failed=failed, refresh=False)

    try:
        with args.output.open("a", encoding="utf-8") as out:
            error_out = None
            try:
                if args.error_output:
                    args.error_output.parent.mkdir(parents=True, exist_ok=True)
                    error_out = args.error_output.open("a", encoding="utf-8")
                with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
                    futures = [executor.submit(annotate_one, item, args, api_key, extra_body) for item in pending]
                    for future in as_completed(futures):
                        status, payload, error = future.result()
                        if status == "success":
                            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            out.flush()
                            processed += 1
                        else:
                            failed += 1
                            if error_out is not None:
                                error_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                                error_out.flush()
                            print(f"failed {(payload.get(question_id), payload.get(window_index), payload.get(sample_index))}: {error}", file=sys.stderr)
                            if args.fail_fast:
                                raise RuntimeError(error)
                        if pbar is not None:
                            update_progress()
                        elif (processed + failed) % args.progress_every == 0:
                            print(f"processed={processed} skipped={skipped} failed={failed}", file=sys.stderr)
            finally:
                if error_out is not None:
                    error_out.close()
    finally:
        if pbar is not None:
            pbar.close()

    print(json.dumps({"processed": processed, "skipped": skipped, "failed": failed, "output": str(args.output), "concurrency": args.concurrency}, ensure_ascii=False))

def add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", type=Path, default=DEFAULT_USER_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-inputs", help="render model inputs for selected split ids")
    add_common_build_args(p_build)
    p_build.set_defaults(func=build_inputs)

    p_annotate = sub.add_parser("annotate", help="call a local OpenAI-compatible API for rendered inputs")
    p_annotate.add_argument("--input", type=Path, default=DEFAULT_INPUTS)
    p_annotate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p_annotate.add_argument("--error-output", type=Path, default=PRE_DIR / "window_annotations_train100_errors.jsonl")
    p_annotate.add_argument(
        "--base-url",
        default=os.environ.get("LAZYMEM_TEACHER_BASE_URL", "http://127.0.0.1:8022/v1"),
    )
    p_annotate.add_argument("--api-key", default="")
    p_annotate.add_argument("--api-key-env", default="LAZYMEM_TEACHER_API_KEY")
    p_annotate.add_argument(
        "--model",
        default=os.environ.get("LAZYMEM_TEACHER_MODEL", "qwen-32b"),
    )
    p_annotate.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode")
    p_annotate.add_argument("--extra-body-json", default="", help="JSON object merged into the request body")
    p_annotate.add_argument("--temperature", type=float, default=0.0)
    p_annotate.add_argument("--max-tokens", type=int, default=4096)
    p_annotate.add_argument("--timeout", type=int, default=180)
    p_annotate.add_argument("--retries", type=int, default=3)
    p_annotate.add_argument("--retry-sleep", type=float, default=5.0)
    p_annotate.add_argument("--sleep", type=float, default=0.0)
    p_annotate.add_argument("--limit", type=int, default=None)
    p_annotate.add_argument("--resume", action="store_true", help="skip records already present in output")
    p_annotate.add_argument("--fail-fast", action="store_true")
    p_annotate.add_argument("--progress-every", type=int, default=10)
    p_annotate.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="show a live stderr progress bar")
    p_annotate.add_argument("--concurrency", type=int, default=16, help="number of concurrent API requests")
    p_annotate.add_argument("--connectivity-timeout", type=int, default=10)
    p_annotate.add_argument("--connectivity-check", action=argparse.BooleanOptionalAction, default=True)
    p_annotate.set_defaults(func=annotate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
