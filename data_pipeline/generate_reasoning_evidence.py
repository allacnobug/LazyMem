"""Generate question-level reasoning evidence for the GRPO quality reward."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bars are optional
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_FILE = REPO_ROOT / "data" / "longmemeval_s_cleaned.json"
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reward"
DEFAULT_ENDPOINT = "http://127.0.0.1:8022/v1/chat/completions"
DEFAULT_MODEL = "qwen-32b"

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a careful reasoning distillation assistant. Given a memory QA query, "
    "query timestamp, supporting evidence, and an answer for silent consistency "
    "checking, write a concise and faithful explanation of how the evidence supports "
    "the answer. Use only the supplied evidence and temporal clues. Do not mention "
    "gold labels, ground truth, annotations, reference answers, or datasets. Do not "
    "output a separate short answer; output only the evidence-based reasoning."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        question_id = line.strip()
        if not question_id or question_id.startswith("#") or question_id in seen:
            continue
        ids.append(question_id)
        seen.add(question_id)
    return ids


def evidence_turns(entry: dict[str, Any]) -> list[dict[str, str]]:
    """Select the same question-level evidence turns used by the original pipeline."""

    answer_session_ids = {str(value) for value in entry.get("answer_session_ids") or []}
    all_answer_session_turns: list[dict[str, str]] = []
    marked_answer_turns: list[dict[str, str]] = []

    session_ids = entry.get("haystack_session_ids") or []
    session_dates = entry.get("haystack_dates") or []
    sessions = entry.get("haystack_sessions") or []
    for session_id, timestamp, session in zip(session_ids, session_dates, sessions):
        if str(session_id) not in answer_session_ids:
            continue
        for message in session or []:
            if not isinstance(message, dict):
                continue
            turn = {
                "timestamp": str(timestamp or ""),
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            if not turn["content"].strip():
                continue
            all_answer_session_turns.append(turn)
            if bool(message.get("has_answer")):
                marked_answer_turns.append(turn)

    question_id = str(entry.get("question_id") or "")
    if "_abs" in question_id:
        return all_answer_session_turns
    return marked_answer_turns or all_answer_session_turns


def build_user_prompt(entry: dict[str, Any], turns: list[dict[str, str]]) -> str:
    evidence_blocks = [
        (
            f"[{index}] timestamp: {turn['timestamp']}\n"
            f"role: {turn['role']}\n"
            f"content: {turn['content']}"
        )
        for index, turn in enumerate(turns, start=1)
    ]
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
    return (
        f"query: {entry.get('question', '')}\n"
        f"query_timestamp: {entry.get('question_date', '')}\n"
        f"evidence:\n{evidence}\n\n"
        f"answer_for_silent_check_only: {entry.get('answer', '')}\n\n"
        "Write a natural evidence-based reasoning process in 3-8 sentences. First "
        "state what is being asked, then identify the relevant facts and temporal "
        "relationships, and finally explain how they lead to the answer. If the "
        "evidence is insufficient, explain what is missing. Do not mention or quote "
        "the silent-check answer, and do not output a separate final answer."
    )


def split_thinking(text: str) -> tuple[str, str]:
    """Return an optional hidden think block and the visible reasoning text."""

    match = THINK_RE.search(text)
    if match is None:
        return "", text.strip()
    thinking = match.group(1).strip()
    visible = (text[: match.start()] + text[match.end() :]).strip()
    return thinking, visible


def call_model(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    entry: dict[str, Any],
    turns: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
    enable_thinking: bool,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(entry, turns)},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            message = response.json().get("choices", [{}])[0].get("message", {})
            content = str(message.get("content") or "").strip()
            reasoning_content = str(message.get("reasoning_content") or "").strip()
            thinking, visible = split_thinking(content)
            if visible:
                return reasoning_content or thinking, visible
            raise RuntimeError("model returned empty visible reasoning")
        except Exception as exc:  # requests exposes several transport exceptions
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        question_id = row.get("question_id")
        evidence = row.get("reasoning_answer")
        if isinstance(question_id, str) and isinstance(evidence, str) and evidence.strip():
            completed[question_id] = row
    return completed


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_split(
    *,
    split: str,
    question_ids: list[str],
    dataset_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    endpoint: str,
    model: str,
    api_key: str,
    workers: int,
    timeout: int,
    retries: int,
    max_tokens: int,
    enable_thinking: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_path = output_dir / split / "reasoning_qwen32b.split.jsonl"
    error_path = output_dir / split / "reasoning_qwen32b.errors.jsonl"
    completed = {} if overwrite else load_completed(output_path)
    missing_dataset_ids = [question_id for question_id in question_ids if question_id not in dataset_by_id]
    if missing_dataset_ids:
        preview = ", ".join(missing_dataset_ids[:5])
        raise ValueError(
            f"{split}: {len(missing_dataset_ids)} question IDs are absent from the dataset: {preview}"
        )

    pending = [question_id for question_id in question_ids if question_id not in completed]
    errors: list[dict[str, Any]] = []
    progress = tqdm(total=len(pending), desc=split, unit="question") if tqdm is not None else None

    def generate(question_id: str) -> dict[str, Any]:
        entry = dataset_by_id[question_id]
        turns = evidence_turns(entry)
        if not turns:
            raise ValueError("no evidence turns found in answer sessions")
        thinking, reasoning = call_model(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            entry=entry,
            turns=turns,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        return {
            "question_id": question_id,
            "reasoning": reasoning,
            "reasoning_think": thinking,
            "reasoning_answer": reasoning,
            "gold_answer": str(entry.get("answer") or ""),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(generate, question_id): question_id for question_id in pending
        }
        for future in as_completed(future_to_id):
            question_id = future_to_id[future]
            try:
                completed[question_id] = future.result()
            except Exception as exc:
                errors.append({"question_id": question_id, "error": str(exc)})
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    ordered_rows = [completed[question_id] for question_id in question_ids if question_id in completed]
    write_jsonl(output_path, ordered_rows)
    if errors:
        write_jsonl(error_path, errors)
    elif error_path.exists():
        error_path.unlink()

    report = {
        "split": split,
        "requested_questions": len(question_ids),
        "previously_completed": len(question_ids) - len(pending),
        "generated_this_run": len(pending) - len(errors),
        "failed_this_run": len(errors),
        "output_rows": len(ordered_rows),
        "output_file": str(output_path),
        "error_file": str(error_path) if errors else None,
    }
    print(json.dumps(report, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reasoning_qwen32b.split.jsonl files for GRPO rewards."
    )
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key-env",
        default="LAZYMEM_REASONING_API_KEY",
        help="Environment variable containing an optional bearer token.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.retries <= 0:
        raise ValueError("--retries must be positive")

    dataset = load_json(args.dataset_file)
    if not isinstance(dataset, list):
        raise ValueError(f"{args.dataset_file}: expected a JSON array")
    dataset_by_id = {
        str(entry["question_id"]): entry
        for entry in dataset
        if isinstance(entry, dict) and entry.get("question_id") is not None
    }
    api_key = os.environ.get(args.api_key_env, "")

    reports = []
    for split in args.splits:
        question_ids = read_ids(args.split_dir / f"{split}.txt")
        reports.append(
            process_split(
                split=split,
                question_ids=question_ids,
                dataset_by_id=dataset_by_id,
                output_dir=args.output_dir,
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
                max_tokens=args.max_tokens,
                enable_thinking=not args.disable_thinking,
                overwrite=args.overwrite,
            )
        )

    manifest_path = args.output_dir / "reasoning_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_file": str(args.dataset_file),
                "split_dir": str(args.split_dir),
                "endpoint": args.endpoint,
                "model": args.model,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    failed = sum(int(report["failed_this_run"]) for report in reports)
    if failed:
        raise SystemExit(f"{failed} questions failed; rerun to resume after inspecting error files")


if __name__ == "__main__":
    main()
