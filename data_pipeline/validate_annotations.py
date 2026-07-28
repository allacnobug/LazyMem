#!/usr/bin/env python3
"""Validate LazyMem window annotation JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, {"__load_error__": str(exc)}


def parse_answer(answer: Any) -> Tuple[Any, str | None]:
    if not isinstance(answer, str):
        return None, "model_answer is not a string"
    text = answer.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        # Be tolerant of accidental markdown fences, but report that the raw
        # answer was not directly parseable as requested by the prompt.
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                inner = "\n".join(lines[1:-1]).strip()
                try:
                    return json.loads(inner), "model_answer is fenced markdown, not directly parseable JSON"
                except json.JSONDecodeError as exc:
                    return None, f"model_answer fenced content is invalid JSON: {exc}"
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1]), "model_answer has extra text around JSON"
            except json.JSONDecodeError as exc:
                return None, f"model_answer embedded JSON is invalid: {exc}"
        return None, "model_answer is invalid JSON"


def extract_window(row: Dict[str, Any]) -> Dict[str, Any] | None:
    window = row.get("model_input_window")
    if isinstance(window, dict):
        return window
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    user_content = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_content = str(msg.get("content") or "")
    marker = "# History Window"
    if marker not in user_content:
        return None
    window_text = user_content.split(marker, 1)[1].strip()
    try:
        parsed = json.loads(window_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def count_window_messages(row: Dict[str, Any]) -> int | None:
    window = extract_window(row)
    if not isinstance(window, dict):
        return None
    count = 0
    for key in ("previous_bridge", "core_messages", "next_bridge"):
        messages = window.get(key)
        if not isinstance(messages, list):
            return None
        for msg in messages:
            if not isinstance(msg, dict):
                return None
            if str(msg.get("role") or "").strip() and str(msg.get("content") or "").strip():
                count += 1
    return count


def window_messages(row: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    window = extract_window(row)
    if not isinstance(window, dict):
        return None
    result: List[Dict[str, Any]] = []
    for key in ("previous_bridge", "core_messages", "next_bridge"):
        messages = window.get(key)
        if not isinstance(messages, list):
            return None
        for msg in messages:
            if not isinstance(msg, dict):
                return None
            if str(msg.get("role") or "").strip() and str(msg.get("content") or "").strip():
                result.append(msg)
    return result


def private_key(row: Dict[str, Any]) -> Tuple[str, Any, Any]:
    return (
        str(row.get("question_id")),
        row.get("segment_index"),
        row.get("window_index"),
    )


def load_gold_map(path: Path) -> Dict[Tuple[str, Any, Any], set[int]]:
    gold_map: Dict[Tuple[str, Any, Any], set[int]] = {}
    for _, row in iter_jsonl(path):
        if "__load_error__" in row:
            continue
        raw_gold = row.get("gold_global_message_idxs") or []
        gold: set[int] = set()
        for value in raw_gold:
            try:
                gold.add(int(value))
            except (TypeError, ValueError):
                continue
        gold_map[private_key(row)] = gold
    return gold_map


def gold_keep_drop_counts(row: Dict[str, Any], decisions: List[Any], gold_map: Dict[Tuple[str, Any, Any], set[int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    messages = window_messages(row)
    if messages is None or len(messages) != len(decisions):
        return counts
    gold = gold_map.get(private_key(row))
    if gold is None:
        counts["missing_gold_window"] += 1
        return counts
    for msg, decision in zip(messages, decisions):
        if not isinstance(decision, dict):
            continue
        op = decision.get("op")
        try:
            global_idx = int(msg.get("global_message_index"))
        except (TypeError, ValueError):
            counts["missing_message_global_idx"] += 1
            continue
        prefix = "gold" if global_idx in gold else "other"
        if op == "KEEP":
            counts[f"{prefix}_keep"] += 1
        elif op == "DROP":
            counts[f"{prefix}_drop"] += 1
    return counts


def validate_row(row: Dict[str, Any], require_reasoning: bool) -> List[str]:
    errors: List[str] = []
    if "__load_error__" in row:
        return [f"invalid output JSONL row: {row['__load_error__']}"]

    if require_reasoning and not str(row.get("model_reasoning") or "").strip():
        errors.append("model_reasoning is empty")

    parsed_answer, answer_warning = parse_answer(row.get("model_answer"))
    if answer_warning:
        errors.append(answer_warning)
    if not isinstance(parsed_answer, list):
        errors.append("model_answer is not a JSON array")
        return errors

    annotation = row.get("annotation")
    if annotation != parsed_answer:
        errors.append("annotation does not exactly match parsed model_answer")

    expected_count = count_window_messages(row)
    if expected_count is None:
        errors.append("missing or invalid model_input_window")
    elif len(parsed_answer) != expected_count:
        errors.append(f"decision count mismatch: answer={len(parsed_answer)} messages={expected_count}")

    for idx, item in enumerate(parsed_answer):
        if not isinstance(item, dict):
            errors.append(f"decision[{idx}] is not an object")
            continue
        keys = set(item.keys())
        if keys != {"op", "compressed_content", "reason"}:
            errors.append(f"decision[{idx}] keys mismatch: {sorted(keys)}")
        op = item.get("op")
        content = item.get("compressed_content")
        reason = item.get("reason")
        if op not in {"KEEP", "DROP"}:
            errors.append(f"decision[{idx}] invalid op: {op!r}")
        if not isinstance(content, str):
            errors.append(f"decision[{idx}] compressed_content is not string")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"decision[{idx}] reason is empty or not string")
        if op == "DROP" and content != "":
            errors.append(f"decision[{idx}] DROP has non-empty compressed_content")
        if op == "KEEP" and (not isinstance(content, str) or not content.strip()):
            errors.append(f"decision[{idx}] KEEP has empty compressed_content")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--private-windows", type=Path, default=Path(__file__).resolve().parent / "window" / "top50_ctx2_windows_private.jsonl")
    parser.add_argument("--require-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    total = 0
    bad = 0
    issue_counts: Counter[str] = Counter()
    keep_drop_counts: Counter[str] = Counter()
    gold_map = load_gold_map(args.private_windows)
    examples: List[Tuple[int, str, str, int | None, List[str]]] = []

    for lineno, row in iter_jsonl(args.path):
        total += 1
        parsed_answer, _ = parse_answer(row.get("model_answer"))
        if isinstance(parsed_answer, list):
            keep_drop_counts.update(gold_keep_drop_counts(row, parsed_answer, gold_map))
        errors = validate_row(row, args.require_reasoning)
        if errors:
            bad += 1
            for err in errors:
                issue_counts[err] += 1
            if len(examples) < args.max_examples:
                examples.append(
                    (
                        lineno,
                        str(row.get("question_id")),
                        str(row.get("window_index")),
                        row.get("sample_index"),
                        errors,
                    )
                )

    print(json.dumps({
        "file": str(args.path),
        "private_windows": str(args.private_windows),
        "total": total,
        "bad": bad,
        "ok": total - bad,
        "gold_keep": keep_drop_counts.get("gold_keep", 0),
        "gold_drop": keep_drop_counts.get("gold_drop", 0),
        "other_keep": keep_drop_counts.get("other_keep", 0),
        "other_drop": keep_drop_counts.get("other_drop", 0),
        "missing_gold_window": keep_drop_counts.get("missing_gold_window", 0),
        "missing_message_global_idx": keep_drop_counts.get("missing_message_global_idx", 0),
    }, ensure_ascii=False, indent=2))
    if issue_counts:
        print("\nIssue summary:")
        for issue, count in issue_counts.most_common():
            print(f"{count}\t{issue}")
    if examples:
        print("\nExamples:")
        for lineno, qid, window_index, sample_index, errors in examples:
            print(f"line={lineno} question_id={qid} window_index={window_index} sample_index={sample_index}")
            for err in errors:
                print(f"  - {err}")

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
