#!/usr/bin/env python3
"""Validate LazyMem window SFT data or prediction outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIVATE_WINDOWS = REPO_ROOT / "data" / "processed" / "top50_ctx2_windows_private.jsonl"
WINDOW_SECTIONS = ("previous_bridge", "core_messages", "next_bridge")
DECISION_KEYS = {"op", "compressed_content", "reason"}
Row = dict[str, Any]
Key = tuple[str, Any, Any]


def iter_rows(path: Path):
    """Read either JSON array data or JSONL predictions."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                yield 1, {"__load_error__": str(exc)}
                return
            if not isinstance(data, list):
                yield 1, {"__load_error__": "top-level JSON is not an array"}
                return
            for idx, row in enumerate(data, 1):
                yield idx, row if isinstance(row, dict) else {"__load_error__": "row is not object"}
            return

        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, {"__load_error__": str(exc)}
                continue
            yield lineno, row if isinstance(row, dict) else {"__load_error__": "row is not object"}


def metadata(row: Row) -> Row:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else row


def private_key(row: Row) -> Key:
    meta = metadata(row)
    return str(meta.get("question_id")), meta.get("segment_index"), meta.get("window_index")


def load_gold_map(path: Path) -> dict[Key, set[int]]:
    gold_map: dict[Key, set[int]] = {}
    if not path.exists():
        return gold_map
    for _, row in iter_rows(path):
        if "__load_error__" in row:
            continue
        gold: set[int] = set()
        for value in row.get("gold_global_message_idxs") or []:
            try:
                gold.add(int(value))
            except (TypeError, ValueError):
                continue
        gold_map[private_key(row)] = gold
    return gold_map


def last_message_content(row: Row, role: str) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == role:
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def candidate_answer_text(row: Row) -> Any:
    for key in (
        "predict",
        "prediction",
        "response",
        "output",
        "generated_text",
        "generated_response",
        "model_answer",
        "answer",
    ):
        value = row.get(key)
        if isinstance(value, str):
            return value
    for key in ("predicts", "predictions", "outputs"):
        value = row.get(key)
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return last_message_content(row, "assistant")


def strip_think(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def parse_answer(answer: Any) -> tuple[Any, str | None]:
    if not isinstance(answer, str):
        return None, "answer is not a string"
    text = strip_think(answer)
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                inner = "\n".join(lines[1:-1]).strip()
                try:
                    return json.loads(inner), "answer is fenced markdown, not directly parseable JSON"
                except json.JSONDecodeError as exc:
                    return None, f"answer fenced content is invalid JSON: {exc}"
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1]), "answer has extra text around JSON"
            except json.JSONDecodeError as exc:
                return None, f"answer embedded JSON is invalid: {exc}"
        return None, "answer is invalid JSON"


def extract_window(row: Row) -> Row | None:
    window = row.get("model_input_window")
    if isinstance(window, dict):
        return window
    user_content = ""
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_content = str(msg.get("content") or "")
    marker = "# History Window"
    if marker not in user_content:
        return None
    try:
        parsed = json.loads(user_content.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def window_messages(row: Row) -> list[Row] | None:
    window = extract_window(row)
    if not isinstance(window, dict):
        return None
    result: list[Row] = []
    for section in WINDOW_SECTIONS:
        messages = window.get(section)
        if not isinstance(messages, list):
            return None
        for msg in messages:
            if not isinstance(msg, dict):
                return None
            if str(msg.get("role") or "").strip() and str(msg.get("content") or "").strip():
                result.append(msg)
    return result


def validate_decisions(row: Row, decisions: Any, require_reason: bool, require_exact_keys: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(decisions, list):
        return ["answer is not a JSON array"]

    messages = window_messages(row)
    if messages is None:
        errors.append("missing or invalid history window")
    elif len(decisions) != len(messages):
        errors.append(f"decision count mismatch: answer={len(decisions)} messages={len(messages)}")

    for idx, item in enumerate(decisions):
        if not isinstance(item, dict):
            errors.append(f"decision[{idx}] is not an object")
            continue
        keys = set(item.keys())
        if require_exact_keys and keys != DECISION_KEYS:
            errors.append(f"decision[{idx}] keys mismatch: {sorted(keys)}")
        else:
            missing = DECISION_KEYS - keys
            if missing:
                errors.append(f"decision[{idx}] missing keys: {sorted(missing)}")
        op = item.get("op")
        content = item.get("compressed_content")
        reason = item.get("reason")
        if op not in {"KEEP", "DROP"}:
            errors.append(f"decision[{idx}] invalid op: {op!r}")
        if not isinstance(content, str):
            errors.append(f"decision[{idx}] compressed_content is not string")
        if require_reason and (not isinstance(reason, str) or not reason.strip()):
            errors.append(f"decision[{idx}] reason is empty or not string")
        if op == "DROP" and content != "":
            errors.append(f"decision[{idx}] DROP has non-empty compressed_content")
        if op == "KEEP" and (not isinstance(content, str) or not content.strip()):
            errors.append(f"decision[{idx}] KEEP has empty compressed_content")
    return errors


def gold_keep_drop_counts(row: Row, decisions: Any, gold_map: dict[Key, set[int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(decisions, list):
        return counts
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
        try:
            global_idx = int(msg.get("global_message_index"))
        except (TypeError, ValueError):
            counts["missing_message_global_idx"] += 1
            continue
        prefix = "gold" if global_idx in gold else "other"
        op = decision.get("op")
        if op == "KEEP":
            counts[f"{prefix}_keep"] += 1
        elif op == "DROP":
            counts[f"{prefix}_drop"] += 1
    return counts


def op_match_counts(pred: Any, ref: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(pred, list) or not isinstance(ref, list) or len(pred) != len(ref):
        return counts
    for pred_item, ref_item in zip(pred, ref):
        if not isinstance(pred_item, dict) or not isinstance(ref_item, dict):
            continue
        pred_op = pred_item.get("op")
        ref_op = ref_item.get("op")
        if pred_op in {"KEEP", "DROP"} and ref_op in {"KEEP", "DROP"}:
            counts["op_total"] += 1
            if pred_op == ref_op:
                counts["op_match"] += 1
            else:
                counts[f"op_{ref_op.lower()}_as_{pred_op.lower()}"] += 1
    return counts


def pct(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 6)


def row_id(row: Row) -> tuple[str, str, str, Any]:
    meta = metadata(row)
    return (
        str(meta.get("question_id")),
        str(meta.get("segment_index")),
        str(meta.get("window_index")),
        meta.get("sample_index"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="SFT JSON/JSONL file or prediction JSONL file")
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="Optional SFT JSON/JSONL file that supplies windows/reference answers for prediction files.",
    )
    parser.add_argument("--private-windows", type=Path, default=DEFAULT_PRIVATE_WINDOWS)
    parser.add_argument("--require-reason", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-exact-keys", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    refs = []
    if args.reference_data:
        refs = [row for _, row in iter_rows(args.reference_data) if "__load_error__" not in row]

    gold_map = load_gold_map(args.private_windows)
    totals = Counter()
    issue_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    examples: list[tuple[int, tuple[str, str, str, Any], list[str]]] = []

    for idx, (lineno, raw_row) in enumerate(iter_rows(args.path)):
        totals["total"] += 1
        ref_row = refs[idx] if idx < len(refs) else None
        row = raw_row
        if ref_row is not None:
            row = {**ref_row, **raw_row}
            if "messages" not in raw_row and "messages" in ref_row:
                row["messages"] = ref_row["messages"]
            if "metadata" not in raw_row and "metadata" in ref_row:
                row["metadata"] = ref_row["metadata"]

        if "__load_error__" in raw_row:
            decisions = None
            errors = [f"invalid row JSON: {raw_row['__load_error__']}"]
        else:
            decisions, warning = parse_answer(candidate_answer_text(row))
            errors = [warning] if warning else []
            errors.extend(validate_decisions(row, decisions, args.require_reason, args.require_exact_keys))
            outcome_counts.update(gold_keep_drop_counts(row, decisions, gold_map))
            if ref_row is not None:
                ref_decisions, _ = parse_answer(candidate_answer_text(ref_row))
                match_counts.update(op_match_counts(decisions, ref_decisions))

        if errors:
            totals["bad"] += 1
            issue_counts.update(errors)
            if len(examples) < args.max_examples:
                examples.append((lineno, row_id(row), errors))

    gold_total = outcome_counts["gold_keep"] + outcome_counts["gold_drop"]
    other_total = outcome_counts["other_keep"] + outcome_counts["other_drop"]
    keep_total = outcome_counts["gold_keep"] + outcome_counts["other_keep"]
    report: dict[str, Any] = {
        "file": str(args.path),
        "reference_data": str(args.reference_data) if args.reference_data else None,
        "private_windows": str(args.private_windows),
        "total": totals["total"],
        "bad": totals["bad"],
        "ok": totals["total"] - totals["bad"],
        "gold_keep": outcome_counts["gold_keep"],
        "gold_drop": outcome_counts["gold_drop"],
        "other_keep": outcome_counts["other_keep"],
        "other_drop": outcome_counts["other_drop"],
        "gold_keep_rate": pct(outcome_counts["gold_keep"], gold_total),
        "other_drop_rate": pct(outcome_counts["other_drop"], other_total),
        "keep_precision": pct(outcome_counts["gold_keep"], keep_total),
        "missing_gold_window": outcome_counts["missing_gold_window"],
        "missing_message_global_idx": outcome_counts["missing_message_global_idx"],
    }
    if match_counts:
        report.update(
            {
                "op_match": match_counts["op_match"],
                "op_total": match_counts["op_total"],
                "op_accuracy_vs_reference": pct(match_counts["op_match"], match_counts["op_total"]),
                "op_keep_as_drop": match_counts["op_keep_as_drop"],
                "op_drop_as_keep": match_counts["op_drop_as_keep"],
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issue_counts:
        print("\nIssue summary:")
        for issue, count in issue_counts.most_common():
            print(f"{count}\t{issue}")
    if examples:
        print("\nExamples:")
        for lineno, (qid, seg, win, sample), errors in examples:
            print(
                f"line={lineno} question_id={qid} segment_index={seg} "
                f"window_index={win} sample_index={sample}"
            )
            for err in errors:
                print(f"  - {err}")

    raise SystemExit(1 if totals["bad"] else 0)


if __name__ == "__main__":
    main()
