"""Convert validated teacher annotations into LLaMA-Factory SFT datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            yield row


def write_json_array(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)
        f.write("\n")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_FILE = REPO_ROOT / "data" / "processed" / "window_annotations_train100.jsonl"
DEFAULT_PRIVATE_WINDOWS = REPO_ROOT / "data" / "processed" / "top50_ctx2_windows_private.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "data" / "sft"
DEFAULT_DATASET_NAME = "lazymem_window_dft_train100_balanced"


def private_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
    return (str(row.get("question_id")), row.get("segment_index"), row.get("window_index"))


def load_gold_map(path: Path) -> dict[tuple[str, Any, Any], set[int]]:
    gold_map: dict[tuple[str, Any, Any], set[int]] = {}
    for row in iter_jsonl(path):
        gold: set[int] = set()
        for value in row.get("gold_global_message_idxs") or []:
            try:
                gold.add(int(value))
            except (TypeError, ValueError):
                continue
        gold_map[private_key(row)] = gold
    return gold_map


def parse_answer(answer: Any) -> list[dict[str, Any]]:
    if not isinstance(answer, str):
        raise ValueError("model_answer is not a string")
    parsed = json.loads(answer.strip())
    if not isinstance(parsed, list):
        raise ValueError("model_answer is not a JSON array")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("model_answer contains non-object decisions")
    return parsed


def extract_window(row: dict[str, Any]) -> dict[str, Any] | None:
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


def window_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    window = extract_window(row)
    if not isinstance(window, dict):
        raise ValueError("missing or invalid model_input_window")
    result: list[dict[str, Any]] = []
    for key in ("previous_bridge", "core_messages", "next_bridge"):
        messages = window.get(key)
        if not isinstance(messages, list):
            raise ValueError(f"missing or invalid window section: {key}")
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"invalid message in window section: {key}")
            if str(msg.get("role") or "").strip() and str(msg.get("content") or "").strip():
                result.append(msg)
    return result


def row_decisions(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = row.get("annotation")
    if isinstance(annotation, list) and all(isinstance(item, dict) for item in annotation):
        return annotation
    return parse_answer(row.get("model_answer"))


def decision_stats(
    row: dict[str, Any],
    decisions: list[dict[str, Any]],
    gold_map: dict[tuple[str, Any, Any], set[int]],
) -> tuple[Counter[str], list[int]]:
    messages = window_messages(row)
    if len(messages) != len(decisions):
        raise ValueError(f"decision count mismatch: decisions={len(decisions)} messages={len(messages)}")
    gold = gold_map.get(private_key(row))
    if gold is None:
        raise ValueError("missing gold window")

    counts: Counter[str] = Counter()
    gold_drop_indices: list[int] = []
    for idx, (msg, decision) in enumerate(zip(messages, decisions)):
        op = decision.get("op")
        try:
            global_idx = int(msg.get("global_message_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("missing message global idx") from exc
        prefix = "gold" if global_idx in gold else "other"
        if op == "KEEP":
            counts[f"{prefix}_keep"] += 1
        elif op == "DROP":
            counts[f"{prefix}_drop"] += 1
            if prefix == "gold":
                gold_drop_indices.append(global_idx)
        else:
            counts["invalid_op"] += 1
    return counts, gold_drop_indices


def assistant_content(row: dict[str, Any]) -> str:
    reasoning = str(row.get("model_reasoning") or "").strip()
    answer = str(row.get("model_answer") or "").strip()
    if not reasoning:
        raise ValueError("model_reasoning is empty")
    if not answer:
        raise ValueError("model_answer is empty")
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def to_llamafactory_row(
    row: dict[str, Any],
    *,
    category: str,
    stats: Counter[str],
    gold_drop_indices: list[int],
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages is not a list")
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("messages contains an invalid chat message")

    metadata = {
        "question_id": row.get("question_id"),
        "segment_index": row.get("segment_index"),
        "window_index": row.get("window_index"),
        "sample_index": row.get("sample_index"),
        "rank_in_question": row.get("rank_in_question"),
        "query": row.get("query"),
        "query_timestamp": row.get("query_timestamp"),
        "source_model": row.get("model"),
        "source_file_category": category,
        "gold_keep": stats.get("gold_keep", 0),
        "gold_drop": stats.get("gold_drop", 0),
        "other_keep": stats.get("other_keep", 0),
        "other_drop": stats.get("other_drop", 0),
        "gold_drop_global_message_idxs": gold_drop_indices,
        "thinking_sft": True,
        "schema": "window_dft_qwen4b_reasoning_answer_v1",
    }
    return {
        "messages": [*messages, {"role": "assistant", "content": assistant_content(row)}],
        "metadata": metadata,
    }


def dataset_info_for(dataset_name: str, file_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    gold_map: dict[tuple[str, Any, Any], set[int]],
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    keep_rows: list[dict[str, Any]] = []
    drop_only_rows: list[dict[str, Any]] = []
    skipped = Counter()
    global_stats = Counter()
    examples: dict[str, Any] = {}

    for row in raw_rows:
        try:
            decisions = row_decisions(row)
            stats, gold_drop_indices = decision_stats(row, decisions, gold_map)
        except Exception as exc:
            skipped["invalid_row"] += 1
            examples.setdefault("invalid_row", str(exc))
            continue

        global_stats.update(stats)
        if stats.get("invalid_op", 0) > 0:
            skipped["invalid_op"] += 1
            continue

        has_keep = stats.get("gold_keep", 0) + stats.get("other_keep", 0) > 0
        has_drop = stats.get("gold_drop", 0) + stats.get("other_drop", 0) > 0
        if mode == "all":
            if has_keep:
                category = "keep_row"
            elif has_drop:
                category = "drop_only_row"
            else:
                category = "empty_decision_row"
            try:
                all_rows.append(
                    to_llamafactory_row(
                        row,
                        category=category,
                        stats=stats,
                        gold_drop_indices=gold_drop_indices,
                    )
                )
            except Exception as exc:
                skipped["conversion_error"] += 1
                examples.setdefault("conversion_error", str(exc))
            continue

        if stats.get("gold_drop", 0) > 0:
            skipped["gold_drop_row"] += 1
            continue

        try:
            if has_keep:
                keep_rows.append(
                    to_llamafactory_row(
                        row,
                        category="keep_row",
                        stats=stats,
                        gold_drop_indices=gold_drop_indices,
                    )
                )
            elif has_drop:
                drop_only_rows.append(
                    to_llamafactory_row(
                        row,
                        category="drop_only_row",
                        stats=stats,
                        gold_drop_indices=gold_drop_indices,
                    )
                )
            else:
                skipped["empty_decision_row"] += 1
        except Exception as exc:
            skipped["conversion_error"] += 1
            examples.setdefault("conversion_error", str(exc))

    report = {
        "raw_decision_stats": dict(global_stats),
        "skipped": dict(skipped),
        "examples": examples,
    }
    if mode == "all":
        return all_rows, [], report
    return keep_rows, drop_only_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build balanced LLaMA-Factory DFT/SFT data from LazyMem window annotations."
    )
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--private-windows", type=Path, default=DEFAULT_PRIVATE_WINDOWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--mode",
        choices=["balanced-train", "all"],
        default="balanced-train",
        help="balanced-train filters gold-drop rows and balances drop-only rows; all converts every valid row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_name = args.dataset_name + ".json"
    target_path = args.output_dir / target_name
    dataset_info_path = args.output_dir / "dataset_info.json"
    report_path = args.output_dir / f"{args.dataset_name}_prepare_report.json"
    latest_report_path = args.output_dir / "prepare_report.json"
    for path in (target_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    raw_rows = list(iter_jsonl(args.annotation_file))
    gold_map = load_gold_map(args.private_windows)
    keep_rows, drop_only_rows, report = build_rows(raw_rows, gold_map=gold_map, mode=args.mode)

    rng = random.Random(args.seed)
    if args.mode == "balanced-train":
        sampled_drop_count = min(len(drop_only_rows), len(keep_rows))
        sampled_drop_rows = rng.sample(drop_only_rows, sampled_drop_count)
        selected_rows = keep_rows + sampled_drop_rows
    else:
        sampled_drop_rows = []
        selected_rows = keep_rows
    rng.shuffle(selected_rows)

    write_json_array(target_path, selected_rows)
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        if not isinstance(dataset_info, dict):
            raise ValueError(f"{dataset_info_path} must contain a JSON object")
    else:
        dataset_info = {}
    if args.dataset_name in dataset_info and not args.overwrite:
        raise FileExistsError(f"{args.dataset_name} exists in {dataset_info_path}; pass --overwrite to replace it")
    dataset_info.update(dataset_info_for(args.dataset_name, target_name))
    write_json(dataset_info_path, dataset_info)

    report.update(
        {
            "annotation_file": str(args.annotation_file),
            "private_windows": str(args.private_windows),
            "output_dir": str(args.output_dir),
            "dataset_name": args.dataset_name,
            "mode": args.mode,
            "target_file": str(target_path),
            "seed": args.seed,
            "raw_rows": len(raw_rows),
            "gold_windows": len(gold_map),
            "keep_rows_after_gold_drop_filter": len(keep_rows),
            "drop_only_rows_after_gold_drop_filter": len(drop_only_rows),
            "sampled_drop_only_rows": len(sampled_drop_rows),
            "final_rows": len(selected_rows),
            "final_category_counts": dict(
                Counter((row.get("metadata") or {}).get("source_file_category") for row in selected_rows)
            ),
        }
    )
    write_json(report_path, report)
    write_json(latest_report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
