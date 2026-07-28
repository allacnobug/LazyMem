#!/usr/bin/env python3
"""Build QA prompts from KEEP decisions or gold-memory inputs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = REPO_ROOT / "outputs" / "validation_dumps"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "evaluation" / "qa_prompts"
WINDOW_KEYS = ("previous_bridge", "core_messages", "next_bridge")


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("**/*.jsonl"))


def extract_json_after_marker(text: str, marker: str) -> Any:
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise ValueError(f"missing marker: {marker}")
    start = text.find("{", marker_pos)
    if start < 0:
        raise ValueError(f"missing JSON object after marker: {marker}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[start:])
    return value


def parse_query(input_text: str) -> Dict[str, str]:
    query_marker = "# Query"
    history_marker = "# History Window"
    qpos = input_text.find(query_marker)
    hpos = input_text.find(history_marker)
    if qpos < 0 or hpos < 0 or hpos <= qpos:
        raise ValueError("missing # Query or # History Window section")

    section = input_text[qpos + len(query_marker) : hpos].strip()
    query_time = ""
    query = ""
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("Query Time:"):
            query_time = line.split(":", 1)[1].strip()
        elif line.startswith("Query:"):
            query = line.split(":", 1)[1].strip()
        elif query:
            query += "\n" + line

    if not query:
        raise ValueError("could not parse Query text")
    return {"query_time": query_time, "query": query}


def iter_role_messages(history: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for window_key in WINDOW_KEYS:
        for message in history.get(window_key, []) or []:
            role = str(message.get("role") or message.get("speaker") or "").strip()
            content = str(message.get("content", ""))
            if role and content.strip():
                if "role" not in message:
                    message = {**message, "role": role}
                yield message


def extract_decision_array(output_text: str) -> List[Dict[str, Any]]:
    candidates = []
    search_text = output_text
    if "</think>" in search_text:
        search_text = search_text.rsplit("</think>", 1)[1]

    decoder = json.JSONDecoder()
    for idx, char in enumerate(search_text):
        if char != "[":
            continue
        try:
            value, end = decoder.raw_decode(search_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            candidates.append((idx, end, value))

    if not candidates:
        raise ValueError("could not find a JSON array in output")

    value = candidates[-1][2]
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("parsed output JSON array is not a list of objects")
    return value


def query_key(query_info: Dict[str, Any]) -> str:
    question_id = str(
        query_info.get("question_id") or query_info.get("query_id") or ""
    ).strip()
    if question_id:
        return f"id:{question_id}"
    return f"{query_info.get('query_time', '')}\n{query_info.get('query', '')}"


def normalize_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**18


def render_prompt(query_info: Dict[str, str], memories: List[Dict[str, Any]]) -> str:
    lines = [
        "You are answering a user question using compressed chronological conversation memory.",
        "Use the memory when it is relevant. If the memory is insufficient, say what is missing instead of inventing facts.",
        "",
        "# Query",
    ]
    if query_info.get("query_time"):
        lines.append(f"Query Time: {query_info['query_time']}")
    lines.extend([f"Query: {query_info['query']}", "", "# Memory"])

    if memories:
        for item in memories:
            lines.append(
                "[{timestamp}] {role}: {content}".format(
                    timestamp=item.get("timestamp", ""),
                    role=item.get("role", ""),
                    content=item.get("compressed_content", ""),
                )
            )
    else:
        lines.append("(No kept memory was found.)")

    lines.extend(["", "# Answer"])
    return "\n".join(lines)


def render_prompt_from_memory_text(query_info: Dict[str, str], memory_text: str) -> str:
    lines = [
        "You are answering a user question using chronological conversation memory.",
        "Use the memory when it is relevant. If the memory is insufficient, say what is missing instead of inventing facts.",
        "",
        "# Query",
    ]
    if query_info.get("query_time"):
        lines.append(f"Query Time: {query_info['query_time']}")
    lines.extend([f"Query: {query_info['query']}", "", "# Memory"])

    if memory_text.strip():
        lines.append(memory_text.strip())
    else:
        lines.append("(No gold memory was found.)")

    lines.extend(["", "# Answer"])
    return "\n".join(lines)


def public_memory_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "global_message_index": item.get("global_message_index"),
        "role": item.get("role", ""),
        "timestamp": item.get("timestamp", ""),
        "compressed_content": item.get("compressed_content", ""),
    }


def memory_identity(
    message: Dict[str, Any],
    compressed: str,
    dedup_mode: str,
    source_file: str,
    line_number: int,
) -> Tuple[str, ...]:
    if dedup_mode == "none":
        return (
            source_file,
            str(line_number),
            str(message.get("global_message_index", "")),
            str(message.get("role", "")),
            str(message.get("timestamp", "")),
            compressed,
        )
    if dedup_mode == "exact":
        return (
            str(message.get("global_message_index", "")),
            str(message.get("role", "")),
            str(message.get("timestamp", "")),
            compressed,
        )
    return (
        str(message.get("global_message_index", "")),
        str(message.get("role", "")),
        str(message.get("timestamp", "")),
    )


def build_prompts(
    validation_root: Path,
    dedup_mode: str,
    include_source_metadata: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    parse_errors: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "validation_root": str(validation_root),
        "files": [],
        "total_records": 0,
        "kept_messages": 0,
        "duplicate_kept_messages": 0,
        "parse_errors": 0,
        "dedup_mode": dedup_mode,
        "include_source_metadata": include_source_metadata,
    }

    for path in iter_jsonl_files(validation_root):
        file_stats = {
            "file": str(path),
            "records": 0,
            "kept_messages": 0,
            "duplicate_kept_messages": 0,
            "parse_errors": 0,
        }
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                stats["total_records"] += 1
                file_stats["records"] += 1
                try:
                    record = json.loads(line)
                    query_info = parse_query(record["input"])
                    question_id = str(
                        record.get("question_id") or record.get("query_id") or ""
                    ).strip()
                    query_info.update(
                        {
                            "question_id": question_id,
                            "query_id": str(
                                record.get("query_id") or question_id
                            ).strip(),
                            "sample_id": record.get("sample_id", ""),
                            "qa_idx": record.get("qa_idx"),
                            "question_type": record.get("question_type", ""),
                            "category": record.get("category"),
                        }
                    )
                    history = extract_json_after_marker(record["input"], "# History Window")
                    messages = list(iter_role_messages(history))
                    decisions = extract_decision_array(record["output"])
                except Exception as exc:
                    stats["parse_errors"] += 1
                    file_stats["parse_errors"] += 1
                    parse_errors.append(
                        {
                            "source_file": str(path),
                            "line_number": line_number,
                            "error": str(exc),
                            "raw_line_prefix": line[:1000],
                        }
                    )
                    continue

                if len(decisions) != len(messages):
                    stats["parse_errors"] += 1
                    file_stats["parse_errors"] += 1
                    parse_errors.append(
                        {
                            "source_file": str(path),
                            "line_number": line_number,
                            "error": (
                                "decision/message length mismatch: "
                                f"decisions={len(decisions)} messages={len(messages)}"
                            ),
                            "query": query_info.get("query", ""),
                            "query_time": query_info.get("query_time", ""),
                            "output_suffix": str(record.get("output", ""))[-1000:],
                        }
                    )
                    continue

                key = query_key(query_info)
                if key not in grouped:
                    grouped[key] = {
                        "query_time": query_info.get("query_time", ""),
                        "query": query_info["query"],
                        "question_id": query_info.get("question_id", ""),
                        "query_id": query_info.get("query_id", ""),
                        "sample_id": query_info.get("sample_id", ""),
                        "qa_idx": query_info.get("qa_idx"),
                        "question_type": query_info.get("question_type", ""),
                        "category": query_info.get("category"),
                        "_memory_by_identity": OrderedDict(),
                        "_source_records": [],
                    }

                grouped[key]["_source_records"].append(
                    {"source_file": str(path), "line_number": line_number}
                )

                for message, decision in zip(messages, decisions):
                    if str(decision.get("op", "")).upper() != "KEEP":
                        continue
                    compressed = str(decision.get("compressed_content", "")).strip()
                    if not compressed:
                        continue
                    identity = memory_identity(
                        message,
                        compressed,
                        dedup_mode,
                        str(path),
                        line_number,
                    )
                    new_item = {
                        "global_message_index": message.get("global_message_index"),
                        "role": message.get("role", ""),
                        "timestamp": message.get("timestamp", ""),
                        "compressed_content": compressed,
                        "source_file": str(path),
                        "line_number": line_number,
                    }
                    old_item = grouped[key]["_memory_by_identity"].get(identity)
                    if old_item is not None:
                        stats["duplicate_kept_messages"] += 1
                        file_stats["duplicate_kept_messages"] += 1
                        if len(compressed) > len(str(old_item.get("compressed_content", ""))):
                            grouped[key]["_memory_by_identity"][identity] = new_item
                        continue
                    grouped[key]["_memory_by_identity"][identity] = new_item
                    stats["kept_messages"] += 1
                    file_stats["kept_messages"] += 1

        stats["files"].append(file_stats)

    prompts: List[Dict[str, Any]] = []
    for group in grouped.values():
        memories = list(group["_memory_by_identity"].values())
        memories.sort(
            key=lambda item: (
                normalize_index(item.get("global_message_index")),
                str(item.get("timestamp", "")),
                str(item.get("role", "")),
            )
        )
        public_memories = [public_memory_item(item) for item in memories]
        prompt_record = {
            "question_id": group["question_id"],
            "query_id": group["query_id"],
            "sample_id": group["sample_id"],
            "qa_idx": group["qa_idx"],
            "query_time": group["query_time"],
            "query": group["query"],
            "question": group["query"],
            "question_type": group["question_type"],
            "category": group["category"],
            "memory": public_memories,
        }
        if include_source_metadata:
            prompt_record["source_records"] = group["_source_records"]
            prompt_record["debug_memory"] = memories
        prompt_record["qa_prompt"] = render_prompt(prompt_record, public_memories)
        prompts.append(prompt_record)

    stats["queries"] = len(prompts)
    return prompts, stats, parse_errors


def build_gold_memory_prompts(
    input_path: Path,
    memory_field: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    prompts: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input": str(input_path),
        "memory_field": memory_field,
        "total_records": 0,
        "queries": 0,
        "parse_errors": 0,
        "missing_memory": 0,
    }

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total_records"] += 1
            try:
                record = json.loads(line)
                query = str(record.get("question") or record.get("query") or "").strip()
                if not query:
                    raise ValueError("missing question/query")
                memory_text = str(record.get(memory_field) or "")
            except Exception as exc:
                stats["parse_errors"] += 1
                parse_errors.append(
                    {
                        "source_file": str(input_path),
                        "line_number": line_number,
                        "error": str(exc),
                        "raw_line_prefix": line[:1000],
                    }
                )
                continue

            if not memory_text.strip():
                stats["missing_memory"] += 1

            query_info = {
                "query_time": str(
                    record.get("query_time")
                    or record.get("question_date")
                    or record.get("timestamp")
                    or ""
                ),
                "query": query,
            }
            prompt_record = {
                "query_id": record.get("query_id") or record.get("question_id") or "",
                "question_id": record.get("question_id") or record.get("query_id") or "",
                "sample_id": record.get("sample_id", ""),
                "qa_idx": record.get("qa_idx"),
                "query_time": query_info["query_time"],
                "query": query_info["query"],
                "question": query_info["query"],
                "answer": record.get("answer", ""),
                "gold_answer": record.get("gold_answer", record.get("answer", "")),
                "category": record.get("category"),
                "question_type": record.get("question_type", ""),
                "evidence_ids": record.get("evidence_ids", []),
                "gold_global_message_idxs": record.get("gold_global_message_idxs", []),
                "memory": memory_text,
                "source_records": [
                    {"source_file": str(input_path), "line_number": line_number}
                ],
            }
            prompt_record["qa_prompt"] = render_prompt_from_memory_text(
                query_info,
                memory_text,
            )
            prompts.append(prompt_record)

    stats["queries"] = len(prompts)
    return prompts, stats, parse_errors


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract KEEP compressed_content from validation outputs, attach "
            "role/timestamp/global_message_index from input messages, group by "
            "query, sort chronologically, and render QA prompts."
        )
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help=f"default: {DEFAULT_VALIDATION_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional summary JSON path. No summary file is written by default.",
    )
    parser.add_argument(
        "--errors-out",
        type=Path,
        default=None,
        help="Optional parse error JSONL path. No error file is written by default.",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=("message", "exact", "none"),
        default="message",
        help=(
            "message: deduplicate by global_message_index/role/timestamp and keep "
            "the longer compression; exact: deduplicate only identical compressed "
            "content; none: keep every KEEP occurrence. default: message"
        ),
    )
    parser.add_argument(
        "--include-source-metadata",
        action="store_true",
        help=(
            "Also write source_records/debug_memory into qa_prompts.jsonl. "
            "Default is off so the file mirrors the model-facing data."
        ),
    )
    parser.add_argument(
        "--gold-input",
        type=Path,
        default=None,
        help=(
            "Optional JSONL whose question/query and gold_memory fields are rendered "
            "directly into <out-dir>/qa_prompts.jsonl."
        ),
    )
    parser.add_argument(
        "--gold-memory-field",
        default="gold_memory",
        help="Memory text field to use in --gold-input mode. default: gold_memory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gold_input:
        prompts, stats, parse_errors = build_gold_memory_prompts(
            args.gold_input,
            args.gold_memory_field,
        )
    else:
        prompts, stats, parse_errors = build_prompts(
            args.validation_root,
            args.dedup_mode,
            args.include_source_metadata,
        )

    output_path = args.out_dir / "qa_prompts.jsonl"
    write_jsonl(output_path, prompts)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            f.write("\n")
    if args.errors_out:
        write_jsonl(args.errors_out, parse_errors)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote: {output_path}")
    if args.summary_out:
        print(f"Wrote: {args.summary_out}")
    if args.errors_out:
        print(f"Wrote: {args.errors_out}")


if __name__ == "__main__":
    main()
