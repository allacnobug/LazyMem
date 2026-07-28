"""Convert LoCoMo history windows into test-only VERL inference data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_WINDOWS = DEFAULT_DATA_DIR / "processed" / "locomo_top50_ctx2_windows.jsonl"
DEFAULT_PRIVATE = DEFAULT_DATA_DIR / "processed" / "locomo_top50_ctx2_windows_private.jsonl"
DEFAULT_LOCOMO_FILE = DEFAULT_DATA_DIR / "locomo10.json"
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "data" / "locomo_test"

SESSION_RE = re.compile(r"^session_(\d+)$")


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def int_list(value: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def sorted_session_keys(conversation: dict[str, Any]) -> list[str]:
    session_keys: list[tuple[int, str]] = []
    for key, value in (conversation or {}).items():
        match = SESSION_RE.match(str(key))
        if match and isinstance(value, list):
            session_keys.append((int(match.group(1)), str(key)))
    return [key for _, key in sorted(session_keys)]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def format_message_content(message: dict[str, Any]) -> str:
    speaker = clean_text(message.get("speaker", ""))
    text = clean_text(message.get("text", ""))
    parts: list[str] = []
    if speaker:
        parts.append(f"{speaker}: {text}")
    else:
        parts.append(text)

    caption = clean_text(message.get("blip_caption", ""))
    if caption:
        parts.append(f"Image caption: {caption}")

    image_query = clean_text(message.get("query", ""))
    if image_query:
        parts.append(f"Image query: {image_query}")

    return "\n".join(part for part in parts if part)


def build_locomo_indexes(data: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    qa_by_qid: dict[str, dict[str, Any]] = {}
    message_by_sample: dict[str, dict[str, dict[str, Any]]] = {}

    for sample in data:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            continue

        for qa_idx, qa in enumerate(sample.get("qa", []) or []):
            qid = f"{sample_id}::qa_{qa_idx:03d}"
            qa_by_qid[qid] = {
                "question_id": qid,
                "sample_id": sample_id,
                "qa_idx": qa_idx,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "category": qa.get("category", ""),
                "evidence": qa.get("evidence", []) or [],
            }

        conversation = sample.get("conversation") or {}
        sample_messages: dict[str, dict[str, Any]] = {}
        global_idx = 0
        for session_idx, source_session_id in enumerate(sorted_session_keys(conversation)):
            session = conversation.get(source_session_id) or []
            timestamp = str(conversation.get(f"{source_session_id}_date_time", "") or "")
            for message_idx, message in enumerate(session, start=1):
                if not isinstance(message, dict):
                    message = {}
                dia_id = str(message.get("dia_id") or f"{source_session_id}:{message_idx}")
                sample_messages[dia_id] = {
                    "dia_id": dia_id,
                    "source_session_id": source_session_id,
                    "session_idx": session_idx,
                    "message_idx_in_session": message_idx,
                    "global_message_index": global_idx,
                    "timestamp": timestamp,
                    "speaker": str(message.get("speaker", "") or ""),
                    "content": format_message_content(message),
                }
                global_idx += 1
        message_by_sample[sample_id] = sample_messages

    return qa_by_qid, message_by_sample


def compact_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
        return None
    retrieval_rank = message.get("retrieval_rank")
    return {
        "position": message.get("position"),
        "speaker": message.get("speaker"),
        "timestamp": message.get("timestamp"),
        "global_message_index": message.get("global_message_index"),
        "retrieval_rank": retrieval_rank if isinstance(retrieval_rank, int) else None,
        "content": content,
    }


def model_window(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_window = record.get("window") if isinstance(record.get("window"), dict) else {}
    rendered: dict[str, list[dict[str, Any]]] = {}
    for section in ("previous_bridge", "core_messages", "next_bridge"):
        rendered[section] = []
        for message in raw_window.get(section) or []:
            if not isinstance(message, dict):
                continue
            item = compact_message(message)
            if item is not None:
                rendered[section].append(item)
    return rendered


def source_messages_from_window(window: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for section in ("previous_bridge", "core_messages", "next_bridge"):
        for message in window.get(section) or []:
            idx = message.get("global_message_index")
            if not isinstance(idx, int):
                continue
            messages.append(
                {
                    "retrieval_rank": message.get("retrieval_rank"),
                    "timestamp": message.get("timestamp"),
                    "global_message_index": idx,
                    "position": message.get("position"),
                    "role": message.get("role"),
                    "speaker": message.get("speaker"),
                    "content": message.get("content"),
                }
            )
    return messages


def private_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
    return (str(row.get("question_id")), row.get("segment_index"), row.get("window_index"))


def public_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
    raw_window = row.get("window") if isinstance(row.get("window"), dict) else {}
    return (str(row.get("question_id")), raw_window.get("segment_index"), raw_window.get("window_index"))


def load_private_by_key(path: Path) -> dict[tuple[str, Any, Any], dict[str, Any]]:
    out: dict[tuple[str, Any, Any], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        out[private_key(row)] = row
    return out


def render_user_prompt(template: str, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    query = record.get("query", "")
    query_time = record.get("query_timestamp") or record.get("query_time") or ""
    query_block = f"Query Time: {query_time}\nQuery: {query}"
    window = model_window(record)
    prompt = template.replace("{query_time and query}", query_block)
    prompt = prompt.replace("{window}", json.dumps(window, ensure_ascii=False, indent=2))
    return prompt, window


def evidence_text(
    *,
    sample_id: str,
    gold_source_message_keys: list[Any],
    message_by_sample: dict[str, dict[str, dict[str, Any]]],
) -> str:
    sample_messages = message_by_sample.get(sample_id, {})
    lines: list[str] = []
    for raw_key in gold_source_message_keys:
        dia_id = str(raw_key)
        info = sample_messages.get(dia_id)
        if not info:
            lines.append(f"- [dia_id={dia_id}] <missing source message>")
            continue
        lines.append(
            "- "
            f"[dia_id={dia_id}; global_message_index={info.get('global_message_index')}; "
            f"timestamp={info.get('timestamp')}; speaker={info.get('speaker')}] "
            f"{info.get('content')}"
        )
    return "\n".join(lines)


def placeholder_ground_truth() -> str:
    return ""


def build_rl_row(
    *,
    record: dict[str, Any],
    private: dict[str, Any],
    qa: dict[str, Any],
    sample_index: int,
    rank_in_question: int,
    system_prompt: str,
    user_template: str,
    message_by_sample: dict[str, dict[str, dict[str, Any]]],
    data_source: str,
) -> dict[str, Any]:
    qid = str(record.get("question_id", ""))
    user_prompt, window = render_user_prompt(user_template, record)
    raw_window = record.get("window") if isinstance(record.get("window"), dict) else {}
    source_messages = source_messages_from_window(window)

    answer = stringify(private.get("answer")) or stringify(qa.get("answer"))
    gold_in_window = int_list(private.get("gold_in_window"))
    question_gold = int_list(private.get("gold_global_message_idxs"))
    gold_source_message_keys = private.get("gold_source_message_keys")
    if not isinstance(gold_source_message_keys, list):
        gold_source_message_keys = []
    required_evidence = evidence_text(
        sample_id=str(record.get("sample_id") or private.get("sample_id") or qa.get("sample_id") or ""),
        gold_source_message_keys=gold_source_message_keys,
        message_by_sample=message_by_sample,
    )

    extra_info = {
        "split": "test",
        "index": sample_index,
        "metadata": {
            "question_id": qid,
            "sample_id": record.get("sample_id"),
            "qa_idx": record.get("qa_idx"),
            "sample_index": sample_index,
            "source_window_sample_index": record.get("sample_index"),
            "rank_in_question": rank_in_question,
            "query_timestamp": record.get("query_timestamp") or record.get("query_time"),
            "dataset": record.get("dataset"),
            "domain": record.get("domain"),
            "question_type": record.get("question_type"),
            "category": record.get("category"),
            "segment_index": raw_window.get("segment_index"),
            "window_index": raw_window.get("window_index"),
            "segment_type": raw_window.get("segment_type"),
            "context_window": raw_window.get("context_window"),
        },
        "query": record.get("query") if isinstance(record.get("query"), str) else "",
        "gold_answer": answer,
        "answer": answer,
        "required_evidence": required_evidence,
        "gold_global_message_idxs": gold_in_window,
        "question_gold_global_message_idxs": question_gold,
        "gold_source_message_keys": [str(key) for key in gold_source_message_keys],
        "evidence": private.get("evidence") if isinstance(private.get("evidence"), list) else qa.get("evidence", []),
        "source_messages": source_messages,
        "source_message_count": len(source_messages),
        "reward_version": "active_rl_locomo_window_gold_in_window_v1",
    }

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "ability": "memory_window_editing",
        "reward_model": {
            "style": "active_rl_reward_only",
            "ground_truth": placeholder_ground_truth(),
        },
        "extra_info": extra_info,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_preview(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    if limit <= 0:
        return
    write_jsonl(rows[:limit], path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare test-only active RL data from LoCoMo window JSONL.")
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--private-windows", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--locomo-file", type=Path, default=DEFAULT_LOCOMO_FILE)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--user-prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-source", default="lazymem_active_rl_locomo_window_top50_ctx2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = list(iter_jsonl(args.windows))
    if args.limit > 0:
        windows = windows[: args.limit]
    private_by_key = load_private_by_key(args.private_windows)
    locomo_data = read_json(args.locomo_file)
    if not isinstance(locomo_data, list):
        raise ValueError(f"Expected list dataset in {args.locomo_file}")
    qa_by_qid, message_by_sample = build_locomo_indexes(locomo_data)

    system_prompt_path = args.system_prompt_file or args.prompt_dir / "annotate_window_system.txt"
    user_prompt_path = args.user_prompt_file or args.prompt_dir / "annotate_window_user.txt"
    system_prompt = read_text(system_prompt_path).strip()
    user_template = read_text(user_prompt_path).strip()

    rank_counter: Counter[str] = Counter()
    missing_private = 0
    missing_qa = 0
    rows: list[dict[str, Any]] = []
    for sample_index, record in enumerate(windows):
        qid = str(record.get("question_id", ""))
        rank_in_question = rank_counter[qid]
        rank_counter[qid] += 1

        private = private_by_key.get(public_key(record))
        if private is None:
            missing_private += 1
            private = {}
        qa = qa_by_qid.get(qid)
        if qa is None:
            missing_qa += 1
            qa = {}

        rows.append(
            build_rl_row(
                record=record,
                private=private,
                qa=qa,
                sample_index=sample_index,
                rank_in_question=rank_in_question,
                system_prompt=system_prompt,
                user_template=user_template,
                message_by_sample=message_by_sample,
                data_source=args.data_source,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "test.jsonl"
    parquet_path = args.output_dir / "test.parquet"
    preview_path = args.output_dir / "test.preview.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    write_jsonl(rows, jsonl_path)
    write_preview(rows, preview_path, args.preview_rows)
    write_parquet(rows, parquet_path)

    gold_windows = sum(1 for row in rows if row["extra_info"].get("gold_global_message_idxs"))
    source_counts = Counter(row["extra_info"].get("source_message_count", 0) for row in rows)
    manifest = {
        "windows": str(args.windows),
        "private_windows": str(args.private_windows),
        "locomo_file": str(args.locomo_file),
        "prompt_dir": str(args.prompt_dir),
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "output_dir": str(args.output_dir),
        "data_source": args.data_source,
        "split": "test",
        "rows": len(rows),
        "question_ids": len(rank_counter),
        "gold_windows": gold_windows,
        "missing_private_rows": missing_private,
        "missing_qa_rows": missing_qa,
        "source_message_count_histogram": dict(sorted(source_counts.items())),
        "outputs": {
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "preview": str(preview_path),
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
