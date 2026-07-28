"""Prepare paper-aligned LazyMem train, validation, and test data for VERL."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_WINDOWS = DEFAULT_DATA_DIR / "processed" / "top50_ctx2_windows.jsonl"
DEFAULT_PRIVATE = DEFAULT_DATA_DIR / "processed" / "top50_ctx2_windows_private.jsonl"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_DIR / "splits"
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_REWARD_DATA_DIR = DEFAULT_DATA_DIR / "reward"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "data" / "rl"


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


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        qid = line.strip()
        if not qid or qid.startswith("#") or qid in seen:
            continue
        ids.append(qid)
        seen.add(qid)
    return ids


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _int_list(value: Any) -> list[int]:
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


def compact_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
        return None
    retrieval_rank = message.get("retrieval_rank")
    return {
        "position": message.get("position"),
        "role": role,
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
    for section_messages in window.values():
        for message in section_messages:
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
                    "content": message.get("content"),
                }
            )
    return messages


def private_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
    return (str(row.get("question_id")), row.get("segment_index"), row.get("window_index"))


def load_private_by_key(path: Path) -> dict[tuple[str, Any, Any], dict[str, Any]]:
    out: dict[tuple[str, Any, Any], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        out[private_key(row)] = row
    return out


def load_reasoning_by_question(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    for row in iter_jsonl(path):
        qid = row.get("question_id")
        if not isinstance(qid, str):
            continue
        out[qid] = {
            "required_evidence": row.get("reasoning_answer") if isinstance(row.get("reasoning_answer"), str) else "",
            "gold_answer": row.get("gold_answer") if isinstance(row.get("gold_answer"), str) else "",
        }
    return out


def render_user_prompt(template: str, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    query = record.get("query", "")
    query_time = record.get("query_timestamp") or record.get("query_time") or ""
    query_block = f"Query Time: {query_time}\nQuery: {query}"
    window = model_window(record)
    prompt = template.replace("{query_time and query}", query_block)
    prompt = prompt.replace("{window}", json.dumps(window, ensure_ascii=False, indent=2))
    return prompt, window


def build_input_item(
    *,
    record: dict[str, Any],
    qid: str,
    sample_index: int,
    rank_in_question: int,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    user_prompt, window = render_user_prompt(user_template, record)
    raw_window = record.get("window") if isinstance(record.get("window"), dict) else {}
    return {
        "sample_index": sample_index,
        "source_window_sample_index": record.get("sample_index"),
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


def placeholder_ground_truth() -> str:
    return ""


def to_rl_row(
    item: dict[str, Any],
    private: dict[str, Any],
    reasoning: dict[str, str],
    split: str,
    index: int,
) -> dict[str, Any]:
    gold_in_window = _int_list(private.get("gold_in_window"))
    question_gold = _int_list(private.get("gold_global_message_idxs"))
    messages = source_messages_from_window(item.get("model_input_window") if isinstance(item.get("model_input_window"), dict) else {})
    extra_info = {
        "split": split,
        "index": index,
        "metadata": {
            "question_id": item.get("question_id"),
            "sample_index": item.get("sample_index"),
            "source_window_sample_index": item.get("source_window_sample_index"),
            "rank_in_question": item.get("rank_in_question"),
            "query_timestamp": item.get("query_timestamp"),
            "dataset": item.get("dataset"),
            "domain": item.get("domain"),
            "question_type": item.get("question_type"),
            "segment_index": item.get("segment_index"),
            "window_index": item.get("window_index"),
            "segment_type": item.get("segment_type"),
            "context_window": item.get("context_window"),
        },
        "query": item.get("query") if isinstance(item.get("query"), str) else "",
        "gold_answer": reasoning.get("gold_answer") or (private.get("answer") if isinstance(private.get("answer"), str) else ""),
        "required_evidence": reasoning.get("required_evidence", ""),
        "gold_global_message_idxs": gold_in_window,
        "question_gold_global_message_idxs": question_gold,
        "source_messages": messages,
        "source_message_count": len(messages),
        "reward_version": "active_rl_window_gold_in_window_v1",
    }
    return {
        "data_source": "lazymem_active_rl_window_top50_ctx2",
        "prompt": item.get("messages"),
        "ability": "memory_window_editing",
        "reward_model": {
            "style": "active_rl_reward_only",
            "ground_truth": placeholder_ground_truth(),
        },
        "extra_info": extra_info,
    }


def load_windows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_write_parquet(rows: list[dict[str, Any]], path: Path) -> tuple[bool, str | None]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        return False, f"missing pyarrow: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return True, None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_preview(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    if limit <= 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows[:limit]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def balance_train_rows(
    rows: list[dict[str, Any]],
    *,
    total_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if total_size <= 0 or total_size % 2:
        raise ValueError("--balanced-train-size must be a positive even integer")
    gold_rows = [
        row
        for row in rows
        if row["extra_info"].get("gold_global_message_idxs")
    ]
    non_gold_rows = [
        row
        for row in rows
        if not row["extra_info"].get("gold_global_message_idxs")
    ]
    half = total_size // 2
    if not gold_rows:
        raise ValueError("cannot balance training data: no gold-bearing windows")
    if len(non_gold_rows) < half:
        raise ValueError(
            "cannot balance training data without replacement: "
            f"need {half} non-gold windows, found {len(non_gold_rows)}"
        )

    rng = random.Random(seed)
    sampled = rng.choices(gold_rows, k=half)
    sampled.extend(rng.sample(non_gold_rows, k=half))
    rng.shuffle(sampled)

    balanced: list[dict[str, Any]] = []
    for index, source_row in enumerate(sampled):
        row = copy.deepcopy(source_row)
        extra_info = row["extra_info"]
        extra_info["source_index"] = extra_info.get("index")
        extra_info["index"] = index
        balanced.append(row)
    return balanced


def prepare_split(
    *,
    split: str,
    split_ids: list[str],
    windows: list[dict[str, Any]],
    private_by_key: dict[tuple[str, Any, Any], dict[str, Any]],
    reasoning_by_question: dict[str, dict[str, str]],
    system_prompt: str,
    user_template: str,
    output_dir: Path,
    preview_rows: int,
    formats: set[str],
    balance_training: bool,
    balanced_train_size: int,
    seed: int,
) -> dict[str, Any]:
    id_set = set(split_ids)
    by_id: dict[str, list[dict[str, Any]]] = {qid: [] for qid in split_ids}
    for record in windows:
        qid = record.get("question_id")
        if qid in id_set:
            by_id[str(qid)].append(record)

    missing_private = 0
    missing_reasoning = 0
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for qid in split_ids:
        for rank_in_question, record in enumerate(by_id.get(qid, [])):
            item = build_input_item(
                record=record,
                qid=qid,
                sample_index=sample_index,
                rank_in_question=rank_in_question,
                system_prompt=system_prompt,
                user_template=user_template,
            )
            private = private_by_key.get(private_key(item))
            if private is None:
                missing_private += 1
                private = {}
            reasoning = reasoning_by_question.get(qid, {})
            if not reasoning.get("required_evidence"):
                missing_reasoning += 1
            rows.append(to_rl_row(item, private, reasoning, split, sample_index))
            sample_index += 1

    unbalanced_rows = len(rows)
    unbalanced_gold_windows = sum(
        1 for row in rows if row["extra_info"].get("gold_global_message_idxs")
    )
    if split == "train" and balance_training:
        rows = balance_train_rows(
            rows,
            total_size=balanced_train_size,
            seed=seed,
        )

    outputs: dict[str, Any] = {}
    if "jsonl" in formats:
        target = output_dir / f"{split}.jsonl"
        write_jsonl(rows, target)
        outputs["jsonl"] = {"target": str(target), "ok": True}
    if "parquet" in formats:
        target = output_dir / f"{split}.parquet"
        ok, error = try_write_parquet(rows, target)
        outputs["parquet"] = {"target": str(target), "ok": ok, "error": error}
    write_jsonl_preview(rows, output_dir / f"{split}.preview.jsonl", preview_rows)

    gold_windows = sum(1 for row in rows if row["extra_info"].get("gold_global_message_idxs"))
    source_counts = Counter(row["extra_info"].get("source_message_count", 0) for row in rows)
    seen_ids = {qid for qid, values in by_id.items() if values}
    return {
        "split": split,
        "ids_file_count": len(split_ids),
        "matched_question_ids": len(seen_ids),
        "missing_question_ids": [qid for qid in split_ids if qid not in seen_ids],
        "rows": len(rows),
        "gold_windows": gold_windows,
        "unbalanced_rows": unbalanced_rows,
        "unbalanced_gold_windows": unbalanced_gold_windows,
        "balanced": split == "train" and balance_training,
        "balance_seed": seed if split == "train" and balance_training else None,
        "missing_private_rows": missing_private,
        "missing_required_evidence_rows": missing_reasoning,
        "source_message_count_histogram": dict(sorted(source_counts.items())),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare active RL verl data from LazyMem window JSONL splits.")
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--private-windows", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--reward-data-dir", type=Path, default=DEFAULT_REWARD_DATA_DIR)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--user-prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--formats", nargs="+", choices=["parquet", "jsonl"], default=["parquet", "jsonl"])
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help=(
            "Balance the train split to equal gold-bearing and non-gold windows. "
            "Gold rows are sampled with replacement; non-gold rows without replacement."
        ),
    )
    parser.add_argument("--balanced-train-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    windows = load_windows(args.windows)
    private_by_key = load_private_by_key(args.private_windows)
    system_prompt_path = args.system_prompt_file or args.prompt_dir / "annotate_window_system.txt"
    user_prompt_path = args.user_prompt_file or args.prompt_dir / "annotate_window_user.txt"
    system_prompt = read_text(system_prompt_path).strip()
    user_template = read_text(user_prompt_path).strip()

    reports = []
    for split in args.splits:
        ids_path = args.split_dir / f"{split}.txt"
        split_ids = read_ids(ids_path)
        reasoning_by_question = load_reasoning_by_question(args.reward_data_dir / split / "reasoning_qwen32b.split.jsonl")
        reports.append(
            prepare_split(
                split=split,
                split_ids=split_ids,
                windows=windows,
                private_by_key=private_by_key,
                reasoning_by_question=reasoning_by_question,
                system_prompt=system_prompt,
                user_template=user_template,
                output_dir=args.output_dir,
                preview_rows=args.preview_rows,
                formats=set(args.formats),
                balance_training=args.balance_train,
                balanced_train_size=args.balanced_train_size,
                seed=args.seed,
            )
        )

    manifest = {
        "windows": str(args.windows),
        "private_windows": str(args.private_windows),
        "split_dir": str(args.split_dir),
        "prompt_dir": str(args.prompt_dir),
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "reward_data_dir": str(args.reward_data_dir),
        "output_dir": str(args.output_dir),
        "formats": args.formats,
        "balance_train": args.balance_train,
        "balanced_train_size": args.balanced_train_size,
        "seed": args.seed,
        "total_windows": len(windows),
        "private_rows": len(private_by_key),
        "reports": reports,
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
