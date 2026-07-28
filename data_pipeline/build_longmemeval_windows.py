"""Build bounded LongMemEval history windows from experiment-15 retrieval results."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "longmemeval_s_cleaned.json"
DEFAULT_RETRIEVAL_FILE = REPO_ROOT / "outputs" / "retrieval" / "longmemeval" / "retrieval_details.jsonl"
DEFAULT_OUT_FILE = REPO_ROOT / "outputs" / "data" / "top50_ctx2_windows.jsonl"
DEFAULT_PRIVATE_FILE = REPO_ROOT / "outputs" / "data" / "top50_ctx2_windows_private.jsonl"
DEFAULT_STATS_FILE = REPO_ROOT / "outputs" / "data" / "top50_ctx2_window_stats.jsonl"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def make_message_id(session_id: str, message_idx_in_session: int) -> str:
    return f"{session_id}_{message_idx_in_session:02d}"


def dedup_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_retrieved_message_ids(row: dict, retrieval_exp_id: Optional[str], top_k: int) -> List[str]:
    """Support current module1 logs and a few simple custom retrieval formats."""
    candidates = None

    # New module1 logs write top_ids_by_exp with --save-topn entries.
    # Older logs used top20_by_exp; module1 currently keeps that key as an alias
    # for backward compatibility, so prefer the explicit non-legacy key first.
    for exp_key in ("top_ids_by_exp", "top20_by_exp", "top50_by_exp"):
        top_by_exp = row.get(exp_key)
        if isinstance(top_by_exp, dict) and top_by_exp:
            if retrieval_exp_id is not None:
                candidates = top_by_exp.get(str(retrieval_exp_id))
            if candidates is None:
                first_key = sorted(top_by_exp.keys())[0]
                candidates = top_by_exp.get(first_key)
            if candidates is not None:
                break

    if candidates is None:
        for key in ("retrieved_top50", "retrieved_top20", "top50", "top20", "retrieved_message_ids", "retrieved_turn_ids", "message_ids", "turn_ids"):
            value = row.get(key)
            if isinstance(value, list):
                candidates = value
                break

    if candidates is None:
        retrieved = row.get("retrieved")
        if isinstance(retrieved, list):
            candidates = [
                item.get("message_id") or item.get("message_key") or item.get("turn_id") or item.get("turn_key") or item.get("id")
                for item in retrieved
                if isinstance(item, dict)
            ]

    if not isinstance(candidates, list):
        return []
    return dedup_keep_order([str(x) for x in candidates if x is not None])[:top_k]


def load_retrieval_map(path: Path, retrieval_exp_id: Optional[str], top_k: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in iter_jsonl(path):
        qid = row.get("question_id")
        if not qid:
            continue
        out[str(qid)] = parse_retrieved_message_ids(row, retrieval_exp_id, top_k)
    return out


def build_dataset_map(data: List[dict]) -> Dict[str, dict]:
    out = {}
    for entry in data:
        qid = entry.get("question_id")
        if qid:
            out[str(qid)] = entry
    return out


def build_message_lookup(entry: dict, one_based_global_idx: bool = False) -> Tuple[Dict[str, dict], List[str]]:
    """Index LongMemEval messages.

    Public window rows never expose session_id/message_id. These identifiers stay in
    private metadata for validation and reconstruction only.
    """
    lookup: Dict[str, dict] = {}
    ordered_keys: List[str] = []
    global_idx = 1 if one_based_global_idx else 0

    for session_idx, (session_id, timestamp, session) in enumerate(
        zip(
            entry.get("haystack_session_ids", []),
            entry.get("haystack_dates", []),
            entry.get("haystack_sessions", []),
        )
    ):
        for message_idx, message in enumerate(session or [], start=1):
            message = message or {}
            key = make_message_id(str(session_id), message_idx)
            lookup[key] = {
                "source_message_key": key,
                "session_id": str(session_id),
                "session_idx": session_idx,
                "message_idx_in_session": message_idx,
                "global_message_index": global_idx,
                "timestamp": str(timestamp),
                "role": str(message.get("role", "")),
                "content": clean_text(message.get("content", "")),
                "has_answer": bool(message.get("has_answer")),
            }
            ordered_keys.append(key)
            global_idx += 1

    return lookup, ordered_keys


def build_key_positions(ordered_keys: List[str]) -> Dict[str, int]:
    return {key: idx for idx, key in enumerate(ordered_keys)}


def public_message(
    info: dict,
    retrieval_rank: Optional[int],
    is_retrieved: bool,
    position: str,
) -> dict:
    return {
        "retrieval_rank": retrieval_rank,
        "timestamp": info["timestamp"],
        "global_message_index": info["global_message_index"],
        "is_retrieved": bool(is_retrieved),
        "position": position,
        "role": info.get("role", ""),
        "content": info.get("content", ""),
    }


def is_next_message(prev_info: dict, cur_info: dict) -> bool:
    return (
        prev_info["session_id"] == cur_info["session_id"]
        and prev_info["message_idx_in_session"] + 1 == cur_info["message_idx_in_session"]
    )


def windows_overlap(prev_info: dict, cur_info: dict, context_window: int) -> bool:
    if prev_info["session_id"] != cur_info["session_id"]:
        return False
    gap = cur_info["message_idx_in_session"] - prev_info["message_idx_in_session"]
    if gap <= 0:
        return False
    # Two local windows [idx-w, idx+w] overlap if their centers are at most 2w+1 apart.
    return gap <= (2 * max(0, context_window) + 1)


def group_contiguous_retrieved_messages(
    retrieved_keys_by_time: List[str],
    lookup: Dict[str, dict],
    context_window: int,
) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []
    for key in retrieved_keys_by_time:
        if not current:
            current = [key]
            continue
        prev = lookup[current[-1]]
        cur = lookup[key]
        if is_next_message(prev, cur) or windows_overlap(prev, cur, context_window):
            current.append(key)
        else:
            groups.append(current)
            current = [key]
    if current:
        groups.append(current)
    return groups


def slice_group(keys: List[str], max_window_messages: int, stride: int) -> List[List[str]]:
    if max_window_messages <= 0 or len(keys) <= max_window_messages:
        return [keys]
    stride = stride if stride > 0 else max(1, max_window_messages - 1)
    windows = []
    start = 0
    while start < len(keys):
        chunk = keys[start : start + max_window_messages]
        if chunk:
            windows.append(chunk)
        if start + max_window_messages >= len(keys):
            break
        start += stride
    return windows


def expand_group_to_dense_span(
    group: List[str],
    ordered_keys: List[str],
    key_positions: Dict[str, int],
    lookup: Dict[str, dict],
) -> List[str]:
    if len(group) <= 1:
        return list(group)
    first_pos = key_positions[group[0]]
    last_pos = key_positions[group[-1]]
    session_id = lookup[group[0]]["session_id"]
    out = []
    for pos in range(first_pos, last_pos + 1):
        key = ordered_keys[pos]
        if lookup[key]["session_id"] != session_id:
            continue
        out.append(key)
    return out


def get_previous_keys(
    first_key: str,
    lookup: Dict[str, dict],
    ordered_keys: List[str],
    key_positions: Dict[str, int],
    context_window: int,
) -> List[str]:
    pos = key_positions.get(first_key)
    if pos is None or pos <= 0 or context_window <= 0:
        return []
    out = []
    for cur_pos in range(pos - 1, max(-1, pos - context_window - 1), -1):
        prev_key = ordered_keys[cur_pos]
        if lookup[prev_key]["session_id"] != lookup[first_key]["session_id"]:
            break
        out.append(prev_key)
    return list(reversed(out))


def get_next_keys(
    last_key: str,
    lookup: Dict[str, dict],
    ordered_keys: List[str],
    key_positions: Dict[str, int],
    context_window: int,
) -> List[str]:
    pos = key_positions.get(last_key)
    if pos is None or pos + 1 >= len(ordered_keys) or context_window <= 0:
        return []
    out = []
    for cur_pos in range(pos + 1, min(len(ordered_keys), pos + context_window + 1)):
        next_key = ordered_keys[cur_pos]
        if lookup[next_key]["session_id"] != lookup[last_key]["session_id"]:
            break
        out.append(next_key)
    return out


def make_window_pair(
    entry: dict,
    lookup: Dict[str, dict],
    retrieval_rank_by_key: Dict[str, int],
    prev_keys: List[str],
    core_keys: List[str],
    next_keys: List[str],
    segment_index: int,
    window_index: int,
    segment_type: str,
    context_window: int,
) -> Tuple[dict, dict]:
    previous_bridge = [
        public_message(
            lookup[key],
            retrieval_rank=retrieval_rank_by_key.get(key),
            is_retrieved=key in retrieval_rank_by_key,
            position="previous_bridge",
        )
        for key in prev_keys
    ]

    core_messages = [
        public_message(
            lookup[key],
            retrieval_rank=retrieval_rank_by_key.get(key),
            is_retrieved=key in retrieval_rank_by_key,
            position="core",
        )
        for key in core_keys
    ]

    next_bridge = [
        public_message(
            lookup[key],
            retrieval_rank=retrieval_rank_by_key.get(key),
            is_retrieved=key in retrieval_rank_by_key,
            position="next_bridge",
        )
        for key in next_keys
    ]

    answer_session_ids = set(entry.get("answer_session_ids", []) or [])
    gold_private_keys = [
        key
        for key, info in lookup.items()
        if info["session_id"] in answer_session_ids and info.get("has_answer")
    ]
    if not gold_private_keys:
        gold_private_keys = [
            key
            for key, info in lookup.items()
            if info["session_id"] in answer_session_ids
        ]

    all_public_turns = previous_bridge + core_messages + next_bridge
    public_global_idxs = {item["global_message_index"] for item in all_public_turns}

    public_row = {
        "question_id": entry.get("question_id", ""),
        "query": entry.get("question", ""),
        "query_timestamp": entry.get("question_date", ""),
        "domain": "dialogue",
        "dataset": "longmemeval",
        "question_type": entry.get("question_type", ""),
        "window": {
            "segment_index": segment_index,
            "window_index": window_index,
            "segment_type": segment_type,
            "context_window": context_window,
            "previous_bridge": previous_bridge,
            "core_messages": core_messages,
            "next_bridge": next_bridge,
        },
    }
    private_row = {
        "question_id": entry.get("question_id", ""),
        "answer": entry.get("answer", ""),
        "segment_index": segment_index,
        "window_index": window_index,
        "segment_type": segment_type,
        "context_window": context_window,
        "core_source_message_keys": core_keys,
        "previous_bridge_source_message_keys": prev_keys,
        "next_bridge_source_message_keys": next_keys,
        "gold_source_message_keys": gold_private_keys,
        "gold_global_message_idxs": [
            lookup[key]["global_message_index"] for key in gold_private_keys if key in lookup
        ],
        "gold_in_window": [
            lookup[key]["global_message_index"]
            for key in gold_private_keys
            if key in lookup and lookup[key]["global_message_index"] in public_global_idxs
        ],
    }
    return public_row, private_row


def build_windows_for_entry(
    entry: dict,
    retrieved_message_ids: List[str],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[dict, dict]], dict]:
    lookup, ordered_keys = build_message_lookup(entry, one_based_global_idx=args.one_based_global_idx)
    key_positions = build_key_positions(ordered_keys)

    retrieval_rank_by_key = {}
    missing_message_ids = []
    for rank, key in enumerate(retrieved_message_ids, start=1):
        if key in lookup:
            retrieval_rank_by_key[key] = rank
        else:
            missing_message_ids.append(key)

    retrieved_keys_by_time = sorted(
        retrieval_rank_by_key.keys(),
        key=lambda key: (lookup[key]["global_message_index"], retrieval_rank_by_key[key]),
    )
    groups = group_contiguous_retrieved_messages(
        retrieved_keys_by_time,
        lookup,
        context_window=args.context_window,
    )

    rows = []
    window_index = 0
    for segment_index, group in enumerate(groups):
        dense_group = expand_group_to_dense_span(group, ordered_keys, key_positions, lookup)
        prev_context = get_previous_keys(
            dense_group[0],
            lookup,
            ordered_keys,
            key_positions,
            context_window=args.context_window,
        )
        next_context = get_next_keys(
            dense_group[-1],
            lookup,
            ordered_keys,
            key_positions,
            context_window=args.context_window,
        )
        expanded_group = prev_context + dense_group + next_context
        dense_set = set(dense_group)
        prev_set = set(prev_context)
        next_set = set(next_context)
        base_segment_type = "continuous" if len(group) > 1 else "singleton"
        for window_keys in slice_group(
            expanded_group,
            args.max_window_messages,
            args.stride,
        ):
            core_keys = [key for key in window_keys if key in dense_set]
            if not core_keys:
                continue
            prev_keys = [key for key in window_keys if key in prev_set]
            next_keys = [key for key in window_keys if key in next_set]
            segment_type = base_segment_type
            if len(expanded_group) > args.max_window_messages:
                segment_type = "continuous_sliced"
            rows.append(
                make_window_pair(
                    entry=entry,
                    lookup=lookup,
                    retrieval_rank_by_key=retrieval_rank_by_key,
                    prev_keys=prev_keys,
                    core_keys=core_keys,
                    next_keys=next_keys,
                    segment_index=segment_index,
                    window_index=window_index,
                    segment_type=segment_type,
                    context_window=args.context_window,
                )
            )
            window_index += 1

    stats = {
        "question_id": entry.get("question_id", ""),
        "retrieved": len(retrieved_message_ids),
        "matched": len(retrieval_rank_by_key),
        "missing": len(missing_message_ids),
        "missing_message_ids": missing_message_ids[:20],
        "segments": len(groups),
        "windows": len(rows),
        "continuous_segments": sum(1 for g in groups if len(g) > 1),
        "singleton_segments": sum(1 for g in groups if len(g) == 1),
    }
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-leaky local memory windows from LongMemEval top-k retrieval results."
    )
    parser.add_argument("--data-file", type=Path, default=Path(DEFAULT_DATA_FILE))
    parser.add_argument("--retrieval-file", type=Path, default=Path(DEFAULT_RETRIEVAL_FILE))
    parser.add_argument("--retrieval-exp-id", default="15")
    parser.add_argument("--out-file", type=Path, default=Path(DEFAULT_OUT_FILE))
    parser.add_argument(
        "--private-file",
        type=Path,
        default=Path(DEFAULT_PRIVATE_FILE),
    )
    parser.add_argument("--stats-file", type=Path, default=Path(DEFAULT_STATS_FILE))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--max-window-messages",
        "--max-core-turns",
        "--max-core-messages",
        dest="max_window_messages",
        type=int,
        default=8,
        help="Maximum total messages per final sub-window, including bridge context.",
    )
    parser.add_argument("--context-window", type=int, default=2, help="Number of previous and next messages from the same session to include as bridge context.")
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="0 means one-message overlap, i.e. stride=max_window_messages-1.",
    )
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--one-based-global-idx",
        action="store_true",
        help="Use 1-based global_message_index instead of the repo's existing 0-based convention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_window_messages <= 0:
        raise ValueError("--max-window-messages must be positive")
    if args.overwrite:
        for path in (args.out_file, args.private_file, args.stats_file):
            if path.exists():
                path.unlink()

    dataset = load_json(args.data_file)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected list dataset in {args.data_file}")
    data_by_qid = build_dataset_map(dataset)
    retrieval_map = load_retrieval_map(args.retrieval_file, args.retrieval_exp_id, args.top_k)

    qids = [qid for qid in retrieval_map.keys() if qid in data_by_qid]
    if args.question_id:
        qids = [qid for qid in qids if qid == args.question_id]
    if args.limit > 0:
        qids = qids[: args.limit]

    iterator = qids
    if tqdm is not None:
        iterator = tqdm(qids, desc="Building LongMemEval windows", unit="query")

    total_windows = 0
    total_missing = 0
    sample_index = 0
    for qid in iterator:
        rows, stats = build_windows_for_entry(data_by_qid[qid], retrieval_map[qid], args)
        for public_row, private_row in rows:
            public_row = dict(public_row)
            private_row = dict(private_row)
            public_row["sample_index"] = sample_index
            private_row["sample_index"] = sample_index
            append_jsonl(args.out_file, public_row)
            append_jsonl(args.private_file, private_row)
            sample_index += 1
        append_jsonl(args.stats_file, stats)
        total_windows += len(rows)
        total_missing += int(stats["missing"])

    print(
        f"Done. queries={len(qids)} windows={total_windows} missing_retrieved_messages={total_missing} "
        f"out_file={args.out_file} private_file={args.private_file} stats_file={args.stats_file}"
    )


if __name__ == "__main__":
    main()
