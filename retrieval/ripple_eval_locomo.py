"""Evaluate LoCoMo retrieval recall after temporal context expansion."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


EXP_ID = "15"
RECALL_KS = (1, 5, 10, 15, 20)
ALL_GOLD_KS = (10, 20, 30, 40, 50)
DEFAULT_WINDOWS = (1, 2, 3)
DEFAULT_DATA_FILE = "data/Benchmark/locomo/data/locomo10.json"
DEFAULT_RESULTS_DIR = "outputs/retrieval/locomo/exp15"
DIA_RE = re.compile(r"^D(\d+):(\d+)$")
SESSION_RE = re.compile(r"^session_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Message-level temporal ripple re-eval for the LoCoMo retrieval experiment."
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--details-file", default=None)
    parser.add_argument("--windows", default="1,2,3")
    parser.add_argument("--exp-id", default=EXP_ID)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--write-rows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    out = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = str(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def sorted_session_keys(conversation: dict) -> List[Tuple[int, str]]:
    session_keys = []
    for key, value in (conversation or {}).items():
        match = SESSION_RE.match(str(key))
        if match and isinstance(value, list):
            session_keys.append((int(match.group(1)), str(key)))
    return sorted(session_keys)


def split_message_id(message_id: str) -> Tuple[str, int]:
    match = DIA_RE.match(str(message_id))
    if not match:
        raise ValueError(f"Not a LoCoMo dia_id: {message_id}")
    return f"D{int(match.group(1))}", int(match.group(2))


def make_message_id(session_code: str, idx: int) -> str:
    return f"{session_code}:{idx}"


def center_first_offsets(window: int) -> List[int]:
    offsets = [0]
    for distance in range(1, window + 1):
        offsets.append(-distance)
        offsets.append(distance)
    return offsets


def dcg(binary_rels: List[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(binary_rels[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: List[str], gold_set: set, k: int) -> float:
    rels = [1 if item in gold_set else 0 for item in ranked_ids]
    best = dcg(sorted(rels, reverse=True), k)
    if best <= 0:
        return 0.0
    return dcg(rels, k) / best


def first_hit_rr(ranked_ids: List[str], gold_set: set) -> float:
    for i, item in enumerate(ranked_ids, start=1):
        if item in gold_set:
            return 1.0 / i
    return 0.0


def build_message_universe(data: List[dict]) -> Dict[str, Dict[str, set]]:
    """Map sample_id -> D-session code -> valid 1-based dia indexes."""
    out: Dict[str, Dict[str, set]] = {}
    for entry in data:
        sample_id = str(entry.get("sample_id"))
        conversation = entry.get("conversation") or {}
        session_to_idxs: Dict[str, set] = {}
        for session_num, session_key in sorted_session_keys(conversation):
            session_code = f"D{session_num}"
            idxs = set()
            for fallback_idx, message in enumerate(conversation.get(session_key) or [], start=1):
                dia_id = str((message or {}).get("dia_id") or "")
                try:
                    dia_session_code, dia_idx = split_message_id(dia_id)
                except Exception:
                    dia_session_code, dia_idx = session_code, fallback_idx
                session_to_idxs.setdefault(dia_session_code, set()).add(dia_idx)
                idxs.add(dia_idx)
            if idxs:
                session_to_idxs.setdefault(session_code, set()).update(idxs)
        out[sample_id] = session_to_idxs
    return out


def ripple_expand(seeds: List[str], session_to_idxs: Dict[str, set], window: int) -> List[str]:
    if window <= 0:
        return list(seeds)
    offsets = center_first_offsets(window)
    expanded = []
    for message_id in seeds:
        try:
            session_code, idx = split_message_id(message_id)
        except Exception:
            continue
        allowed = session_to_idxs.get(session_code, set())
        for offset in offsets:
            candidate_idx = idx + offset
            if candidate_idx <= 0:
                continue
            if candidate_idx in allowed:
                expanded.append(make_message_id(session_code, candidate_idx))
    return dedup_keep_order(expanded)


def evaluate_with_ripple(
    ranked_ids: List[str],
    gold_ids: List[str],
    session_to_idxs: Dict[str, set],
    window: int,
) -> dict:
    gold_set = set(gold_ids)
    if not gold_set:
        return {}

    out = {}
    for k in RECALL_KS:
        expanded = ripple_expand(ranked_ids[:k], session_to_idxs, window)
        out[f"recall@{k}"] = float(any(item in gold_set for item in expanded))

    for k in ALL_GOLD_KS:
        expanded = ripple_expand(ranked_ids[:k], session_to_idxs, window)
        out[f"all_gold_included_rate@{k}"] = float(gold_set.issubset(set(expanded)))

    expanded_rank = ripple_expand(ranked_ids[:50], session_to_idxs, window)
    out["mrr"] = first_hit_rr(expanded_rank, gold_set)
    out["ndcg@10"] = ndcg_at_k(expanded_rank, gold_set, 10)
    return out


def mean_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: float(fmean(row[key] for row in rows)) for key in keys}


def load_details(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_gold_ids(record: dict) -> List[str]:
    return record.get("gold_message_ids") or record.get("gold_turn_ids") or []


def get_ranked_ids(record: dict, exp_id: str) -> List[str]:
    top_by_exp = record.get("top_ids_by_exp") or record.get("top20_by_exp") or {}
    return top_by_exp.get(exp_id) or top_by_exp.get(str(exp_id)) or []


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_locomo(
    args: argparse.Namespace,
    windows: List[int],
    universe_by_sample_id: Dict[str, Dict[str, set]],
    out_dir: Path,
) -> None:
    details_path = Path(args.details_file) if args.details_file else Path(args.results_dir) / "retrieval_details_locomo.jsonl"
    if not details_path.exists():
        print(f"[skip] missing: {details_path}")
        return

    rows = load_details(details_path)
    if not rows:
        print(f"[skip] empty: {details_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ripple_metrics_summary_locomo.json"
    rows_path = out_dir / "ripple_metrics_rows_locomo.jsonl"
    if args.overwrite:
        for path in (summary_path, rows_path):
            if path.exists():
                path.unlink()

    per_window_rows = {window: [] for window in windows}
    iterator = tqdm(rows, desc="ripple locomo", unit="query") if tqdm is not None else rows
    skipped_no_gold = 0
    skipped_no_ranked = 0
    skipped_no_universe = 0

    for record in iterator:
        query_id = str(record.get("query_id") or record.get("question_id"))
        sample_id = str(record.get("sample_id") or query_id.split("::", 1)[0])
        session_to_idxs = universe_by_sample_id.get(sample_id, {})
        gold_ids = get_gold_ids(record)
        ranked_ids = get_ranked_ids(record, args.exp_id)

        if not gold_ids:
            skipped_no_gold += 1
            continue
        if not ranked_ids:
            skipped_no_ranked += 1
            continue
        if not session_to_idxs:
            skipped_no_universe += 1
            continue

        for window in windows:
            metrics = evaluate_with_ripple(
                ranked_ids=ranked_ids,
                gold_ids=gold_ids,
                session_to_idxs=session_to_idxs,
                window=window,
            )
            if not metrics:
                continue
            per_window_rows[window].append(metrics)
            if args.write_rows:
                append_jsonl(
                    rows_path,
                    {
                        "query_id": query_id,
                        "sample_id": sample_id,
                        "window": window,
                        "metrics": metrics,
                    },
                )

    summary = {
        "meta": {
            "exp_id": str(args.exp_id),
            "data_file": args.data_file,
            "details_file": str(details_path),
            "windows": windows,
            "ranking_source": "top_ids_by_exp",
            "gold_source": "gold_message_ids",
            "note": "Metrics are recomputed after LoCoMo dia_id ripple expansion. Window w adds w previous and w next messages in the same D-session for every retrieved seed.",
            "total_detail_rows": len(rows),
            "skipped_no_gold": skipped_no_gold,
            "skipped_no_ranked": skipped_no_ranked,
            "skipped_no_universe": skipped_no_universe,
        },
        "metrics_by_window": {
            f"w{window}": mean_metrics(per_window_rows[window]) for window in windows
        },
        "valid_queries_by_window": {
            f"w{window}": len(per_window_rows[window]) for window in windows
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved: {summary_path}")


def main() -> None:
    args = parse_args()
    windows = parse_csv_ints(args.windows) or list(DEFAULT_WINDOWS)
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    universe_by_sample_id = build_message_universe(data)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "ripple_eval"
    eval_locomo(args, windows, universe_by_sample_id, out_dir)


if __name__ == "__main__":
    main()
