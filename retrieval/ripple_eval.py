"""Evaluate LongMemEval retrieval recall after temporal context expansion."""

import argparse
import json
import math
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
DEFAULT_DATA_FILE = "data/Benchmark/LongMemEval/data/longmemeval_s_cleaned.json"
DEFAULT_RESULTS_DIR = "outputs/retrieval/longmemeval/exp15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Message-level temporal ripple re-eval for the LongMemEval retrieval experiment."
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--variant", choices=["include", "exclude", "both"], default="both")
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
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def split_message_id(message_id: str) -> Tuple[str, int]:
    sid, idx = str(message_id).rsplit("_", 1)
    return sid, int(idx)


def make_message_id(sid: str, idx: int) -> str:
    return f"{sid}_{idx:02d}"


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
    """Map question_id -> source_session_id -> valid 1-based message indexes."""
    out: Dict[str, Dict[str, set]] = {}
    for entry in data:
        qid = str(entry.get("question_id"))
        sid_to_idxs: Dict[str, set] = {}
        for sid, session in zip(entry.get("haystack_session_ids", []), entry.get("haystack_sessions", [])):
            sid = str(sid)
            sid_to_idxs[sid] = set(range(1, len(session or []) + 1))
        out[qid] = sid_to_idxs
    return out


def ripple_expand(seeds: List[str], sid_to_idxs: Dict[str, set], window: int) -> List[str]:
    if window <= 0:
        return list(seeds)
    offsets = center_first_offsets(window)
    expanded = []
    for message_id in seeds:
        try:
            sid, idx = split_message_id(message_id)
        except Exception:
            continue
        allowed = sid_to_idxs.get(sid, set())
        for offset in offsets:
            candidate_idx = idx + offset
            if candidate_idx <= 0:
                continue
            if candidate_idx in allowed:
                expanded.append(make_message_id(sid, candidate_idx))
    return dedup_keep_order(expanded)


def evaluate_with_ripple(
    ranked_ids: List[str],
    gold_ids: List[str],
    sid_to_idxs: Dict[str, set],
    window: int,
) -> dict:
    gold_set = set(gold_ids)
    if not gold_set:
        return {}

    out = {}
    for k in RECALL_KS:
        expanded = ripple_expand(ranked_ids[:k], sid_to_idxs, window)
        out[f"recall@{k}"] = float(any(item in gold_set for item in expanded))

    for k in ALL_GOLD_KS:
        expanded = ripple_expand(ranked_ids[:k], sid_to_idxs, window)
        out[f"all_gold_included_rate@{k}"] = float(gold_set.issubset(set(expanded)))

    expanded_rank = ripple_expand(ranked_ids[:50], sid_to_idxs, window)
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


def eval_variant(
    variant: str,
    args: argparse.Namespace,
    windows: List[int],
    universe_by_qid: Dict[str, Dict[str, set]],
    out_dir: Path,
) -> None:
    details_path = Path(args.results_dir) / f"retrieval_details_{variant}_72cases.jsonl"
    if not details_path.exists():
        print(f"[skip] missing: {details_path}")
        return

    rows = load_details(details_path)
    if not rows:
        print(f"[skip] empty: {details_path}")
        return

    variant_out_dir = out_dir / variant
    variant_out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = variant_out_dir / "ripple_metrics_summary.json"
    rows_path = variant_out_dir / "ripple_metrics_rows.jsonl"
    if args.overwrite:
        for path in (summary_path, rows_path):
            if path.exists():
                path.unlink()

    per_window_rows = {window: [] for window in windows}
    iterator = tqdm(rows, desc=f"ripple {variant}", unit="query") if tqdm is not None else rows
    skipped_no_gold = 0
    skipped_no_ranked = 0

    for record in iterator:
        qid = str(record.get("question_id"))
        sid_to_idxs = universe_by_qid.get(qid, {})
        gold_ids = get_gold_ids(record)
        ranked_ids = get_ranked_ids(record, args.exp_id)

        if not gold_ids:
            skipped_no_gold += 1
            continue
        if not ranked_ids:
            skipped_no_ranked += 1
            continue

        for window in windows:
            metrics = evaluate_with_ripple(
                ranked_ids=ranked_ids,
                gold_ids=gold_ids,
                sid_to_idxs=sid_to_idxs,
                window=window,
            )
            if not metrics:
                continue
            per_window_rows[window].append(metrics)
            if args.write_rows:
                append_jsonl(
                    rows_path,
                    {
                        "question_id": qid,
                        "variant": variant,
                        "window": window,
                        "metrics": metrics,
                    },
                )

    summary = {
        "meta": {
            "exp_id": str(args.exp_id),
            "variant": variant,
            "data_file": args.data_file,
            "details_file": str(details_path),
            "windows": windows,
            "ranking_source": "top_ids_by_exp",
            "gold_source": "gold_message_ids",
            "note": "Metrics are recomputed after message-level temporal ripple expansion. Window w adds w previous and w next messages in the same source session for every retrieved seed.",
            "total_detail_rows": len(rows),
            "skipped_no_gold": skipped_no_gold,
            "skipped_no_ranked": skipped_no_ranked,
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
    universe_by_qid = build_message_universe(data)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "ripple_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["include", "exclude"] if args.variant == "both" else [args.variant]
    for variant in variants:
        eval_variant(variant, args, windows, universe_by_qid, out_dir)


if __name__ == "__main__":
    main()
