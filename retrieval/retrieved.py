"""Run experiment-15 hybrid retrieval and reranking for LongMemEval."""

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional

import requests

try:
    from .local_reranker import local_cross_encoder_rerank
except ImportError:  # Support direct execution as python retrieval/retrieved.py.
    from local_reranker import local_cross_encoder_rerank

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
_THREAD_LOCAL = threading.local()

EXP_ID = "15"
EXP_NAME = "CE-Hybrid-Raw"
DEFAULT_DATA_FILE = "data/Benchmark/LongMemEval/data/longmemeval_s_cleaned.json"
DEFAULT_COLLECTION = "longmemeval_message_qwen8b"
DEFAULT_KEY_OUTPUT_DIR = "outputs/retrieval/longmemeval/exp15"


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def make_message_id(source_session_id: str, message_idx: int) -> str:
    return f"{source_session_id}_{message_idx:02d}"


def build_gold_message_ids(entry: dict, include_fallback_answer_session_messages: bool) -> List[str]:
    answer_sid_set = set(entry.get("answer_session_ids", []) or [])
    gold_ids = []
    for sid, sess in zip(entry.get("haystack_session_ids", []), entry.get("haystack_sessions", [])):
        if sid not in answer_sid_set:
            continue
        for message_idx, message in enumerate(sess or [], start=1):
            if bool((message or {}).get("has_answer", False)):
                gold_ids.append(make_message_id(str(sid), message_idx))

    if not gold_ids and include_fallback_answer_session_messages:
        for sid, sess in zip(entry.get("haystack_session_ids", []), entry.get("haystack_sessions", [])):
            if sid not in answer_sid_set:
                continue
            for message_idx, _message in enumerate(sess or [], start=1):
                gold_ids.append(make_message_id(str(sid), message_idx))
    return dedup_keep_order(gold_ids)


def dcg(binary_rels: List[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(binary_rels[:k]):
        total += rel / math.log2(i + 2)
    return total


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


def evaluate(ranked_ids: List[str], gold_ids: List[str], ks=(1, 5, 10, 15, 20)) -> dict:
    gold_set = set(gold_ids)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(any(g in set(ranked_ids[:k]) for g in gold_set))
    for k in (10, 20, 30, 40, 50):
        out[f"all_gold_included_rate@{k}"] = float(gold_set.issubset(set(ranked_ids[:k]))) if gold_set else 0.0
    out["mrr"] = first_hit_rr(ranked_ids, gold_set)
    out["ndcg@10"] = ndcg_at_k(ranked_ids, gold_set, 10)
    return out


def mean_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: float(fmean(row[key] for row in rows)) for key in keys}


def get_thread_qdrant_client(args):
    if getattr(_THREAD_LOCAL, "qdrant_client", None) is None:
        from qdrant_client import QdrantClient

        _THREAD_LOCAL.qdrant_client = QdrantClient(host=args.host, port=args.port, timeout=60)
    return _THREAD_LOCAL.qdrant_client


def get_query_embedding(args, query: str) -> List[float]:
    payload = {"model": args.embed_model, "input": [query]}
    headers = {}
    api_key = args.embed_api_key or os.environ.get(args.embed_api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(args.embed_url, json=payload, headers=headers, timeout=args.embed_timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response={response.text}") from exc
    data = response.json().get("data", [])
    if not data:
        raise ValueError("Embedding API returned empty data")
    return data[0]["embedding"]


def iter_all_points(client: Any, collection: str, batch_size: int = 512):
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break
        yield points
        if next_offset is None:
            break
        offset = next_offset


def build_qdrant_index(args) -> Dict[str, List[dict]]:
    client = get_thread_qdrant_client(args)
    by_qid: Dict[str, List[dict]] = defaultdict(list)
    for batch in iter_all_points(client, args.collection):
        for point in batch:
            payload = point.payload or {}
            qid = payload.get("question_id")
            sid = payload.get("session_id")
            if not qid or not sid:
                continue
            by_qid[str(qid)].append(
                {
                    "point_id": str(point.id),
                    "session_id": str(sid),
                    "raw_text": str(payload.get("raw_text", "")),
                    "role": str(payload.get("role", "")),
                    "timestamp": str(payload.get("timestamp", "")),
                    "has_answer": bool(payload.get("has_answer")),
                }
            )
    return by_qid


def dense_rank_qdrant(args, question_id: str, query_vec: List[float]) -> List[str]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_thread_qdrant_client(args)
    response = client.query_points(
        collection_name=args.collection,
        query=query_vec,
        query_filter=Filter(must=[FieldCondition(key="question_id", match=MatchValue(value=question_id))]),
        with_payload=True,
        with_vectors=False,
        limit=max(args.dense_candidates, args.topk),
    )
    hits = response.points if hasattr(response, "points") else []
    ranked = []
    for hit in hits:
        sid = (hit.payload or {}).get("session_id")
        if sid:
            ranked.append(str(sid))
    return dedup_keep_order(ranked)


def sparse_rank_bm25(query_tokens: List[str], docs: List[dict]) -> List[str]:
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(doc.get("raw_text", "")) for doc in docs]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return dedup_keep_order([docs[i]["session_id"] for i in ranked_idx])


def rrf_fuse(rank_a: List[str], rank_b: List[str], rrf_k: int) -> List[str]:
    scores = defaultdict(float)
    for i, sid in enumerate(rank_a, start=1):
        scores[sid] += 1.0 / (rrf_k + i)
    for i, sid in enumerate(rank_b, start=1):
        scores[sid] += 1.0 / (rrf_k + i)
    pos_a = {sid: i for i, sid in enumerate(rank_a, start=1)}
    pos_b = {sid: i for i, sid in enumerate(rank_b, start=1)}
    return sorted(scores.keys(), key=lambda sid: (-scores[sid], min(pos_a.get(sid, 10**9), pos_b.get(sid, 10**9)), sid))


def api_rerank(args, query_text: str, ranked_ids: List[str], sid_to_doc: Dict[str, dict]) -> List[str]:
    if not ranked_ids:
        return []
    head = ranked_ids[: max(1, args.rerank_candidates)]
    tail = ranked_ids[max(1, args.rerank_candidates) :]
    documents = [sid_to_doc.get(sid, {}).get("raw_text", "") for sid in head]
    api_key = args.rerank_api_key or os.environ.get(args.rerank_api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing rerank API key. Set --rerank-api-key or {args.rerank_api_key_env}.")
    payload = {
        "model": args.rerank_model,
        "query": query_text,
        "documents": documents,
        "return_documents": False,
        "top_n": len(documents),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(args.rerank_url, json=payload, headers=headers, timeout=args.rerank_timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response={response.text}") from exc
    data = response.json()
    results = data.get("results", [])
    seen = set()
    ordered = []
    for result in results:
        idx = result.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(head):
            continue
        sid = head[idx]
        if sid in seen:
            continue
        ordered.append(sid)
        seen.add(sid)
    ordered.extend(sid for sid in head if sid not in seen)
    return ordered + tail


def rerank(args, query_text: str, ranked_ids: List[str], sid_to_doc: Dict[str, dict]) -> List[str]:
    if args.rerank_backend == "local":
        return local_cross_encoder_rerank(
            query=query_text,
            ranked_ids=ranked_ids,
            documents_by_id=sid_to_doc,
            model_name=args.rerank_model,
            device=args.rerank_device,
            batch_size=args.rerank_batch_size,
            candidate_count=args.rerank_candidates,
        )
    return api_rerank(args, query_text, ranked_ids, sid_to_doc)


def process_entry(entry: dict, args, qdrant_by_qid: Dict[str, List[dict]]) -> dict:
    qid = str(entry["question_id"])
    question = str(entry["question"])
    docs = qdrant_by_qid.get(qid, [])
    if not docs:
        return {"status": "skip_no_docs", "question_id": qid}

    dense_raw = dense_rank_qdrant(args, qid, get_query_embedding(args, question))
    sparse_raw = sparse_rank_bm25(tokenize(question), docs)
    hybrid_rank = rrf_fuse(dense_raw, sparse_raw, args.rrf_k)
    sid_to_doc = {doc["session_id"]: doc for doc in docs}
    ranking = rerank(args, question, hybrid_rank, sid_to_doc)

    def make_variant(gold_ids: List[str], variant_name: str) -> dict:
        if not gold_ids:
            return {"status": "skip_no_gold", "question_id": qid}
        metrics = evaluate(ranking, gold_ids)
        rec = {
            "variant": variant_name,
            "question_id": qid,
            "question": question,
            "gold_message_ids": gold_ids,
            "gold_turn_ids": gold_ids,
            "top_ids_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
            "top20_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
            "metrics_by_exp": {EXP_ID: metrics},
        }
        return {"status": "ok", "question_id": qid, "metrics": metrics, "rec": rec}

    return {
        "status": "ok",
        "question_id": qid,
        "exclude_72": make_variant(build_gold_message_ids(entry, False), "exclude_72"),
        "include_72": make_variant(build_gold_message_ids(entry, True), "include_72"),
    }


def write_summary(path: Path, args, variant_name: str, data_len: int, skipped_no_docs: int, skipped_no_gold: int, rows: List[dict]) -> dict:
    summary = {
        "meta": {
            "experiment_id": EXP_ID,
            "experiment_name": EXP_NAME,
            "data_file": args.data_file,
            "collection": args.collection,
            "topk": args.topk,
            "save_topn": args.save_topn,
            "dense_candidates": args.dense_candidates,
            "rerank_candidates": args.rerank_candidates,
            "rerank_backend": args.rerank_backend,
            "rerank_url": args.rerank_url if args.rerank_backend == "api" else None,
            "rerank_model": args.rerank_model,
            "rerank_device": args.rerank_device,
            "rrf_k": args.rrf_k,
            "workers": args.workers,
            "total_queries": data_len,
            "variant": variant_name,
            "valid_queries": len(rows),
            "skipped_no_gold": skipped_no_gold,
            "skipped_no_docs": skipped_no_docs,
        },
        "metrics": {f"exp_{EXP_ID}": mean_metrics(rows)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final Module-1 experiment 15 (CE-Hybrid-Raw) on message-level LongMemEval points."
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--embed-url", default="http://localhost:8005/v1/embeddings")
    parser.add_argument("--embed-model", default="qwen-embedding")
    parser.add_argument("--embed-timeout", type=int, default=120)
    parser.add_argument("--embed-api-key", default=None)
    parser.add_argument("--embed-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--rerank-backend", choices=("local", "api"), default="local")
    parser.add_argument("--rerank-url", default="http://127.0.0.1:8006/v1/rerank")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-device", default="auto")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-timeout", type=int, default=120)
    parser.add_argument("--rerank-api-key", default=None)
    parser.add_argument("--rerank-api-key-env", default="RERANK_API_KEY")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save-topn", type=int, default=50)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--rerank-candidates", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--key-output-dir", default=DEFAULT_KEY_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_output_dir = Path(args.key_output_dir)
    key_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "exclude_details": key_output_dir / "retrieval_details_exclude_72cases.jsonl",
        "include_details": key_output_dir / "retrieval_details_include_72cases.jsonl",
        "exclude_rows": key_output_dir / "metrics_rows_exclude_72cases.jsonl",
        "include_rows": key_output_dir / "metrics_rows_include_72cases.jsonl",
        "exclude_summary": key_output_dir / "metrics_summary_exclude_72cases.json",
        "include_summary": key_output_dir / "metrics_summary_include_72cases.json",
        "run_log": key_output_dir / "retrieved_run_log.jsonl",
    }
    if args.overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.question_id:
        data = [entry for entry in data if entry.get("question_id") == args.question_id]
    if args.limit > 0:
        data = data[: args.limit]

    qdrant_by_qid = build_qdrant_index(args)

    rows = {"exclude_72": [], "include_72": []}
    skipped_no_docs = 0
    skipped_no_gold = {"exclude_72": 0, "include_72": 0}

    iterator = data
    pbar = tqdm(total=len(data), desc=f"Run exp {EXP_ID}", unit="q") if tqdm else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(process_entry, entry, args, qdrant_by_qid)
            for entry in iterator
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("status") == "skip_no_docs":
                skipped_no_docs += 1
                append_jsonl(paths["run_log"], result)
            elif result.get("status") == "ok":
                for variant in ("exclude_72", "include_72"):
                    sub = result[variant]
                    if sub.get("status") == "ok":
                        append_jsonl(paths["exclude_details"] if variant == "exclude_72" else paths["include_details"], sub["rec"])
                        append_jsonl(paths["exclude_rows"] if variant == "exclude_72" else paths["include_rows"], sub["metrics"])
                        rows[variant].append(sub["metrics"])
                    elif sub.get("status") == "skip_no_gold":
                        skipped_no_gold[variant] += 1
                append_jsonl(paths["run_log"], {"status": "ok", "question_id": result.get("question_id")})
            else:
                append_jsonl(paths["run_log"], result)
            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "exclude_ok": len(rows["exclude_72"]),
                        "include_ok": len(rows["include_72"]),
                        "no_docs": skipped_no_docs,
                    },
                    refresh=False,
                )
    if pbar:
        pbar.close()

    write_summary(
        paths["exclude_summary"],
        args,
        "exclude_72",
        len(data),
        skipped_no_docs,
        skipped_no_gold["exclude_72"],
        rows["exclude_72"],
    )
    write_summary(
        paths["include_summary"],
        args,
        "include_72",
        len(data),
        skipped_no_docs,
        skipped_no_gold["include_72"],
        rows["include_72"],
    )

    print(
        f"Done exp {EXP_ID}. exclude_valid={len(rows['exclude_72'])} "
        f"include_valid={len(rows['include_72'])} skipped_no_docs={skipped_no_docs} "
        f"key_output_dir={key_output_dir}"
    )


if __name__ == "__main__":
    main()
