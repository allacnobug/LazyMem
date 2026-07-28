"""Run experiment-15 hybrid retrieval and reranking for LoCoMo."""

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional

import requests

try:
    from .local_reranker import local_cross_encoder_rerank
except ImportError:  # Support direct execution as python retrieval/retrieved_locomo.py.
    from local_reranker import local_cross_encoder_rerank

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
DIA_RE = re.compile(r"D\d+:\d+")
_THREAD_LOCAL = threading.local()

EXP_ID = "15"
EXP_NAME = "CE-Hybrid-Raw"
DEFAULT_DATA_FILE = "data/Benchmark/locomo/data/locomo10.json"
DEFAULT_COLLECTION = "locomo_message_qwen8b"
DEFAULT_KEY_OUTPUT_DIR = "outputs/retrieval/locomo/exp15"


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
        item = str(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_evidence_refs(evidence: list) -> List[str]:
    refs = []
    for item in evidence or []:
        refs.extend(DIA_RE.findall(str(item)))
    return dedup_keep_order(refs)


def iter_qa_items(data: List[dict], sample_filter: set) -> List[dict]:
    out = []
    for entry in data:
        sample_id = str(entry.get("sample_id", ""))
        if sample_filter and sample_id not in sample_filter:
            continue
        for qa_idx, qa in enumerate(entry.get("qa", []) or []):
            question = str(qa.get("question", "") or "")
            query_id = f"{sample_id}::qa_{qa_idx:03d}"
            out.append(
                {
                    "query_id": query_id,
                    "question_id": query_id,
                    "sample_id": sample_id,
                    "qa_idx": qa_idx,
                    "question": question,
                    "answer": qa.get("answer"),
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence", []) or [],
                    "gold_message_ids": parse_evidence_refs(qa.get("evidence", []) or []),
                }
            )
    return out


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
    by_sample_id: Dict[str, List[dict]] = defaultdict(list)
    for batch in iter_all_points(client, args.collection):
        for point in batch:
            payload = point.payload or {}
            sample_id = payload.get("sample_id") or payload.get("question_id")
            sid = payload.get("session_id") or payload.get("dia_id")
            if not sample_id or not sid:
                continue
            by_sample_id[str(sample_id)].append(
                {
                    "point_id": str(point.id),
                    "session_id": str(sid),
                    "dia_id": str(payload.get("dia_id") or sid),
                    "raw_text": str(payload.get("raw_text", "")),
                    "speaker": str(payload.get("speaker", "")),
                    "timestamp": str(payload.get("timestamp", "")),
                    "has_answer": bool(payload.get("has_answer")),
                }
            )
    return by_sample_id


def dense_rank_qdrant(args, sample_id: str, query_vec: List[float]) -> List[str]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_thread_qdrant_client(args)
    response = client.query_points(
        collection_name=args.collection,
        query=query_vec,
        query_filter=Filter(must=[FieldCondition(key="sample_id", match=MatchValue(value=sample_id))]),
        with_payload=True,
        with_vectors=False,
        limit=max(args.dense_candidates, args.topk),
    )
    hits = response.points if hasattr(response, "points") else []
    ranked = []
    for hit in hits:
        sid = (hit.payload or {}).get("session_id") or (hit.payload or {}).get("dia_id")
        if sid:
            ranked.append(str(sid))
    return dedup_keep_order(ranked)


def sparse_rank_bm25(query_tokens: List[str], docs: List[dict]) -> List[str]:
    corpus_tokens = [tokenize(doc.get("raw_text", "")) for doc in docs]
    try:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
    except Exception:
        scores = simple_bm25_scores(query_tokens, corpus_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return dedup_keep_order([docs[i]["session_id"] for i in ranked_idx])


def simple_bm25_scores(query_tokens: List[str], corpus_tokens: List[List[str]]) -> List[float]:
    n_docs = len(corpus_tokens)
    if n_docs == 0:
        return []
    avgdl = sum(len(doc) for doc in corpus_tokens) / n_docs or 1.0
    dfs = Counter()
    for doc in corpus_tokens:
        dfs.update(set(doc))
    q_tokens = [tok for tok in query_tokens if tok]
    k1 = 1.5
    b = 0.75
    scores = []
    for doc in corpus_tokens:
        tf = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for tok in q_tokens:
            df = dfs.get(tok, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf.get(tok, 0)
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * (freq * (k1 + 1.0) / denom) if denom else 0.0
        scores.append(score)
    return scores


def rrf_fuse(rank_a: List[str], rank_b: List[str], rrf_k: int) -> List[str]:
    scores = defaultdict(float)
    for i, sid in enumerate(rank_a, start=1):
        scores[sid] += 1.0 / (rrf_k + i)
    for i, sid in enumerate(rank_b, start=1):
        scores[sid] += 1.0 / (rrf_k + i)
    pos_a = {sid: i for i, sid in enumerate(rank_a, start=1)}
    pos_b = {sid: i for i, sid in enumerate(rank_b, start=1)}
    return sorted(scores.keys(), key=lambda sid: (-scores[sid], min(pos_a.get(sid, 10**9), pos_b.get(sid, 10**9)), sid))


def _retry_delay_seconds(response: Optional[requests.Response], attempt: int, base: float, max_delay: float) -> float:
    retry_after = None
    if response is not None:
        value = (response.headers or {}).get("Retry-After")
        if value is not None:
            try:
                retry_after = float(str(value).strip())
            except ValueError:
                retry_after = None
    delay = retry_after if retry_after is not None else base * (2**attempt)
    return max(0.0, min(max_delay, delay))


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
    response = None
    max_retries = max(0, int(args.rerank_max_retries))
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(args.rerank_url, json=payload, headers=headers, timeout=args.rerank_timeout)
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            status = response.status_code if response is not None else None
            can_retry = status == 429 or (status is not None and 500 <= status < 600)
            if can_retry and attempt < max_retries:
                delay = _retry_delay_seconds(response, attempt, args.rerank_backoff_base, args.rerank_backoff_max)
                if delay > 0:
                    time.sleep(delay)
                continue
            detail = response.text if response is not None else ""
            raise requests.HTTPError(f"{exc}; response={detail}") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt < max_retries:
                delay = min(args.rerank_backoff_max, args.rerank_backoff_base * (2**attempt))
                if delay > 0:
                    time.sleep(delay)
                continue
            raise requests.HTTPError(f"Rerank request failed after retries: {exc}") from exc
    if response is None:
        return ranked_ids
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


def process_entry(entry: dict, args, qdrant_by_sample_id: Dict[str, List[dict]]) -> dict:
    query_id = str(entry["query_id"])
    sample_id = str(entry["sample_id"])
    question = str(entry["question"])
    gold_ids = entry.get("gold_message_ids", [])
    if not gold_ids:
        return {"status": "skip_no_gold", "query_id": query_id, "sample_id": sample_id}

    docs = qdrant_by_sample_id.get(sample_id, [])
    if not docs:
        return {"status": "skip_no_docs", "query_id": query_id, "sample_id": sample_id}

    dense_raw = dense_rank_qdrant(args, sample_id, get_query_embedding(args, question))
    sparse_raw = sparse_rank_bm25(tokenize(question), docs)
    hybrid_rank = rrf_fuse(dense_raw, sparse_raw, args.rrf_k)
    sid_to_doc = {doc["session_id"]: doc for doc in docs}
    ranking = rerank(args, question, hybrid_rank, sid_to_doc)

    metrics = evaluate(ranking, gold_ids)
    rec = {
        "experiment_id": EXP_ID,
        "query_id": query_id,
        "question_id": query_id,
        "sample_id": sample_id,
        "qa_idx": entry.get("qa_idx"),
        "question": question,
        "answer": entry.get("answer"),
        "category": entry.get("category"),
        "evidence": entry.get("evidence", []),
        "gold_message_ids": gold_ids,
        "gold_turn_ids": gold_ids,
        "top_ids_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
        "top20_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
        "metrics_by_exp": {EXP_ID: metrics},
    }
    return {
        "status": "ok",
        "query_id": query_id,
        "sample_id": sample_id,
        "metrics": metrics,
        "rec": rec,
    }


def write_summary(path: Path, args, data_len: int, skipped_no_docs: int, skipped_no_gold: int, failed_queries: int, rows: List[dict]) -> dict:
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
            "rerank_max_retries": args.rerank_max_retries,
            "rerank_backoff_base": args.rerank_backoff_base,
            "rerank_backoff_max": args.rerank_backoff_max,
            "rrf_k": args.rrf_k,
            "workers": args.workers,
            "total_queries": data_len,
            "valid_queries": len(rows),
            "skipped_no_gold": skipped_no_gold,
            "skipped_no_docs": skipped_no_docs,
            "failed_queries": failed_queries,
        },
        "metrics": {f"exp_{EXP_ID}": mean_metrics(rows)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final Module-1 experiment 15 (CE-Hybrid-Raw) on message-level LoCoMo points."
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
    parser.add_argument("--rerank-max-retries", type=int, default=6)
    parser.add_argument("--rerank-backoff-base", type=float, default=2.0)
    parser.add_argument("--rerank-backoff-max", type=float, default=60.0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save-topn", type=int, default=50)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--rerank-candidates", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--question-id", default=None, help="Alias of --query-id for compatibility.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--key-output-dir", default=DEFAULT_KEY_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_output_dir = Path(args.key_output_dir)
    key_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "details": key_output_dir / "retrieval_details_locomo.jsonl",
        "rows": key_output_dir / "metrics_rows_locomo.jsonl",
        "summary": key_output_dir / "metrics_summary_locomo.json",
        "run_log": key_output_dir / "retrieved_run_log.jsonl",
    }
    if args.overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_filter = set()
    if args.sample_id:
        sample_filter.add(args.sample_id.strip())
    if args.sample_ids:
        sample_filter.update(x.strip() for x in args.sample_ids.split(",") if x.strip())

    queries = iter_qa_items(data, sample_filter)
    query_id_filter = args.query_id or args.question_id
    if query_id_filter:
        queries = [entry for entry in queries if entry.get("query_id") == query_id_filter]
    if args.limit > 0:
        queries = queries[: args.limit]

    qdrant_by_sample_id = build_qdrant_index(args)

    rows = []
    skipped_no_docs = 0
    skipped_no_gold = 0
    failed_queries = 0

    pbar = tqdm(total=len(queries), desc=f"Run LoCoMo exp {EXP_ID}", unit="q") if tqdm else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_qid = {
            executor.submit(process_entry, entry, args, qdrant_by_sample_id): str(entry["query_id"])
            for entry in queries
        }
        for future in concurrent.futures.as_completed(future_to_qid):
            qid = future_to_qid[future]
            try:
                result = future.result()
            except Exception as exc:
                failed_queries += 1
                append_jsonl(paths["run_log"], {"status": "error", "query_id": qid, "error": str(exc)})
                if pbar:
                    pbar.update(1)
                continue

            status = result.get("status")
            if status == "skip_no_docs":
                skipped_no_docs += 1
                append_jsonl(paths["run_log"], result)
            elif status == "skip_no_gold":
                skipped_no_gold += 1
                append_jsonl(paths["run_log"], result)
            elif status == "ok":
                append_jsonl(paths["details"], result["rec"])
                append_jsonl(paths["rows"], result["metrics"])
                rows.append(result["metrics"])
                log_row = {"status": "ok", "query_id": result.get("query_id"), "sample_id": result.get("sample_id")}
                append_jsonl(paths["run_log"], log_row)
            else:
                append_jsonl(paths["run_log"], result)

            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "ok": len(rows),
                        "no_gold": skipped_no_gold,
                        "no_docs": skipped_no_docs,
                        "errors": failed_queries,
                    },
                    refresh=False,
                )
    if pbar:
        pbar.close()

    write_summary(paths["summary"], args, len(queries), skipped_no_docs, skipped_no_gold, failed_queries, rows)

    print(
        f"Done LoCoMo exp {EXP_ID}. valid={len(rows)} skipped_no_gold={skipped_no_gold} "
        f"skipped_no_docs={skipped_no_docs} failed_queries={failed_queries} "
        f"key_output_dir={key_output_dir}"
    )


if __name__ == "__main__":
    main()
