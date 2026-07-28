"""Run LoCoMo retrieval for questions without annotated gold evidence."""

import argparse
import json
from pathlib import Path

try:
    from retrieved_locomo import (
        EXP_ID,
        append_jsonl,
        build_qdrant_index,
        dense_rank_qdrant,
        get_query_embedding,
        iter_qa_items,
        rerank,
        rrf_fuse,
        sparse_rank_bm25,
        tokenize,
    )
except ModuleNotFoundError:
    from retrieved import (
        EXP_ID,
        append_jsonl,
        build_qdrant_index,
        dense_rank_qdrant,
        get_query_embedding,
        iter_qa_items,
        rerank,
        rrf_fuse,
        sparse_rank_bm25,
        tokenize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval only for LoCoMo QA rows without gold evidence.")
    parser.add_argument("--data-file", default="data/Benchmark/locomo/data/locomo10.json")
    parser.add_argument("--collection", default="locomo_message_qwen8b_last2")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--key-output-dir", default="outputs/retrieval/locomo/exp15_no_gold")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.key_output_dir)
    details_path = out_dir / "retrieval_details_locomo.jsonl"
    run_log_path = out_dir / "retrieved_run_log.jsonl"

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    query_id_set = {item.strip() for item in args.query_ids.split(",") if item.strip()}
    queries = [item for item in iter_qa_items(data, set()) if item["query_id"] in query_id_set]
    found = {item["query_id"] for item in queries}
    missing = sorted(query_id_set - found)
    if missing:
        raise ValueError(f"Missing query ids: {missing}")

    existing_details = set()
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    existing_details.add(json.loads(line).get("query_id"))
                except json.JSONDecodeError:
                    pass

    qdrant_by_sample_id = build_qdrant_index(args)
    for entry in queries:
        query_id = entry["query_id"]
        if query_id in existing_details:
            append_jsonl(run_log_path, {"status": "already_in_details", "query_id": query_id, "sample_id": entry["sample_id"]})
            continue

        sample_id = entry["sample_id"]
        question = entry["question"]
        docs = qdrant_by_sample_id.get(sample_id, [])
        if not docs:
            append_jsonl(run_log_path, {"status": "skip_no_docs_no_gold", "query_id": query_id, "sample_id": sample_id})
            continue

        dense_raw = dense_rank_qdrant(args, sample_id, get_query_embedding(args, question))
        sparse_raw = sparse_rank_bm25(tokenize(question), docs)
        hybrid_rank = rrf_fuse(dense_raw, sparse_raw, args.rrf_k)
        sid_to_doc = {doc["session_id"]: doc for doc in docs}
        ranking = rerank(args, question, hybrid_rank, sid_to_doc)

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
            "gold_message_ids": [],
            "gold_turn_ids": [],
            "top_ids_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
            "top20_by_exp": {EXP_ID: ranking[: max(1, args.save_topn)]},
            "metrics_by_exp": {EXP_ID: {}},
            "no_gold": True,
        }
        append_jsonl(details_path, rec)
        log_row = {"status": "ok_no_gold_retrieved", "query_id": query_id, "sample_id": sample_id}
        append_jsonl(run_log_path, log_row)
        print(f"[ok] {query_id} top1={ranking[0] if ranking else 'NONE'}")


if __name__ == "__main__":
    main()
