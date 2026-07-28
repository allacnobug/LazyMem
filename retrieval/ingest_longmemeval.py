"""Embed LongMemEval messages and ingest them into Qdrant with resumable logs.

Each point uses a deterministic UUID derived from the question, source session,
and one-based message position. The payload retains retrieval identifiers,
chronological order, role, timestamp, content, and gold-evidence metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, List

import requests
from tqdm import tqdm


DEFAULT_DATA_PATH = "data/Benchmark/LongMemEval/data/longmemeval_s_cleaned.json"
DEFAULT_COLLECTION = "longmemeval_message_qwen8b"
DEFAULT_KEY_OUTPUT_DIR = "outputs/retrieval/longmemeval/key_output"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_logged_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("message_key") or row.get("turn_key")
            if key:
                keys.add(str(key))
    return keys


def make_message_key(question_id: str, source_session_id: str, message_idx_in_session: int) -> str:
    """Stable logical key. The question prefix prevents Qdrant point-id collisions."""
    return f"{question_id}::{source_session_id}_{message_idx_in_session:02d}"


def make_payload_session_id(source_session_id: str, message_idx_in_session: int) -> str:
    """Retrieval id used by Module-1 metrics within each question."""
    return f"{source_session_id}_{message_idx_in_session:02d}"


def make_point_id(message_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, message_key))


def format_message_text(message: dict) -> str:
    role = str((message or {}).get("role", "message") or "message").strip() or "message"
    content = str((message or {}).get("content", "") or "")
    if role == "user":
        return f"User: {content}"
    if role == "assistant":
        return f"Assistant: {content}"
    return f"{role.title()}: {content}"


def iter_message_items(data: List[dict], question_filter: set) -> Iterable[dict]:
    for entry in data:
        question_id = str(entry.get("question_id", ""))
        if question_filter and question_id not in question_filter:
            continue

        global_message_index = 0
        for session_idx, (source_session_id, session, timestamp) in enumerate(
            zip(
                entry.get("haystack_session_ids", []),
                entry.get("haystack_sessions", []),
                entry.get("haystack_dates", []),
            )
        ):
            for message_idx, message in enumerate(session or [], start=1):
                message = message or {}
                message_key = make_message_key(question_id, str(source_session_id), message_idx)
                payload_session_id = make_payload_session_id(str(source_session_id), message_idx)
                raw_text = format_message_text(message)
                yield {
                    "id": make_point_id(message_key),
                    "message_key": message_key,
                    "text": raw_text,
                    "payload": {
                        "node_id": message_key,
                        "message_key": message_key,
                        "session_id": payload_session_id,
                        "source_session_id": str(source_session_id),
                        "session_idx": session_idx,
                        "message_idx_in_session": message_idx,
                        "global_message_index": global_message_index,
                        "question_id": question_id,
                        "timestamp": str(timestamp),
                        "role": str(message.get("role", "")),
                        "has_answer": bool(message.get("has_answer")),
                        "raw_text": raw_text,
                        "data_source": "LongMemEval",
                        "node_type": "message",
                    },
                }
                global_message_index += 1


def get_embeddings_vllm_batch(texts: List[str], args: argparse.Namespace) -> List[List[float]]:
    payload = {"model": args.embed_model, "input": texts}
    headers = {}
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(args.embed_url, json=payload, headers=headers, timeout=args.embed_timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response={response.text}") from exc
    data = response.json().get("data", [])
    embeddings = [item["embedding"] for item in data]
    if len(embeddings) != len(texts):
        raise ValueError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}")
    for emb in embeddings:
        if len(emb) != args.vector_dim:
            raise ValueError(f"Embedding dim mismatch: got {len(emb)}, expected {args.vector_dim}")
    return embeddings


def ensure_collection(client: QdrantClient, args: argparse.Namespace) -> None:
    from qdrant_client.models import Distance, VectorParams

    if args.recreate_collection and client.collection_exists(args.collection):
        client.delete_collection(args.collection)

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=args.vector_dim, distance=Distance.COSINE),
        )
        for field, schema in [
            ("question_id", "keyword"),
            ("session_id", "keyword"),
            ("source_session_id", "keyword"),
            ("role", "keyword"),
            ("node_type", "keyword"),
            ("timestamp", "keyword"),
            ("has_answer", "bool"),
            ("message_idx_in_session", "integer"),
            ("global_message_index", "integer"),
        ]:
            try:
                client.create_payload_index(args.collection, field, schema)
            except Exception:
                pass
        return

    collection_info = client.get_collection(args.collection)
    current_size = collection_info.config.params.vectors.size
    if current_size != args.vector_dim:
        raise ValueError(
            f"Collection '{args.collection}' vector dim is {current_size}, but --vector-dim={args.vector_dim}."
        )


def flush_batch(batch_items: List[dict], client: QdrantClient, args: argparse.Namespace, success_file: Path, failed_file: Path, stats: dict) -> None:
    from qdrant_client.models import PointStruct

    if not batch_items:
        return

    try:
        embeddings = get_embeddings_vllm_batch([item["text"] for item in batch_items], args)
        points = [
            PointStruct(id=item["id"], vector=emb, payload=item["payload"])
            for item, emb in zip(batch_items, embeddings)
        ]
        client.upsert(collection_name=args.collection, points=points)
    except Exception as exc:
        for item in batch_items:
            append_jsonl(
                failed_file,
                {
                    "message_key": item["message_key"],
                    "question_id": item["payload"]["question_id"],
                    "session_id": item["payload"]["session_id"],
                    "timestamp": item["payload"]["timestamp"],
                    "error": str(exc),
                },
            )
            stats["failed"] += 1
        return

    for item in batch_items:
        append_jsonl(
            success_file,
            {
                "message_key": item["message_key"],
                "question_id": item["payload"]["question_id"],
                "session_id": item["payload"]["session_id"],
                "timestamp": item["payload"]["timestamp"],
                "role": item["payload"]["role"],
                "has_answer": item["payload"]["has_answer"],
            },
        )
        stats["success"] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Message-level LongMemEval ingest for final Module-1.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--vector-dim", type=int, default=4096)
    parser.add_argument("--embed-url", default="http://localhost:8005/v1/embeddings")
    parser.add_argument("--embed-model", default="qwen-embedding")
    parser.add_argument("--embed-timeout", type=int, default=120)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--key-output-dir", default=DEFAULT_KEY_OUTPUT_DIR)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--question-ids", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--overwrite-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_output_dir = Path(args.key_output_dir)
    key_output_dir.mkdir(parents=True, exist_ok=True)
    success_file = key_output_dir / "ingest_success.jsonl"
    failed_file = key_output_dir / "ingest_failed.jsonl"
    stats_file = key_output_dir / "ingest_stats.json"

    if args.overwrite_logs:
        for path in (success_file, failed_file, stats_file):
            if path.exists():
                path.unlink()

    question_filter = set()
    if args.question_id:
        question_filter.add(args.question_id.strip())
    if args.question_ids:
        question_filter.update(x.strip() for x in args.question_ids.split(",") if x.strip())

    data = load_json(Path(args.data_path))
    all_items = list(iter_message_items(data, question_filter))
    if args.limit > 0:
        all_items = all_items[: args.limit]

    done = load_logged_keys(success_file)
    items = [item for item in all_items if item["message_key"] not in done]

    from qdrant_client import QdrantClient

    client = QdrantClient(host=args.host, port=args.port, timeout=60)
    ensure_collection(client, args)

    stats = {
        "collection": args.collection,
        "total_items": len(all_items),
        "skipped_success": len(all_items) - len(items),
        "queued": len(items),
        "success": 0,
        "failed": 0,
    }

    batch = []
    pbar = tqdm(total=len(items), desc="Ingest LongMemEval messages", unit="msg")
    for item in items:
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush_batch(batch, client, args, success_file, failed_file, stats)
            pbar.update(len(batch))
            pbar.set_postfix({"ok": stats["success"], "fail": stats["failed"]}, refresh=False)
            batch = []
    if batch:
        flush_batch(batch, client, args, success_file, failed_file, stats)
        pbar.update(len(batch))
    pbar.close()

    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"Done. collection={args.collection} queued={stats['queued']} "
        f"success={stats['success']} failed={stats['failed']} key_output_dir={key_output_dir}"
    )


if __name__ == "__main__":
    main()
