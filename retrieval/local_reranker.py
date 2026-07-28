"""Shared local CrossEncoder reranking utilities for experiment 15."""

from __future__ import annotations

import threading
from typing import Any


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def get_cross_encoder(model_name: str, device: str) -> Any:
    """Load one process-wide CrossEncoder instance per model and device."""

    key = (model_name, device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {} if device == "auto" else {"device": device}
            model = CrossEncoder(model_name, **kwargs)
            _MODEL_CACHE[key] = model
    return model


def local_cross_encoder_rerank(
    *,
    query: str,
    ranked_ids: list[str],
    documents_by_id: dict[str, dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
    candidate_count: int,
) -> list[str]:
    """Rerank the candidate head locally while preserving the original tail."""

    if not ranked_ids:
        return []
    head_size = max(1, candidate_count)
    head = ranked_ids[:head_size]
    tail = ranked_ids[head_size:]
    pairs = [
        [query, str(documents_by_id.get(item_id, {}).get("raw_text") or "")]
        for item_id in head
    ]
    model = get_cross_encoder(model_name, device)
    with _INFERENCE_LOCK:
        raw_scores = model.predict(
            pairs,
            batch_size=max(1, batch_size),
            show_progress_bar=False,
        )
    scores = [float(score) for score in raw_scores]
    if len(scores) != len(head):
        raise ValueError(
            f"CrossEncoder returned {len(scores)} scores for {len(head)} candidates"
        )
    ordered_indices = sorted(range(len(head)), key=lambda index: (-scores[index], index))
    return [head[index] for index in ordered_indices] + tail
