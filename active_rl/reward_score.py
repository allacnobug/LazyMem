"""Compute LazyMem action and quality rewards with persistent phase switching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ALLOWED_OPS = {"KEEP", "DROP"}
JUDGE_CACHE: dict[str, dict[str, Any]] = {}
EVAL_METRIC_STATE: dict[str, dict[str, float]] = {}
EVAL_METRIC_FILE_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "size": None, "rows": []}
PHASE2_LATCH_STATE: dict[str, bool] = {}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}

def _merge_component_norm_info(*infos: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for info in infos:
        merged.update(info)
    enabled_values = [
        float(value)
        for key, value in merged.items()
        if key.endswith("_norm_enabled")
    ]
    merged["component_norm_enabled"] = min(enabled_values) if enabled_values else 0.0
    return merged



def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else []
    return []


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _tokens(text: str | None) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _maybe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_json_fragment(fragment: str) -> tuple[dict[str, Any] | None, bool]:
    stripped = (fragment or "").strip()
    if not stripped:
        return None, False
    try:
        obj = json.loads(stripped)
        if isinstance(obj, list):
            return {"local_decisions": obj}, True
    except json.JSONDecodeError:
        pass

    return None, False


def _parse_json(text: str) -> tuple[dict[str, Any] | None, bool, bool]:
    stripped = (text or "").strip()

    think_end = stripped.rfind("</think>")
    if think_end != -1:
        final_text = stripped[think_end + len("</think>") :].strip()
        obj, ok = _parse_json_fragment(final_text)
        if ok:
            return obj, False, True
        return None, False, False

    obj, ok = _parse_json_fragment(stripped)
    if ok:
        return obj, True, True
    return None, False, False


def _fill_decision_locators_from_sources(
    pred_obj: dict[str, Any] | None,
    extra_info: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(pred_obj, dict):
        return pred_obj
    decisions = pred_obj.get("local_decisions")
    if not isinstance(decisions, list):
        return pred_obj
    source_messages = _source_messages_for_judge(extra_info)
    if not source_messages:
        return pred_obj

    filled: list[Any] = []
    changed = False
    for i, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            filled.append(decision)
            continue
        if isinstance(decision.get("global_message_index"), int):
            filled.append(decision)
            continue
        source = source_messages[i] if i < len(source_messages) else None
        if not isinstance(source, dict):
            filled.append(decision)
            continue
        merged = dict(decision)
        for key in ("retrieval_rank", "timestamp", "global_message_index", "position", "role"):
            if key not in merged or merged.get(key) is None:
                merged[key] = source.get(key)
        filled.append(merged)
        changed = True

    if not changed:
        return pred_obj
    out = dict(pred_obj)
    out["local_decisions"] = filled
    return out


def _schema_errors(obj: dict[str, Any] | None, expected_decisions: int | None = None) -> list[str]:
    if obj is None:
        return ["not_object"]
    decisions = obj.get("local_decisions")
    if not isinstance(decisions, list):
        return ["local_decisions_not_list"]

    errors: list[str] = []
    if expected_decisions is not None and len(decisions) != expected_decisions:
        errors.append(f"decision_count_mismatch_expected_{expected_decisions}_got_{len(decisions)}")
    seen_indexes: set[int] = set()
    for i, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"decision_{i}_not_object")
            continue
        if decision.get("op") not in ALLOWED_OPS:
            errors.append(f"decision_{i}_bad_op")
        idx = decision.get("global_message_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            errors.append(f"decision_{i}_bad_global_message_index")
        elif idx in seen_indexes:
            errors.append(f"decision_{i}_duplicate_global_message_index")
        else:
            seen_indexes.add(idx)
        compressed_content = decision.get("compressed_content")
        if not isinstance(compressed_content, str):
            errors.append(f"decision_{i}_bad_compressed_content")
        if decision.get("op") == "DROP" and compressed_content != "":
            errors.append(f"decision_{i}_drop_nonempty_compressed_content")
        if not isinstance(decision.get("reason"), str):
            errors.append(f"decision_{i}_bad_reason")
    return errors


def _format_gate(
    raw_text: str,
    obj: dict[str, Any] | None,
    recovered_ok: bool,
    expected_decisions: int | None = None,
) -> tuple[int, dict[str, Any]]:
    errors = _schema_errors(obj, expected_decisions)
    no_markdown = "```" not in (raw_text or "")
    pass_gate = bool(recovered_ok and not errors and no_markdown)
    return int(pass_gate), {
        "g_format": int(pass_gate),
        "recoverable_json": int(recovered_ok),
        "schema_valid": int(not errors),
        "no_markdown": int(no_markdown),
        "schema_errors": errors[:10],
    }


def _int_set(value: Any) -> set[int]:
    out: set[int] = set()
    for item in _as_list(value):
        if isinstance(item, bool):
            continue
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _source_by_index(extra_info: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for message in _as_list(extra_info.get("source_messages")):
        if not isinstance(message, dict):
            continue
        idx = message.get("global_message_index")
        if isinstance(idx, bool):
            continue
        try:
            out[int(idx)] = message
        except (TypeError, ValueError):
            continue
    return out


def _decision_category(decision: dict[str, Any], gold_idxs: set[int]) -> str:
    idx = decision.get("global_message_index")
    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        idx_int = -1
    return "gold" if idx_int in gold_idxs else "other"


def _action_reward(
    pred_obj: dict[str, Any] | None,
    extra_info: dict[str, Any],
    kwargs: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    del kwargs
    decisions = pred_obj.get("local_decisions") if isinstance(pred_obj, dict) else []
    if not isinstance(decisions, list):
        decisions = []

    gold_idxs = _int_set(extra_info.get("gold_global_message_idxs"))
    source_map = _source_by_index(extra_info)
    decision_map: dict[int, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        idx = decision.get("global_message_index")
        if isinstance(idx, bool):
            continue
        try:
            decision_map[int(idx)] = decision
        except (TypeError, ValueError):
            continue

    action_items: list[dict[str, Any]] = []
    counts = {
        "gold_keep": 0,
        "gold_drop": 0,
        "other_keep": 0,
        "other_drop": 0,
    }

    for idx_int, source in source_map.items():
        decision = decision_map.get(idx_int, {})
        action = str(decision.get("op", ""))
        category = "gold" if idx_int in gold_idxs else "other"
        correct_action = "KEEP" if category == "gold" else "DROP"
        is_correct = action == correct_action
        if category == "gold":
            counts["gold_keep" if is_correct else "gold_drop"] += 1
        else:
            counts["other_drop" if is_correct else "other_keep"] += 1
        source_content = source.get("content")
        compressed_content = decision.get("compressed_content")
        action_items.append(
            {
                "global_message_index": idx_int,
                "action": action,
                "category": category,
                "reward": float(is_correct),
                "is_gold": int(category == "gold"),
                "source_role": source.get("role"),
                "source_content_chars": len(source_content.strip()) if isinstance(source_content, str) else 0,
                "compressed_content_chars": len(compressed_content.strip()) if isinstance(compressed_content, str) else 0,
            }
        )

    gold_count = counts["gold_keep"] + counts["gold_drop"]
    other_count = counts["other_keep"] + counts["other_drop"]
    gold_accuracy = _safe_div(counts["gold_keep"], gold_count)
    non_gold_accuracy = _safe_div(counts["other_drop"], other_count)
    eta = _clip(_float_env("LAZYMEM_ACTIVE_RL_ACTION_ETA", 0.9150326797385621), 0.0, 1.0)

    if gold_count > 0 and other_count > 0:
        score = eta * gold_accuracy + (1.0 - eta) * non_gold_accuracy
    elif gold_count > 0:
        score = gold_accuracy
    elif other_count > 0:
        score = non_gold_accuracy
    else:
        score = 0.0

    return score, {
        "R_action": score,
        "R_gold": gold_accuracy,
        "R_other": non_gold_accuracy,
        "action_eta": eta,
        "gold_accuracy": gold_accuracy if gold_count > 0 else None,
        "non_gold_accuracy": non_gold_accuracy if other_count > 0 else None,
        "action_norm_min": 0.0,
        "action_norm_max": 1.0,
        "gold_count": gold_count,
        "other_count": other_count,
        "action_items": action_items,
        **counts,
    }


def _constructed_memory(pred_obj: dict[str, Any] | None, extra_info: dict[str, Any]) -> str:
    supplied = extra_info.get("constructed_memory")
    if isinstance(supplied, str):
        return supplied

    decisions = pred_obj.get("local_decisions") if isinstance(pred_obj, dict) else []
    parts: list[str] = []
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict) or decision.get("op") != "KEEP":
                continue
            content = decision.get("compressed_content")
            if isinstance(content, str) and content.strip():
                idx = decision.get("global_message_index")
                prefix = f"[{idx}] " if isinstance(idx, int) else ""
                parts.append(prefix + content.strip())
    return "\n".join(parts)


def _extra_text(extra_info: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = extra_info.get(alias)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _query_time(extra_info: dict[str, Any]) -> str:
    query_time = _extra_text(extra_info, ("query_timestamp", "query_time", "question_time"))
    if query_time:
        return query_time
    metadata = extra_info.get("metadata")
    if isinstance(metadata, dict):
        return _extra_text(metadata, ("query_timestamp", "query_time", "question_time"))
    return ""


def _source_messages_for_judge(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in _as_list(extra_info.get("source_messages")):
        if isinstance(message, dict):
            messages.append(message)
    return messages


def _judge_source_message(message: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    content = message.get("content")
    timestamp = message.get("timestamp")
    return {
        "role": role if isinstance(role, str) else "",
        "content": content if isinstance(content, str) else "",
        "timestamp": timestamp if isinstance(timestamp, str) else "",
    }


def _judge_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_disk_cache(path: str | None) -> None:
    if not path or JUDGE_CACHE:
        return
    cache_path = Path(path)
    if not cache_path.exists():
        return
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = row.get("key")
                value = row.get("value")
                if isinstance(key, str) and isinstance(value, dict):
                    JUDGE_CACHE[key] = value
    except (OSError, json.JSONDecodeError):
        return


def _append_disk_cache(path: str | None, key: str, value: dict[str, Any]) -> None:
    if not path:
        return
    try:
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        return


def _append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        return


def _maybe_log_judge_io(row: dict[str, Any]) -> None:
    if not _bool_env("LAZYMEM_ACTIVE_RL_JUDGE_SAVE_IO", False):
        return
    path = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_IO_PATH", "outputs/judge_io.jsonl")
    _append_jsonl(path, row)


def _score_level(value: Any) -> int:
    if isinstance(value, bool):
        return 2 if value else 0
    try:
        parsed = int(value)
        if parsed in {0, 1, 2}:
            return parsed
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    mapping = {
        "bad": 0,
        "low": 0,
        "poor": 0,
        "incorrect": 0,
        "no": 0,
        "partial": 1,
        "medium": 1,
        "ok": 1,
        "some": 1,
        "good": 2,
        "high": 2,
        "excellent": 2,
        "yes": 2,
    }
    return mapping.get(text, 0)


def _validate_judge_obj(obj: dict[str, Any]) -> tuple[bool, str]:
    for key in ("faithfulness", "utility"):
        if key not in obj:
            return False, f"missing {key}"
        _score_level(obj.get(key))
    return True, ""


def _parse_judge_json(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("could not recover judge JSON object", stripped, 0)


def _judge_payload(
    *,
    model: str,
    query_time: str,
    query: str,
    gold_answer: str,
    required_evidence: str,
    source_messages: list[dict[str, Any]],
    constructed_memory: str,
    repair_error: str | None = None,
    invalid_output: str | None = None,
) -> dict[str, Any]:
    system = (
        "You are a strict reward judge for memory construction. "
        "Evaluate whether a kept constructed_memory is faithful and useful. "
        "Keep the two dimensions strictly separate: "
        "faithfulness only judges the relationship between constructed_memory and source_messages; "
        "utility judges how helpful constructed_memory is for the gold_answer and referenced_reasoning_chain. "
        "Return exactly one valid JSON object. Do not output Markdown, explanations, or extra text."
    )
    user: dict[str, Any] = {
        "query_time": query_time,
        "query": query,
        "gold_answer": gold_answer,
        "referenced_reasoning_chain": required_evidence,
        "source_messages": source_messages,
        "constructed_memory": constructed_memory,
        "evaluation_rules": [
            "Score faithfulness using only source_messages. Do not use gold_answer or referenced_reasoning_chain for faithfulness.",
            "Each source message may include a timestamp. Treat that timestamp as part of the source evidence when checking dates, chronology, ordering, and time intervals.",
            "Treat query_time as the time when the query was asked. Compare it with source message timestamps when judging temporal relevance, recency, chronology, and knowledge updates.",
            "Faithfulness checks whether constructed_memory contains unsupported, fabricated, contradictory, or over-inferred information relative to source_messages, including their timestamps.",
            "Score utility using query, gold_answer, and referenced_reasoning_chain together.",
            "Utility checks whether constructed_memory helps answer the query, reach or verify gold_answer, or perform the reasoning in referenced_reasoning_chain.",
            "Utility does not require a direct one-hop contribution. Indirectly useful evidence, constraints, background, disambiguation, chronology, entities, or context can receive utility credit if they help the final answer or reasoning chain.",
            "Score faithfulness and utility independently; a faithful but useless memory may have faithfulness=2 and utility=0.",
            "A useful but unfaithful memory must have faithfulness=0; do not raise faithfulness because it matches gold_answer or referenced_reasoning_chain.",
            "If constructed_memory is only generic background or repeats information irrelevant to gold_answer and referenced_reasoning_chain, assign a low utility score.",
            "If constructed_memory preserves key direct or indirect evidence needed by gold_answer or referenced_reasoning_chain, assign a high utility score.",
            "When uncertain, choose the lower score.",
        ],
        "rubric": {
            "faithfulness": {
                "0": "Contains information unsupported by source_messages, fabrication, clear contradiction, or states uncertain information as fact.",
                "1": "Mostly faithful, but has minor ambiguity, mild over-generalization, imprecise wording, or details not explicitly supported by source_messages.",
                "2": "Fully faithful; all key information is explicitly supported by source_messages, with no contradiction, hallucination, or over-inference.",
            },
            "utility": {
                "0": "Not helpful for gold_answer or referenced_reasoning_chain, irrelevant, or misleading for the final answer or reasoning.",
                "1": "Somewhat helpful for gold_answer or referenced_reasoning_chain, including indirect support, useful background, disambiguating context, chronology, entities, constraints, or partially relevant evidence.",
                "2": "Strongly helpful for gold_answer or referenced_reasoning_chain; preserves key direct or indirect evidence, constraints, facts, or context needed to reach, verify, or explain the answer.",
            },
        },
        "required_output": {
            "faithfulness": 0,
            "utility": 0,
            "unsupported_claims": [],
            "useful_evidence": [],
            "reason": "Brief reason in English",
        },
    }
    if repair_error:
        user["retry_instruction"] = "Your previous output was invalid. Re-evaluate and return only one valid JSON object."
        user["previous_error"] = repair_error[:500]
        if invalid_output:
            user["previous_invalid_output"] = invalid_output[:1200]

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": _float_env("LAZYMEM_ACTIVE_RL_JUDGE_TEMPERATURE", 0.0),
        "max_tokens": _int_env("LAZYMEM_ACTIVE_RL_JUDGE_MAX_TOKENS", 1024),
        "response_format": {"type": "json_object"},
    }
    if _bool_env("LAZYMEM_ACTIVE_RL_JUDGE_ENABLE_THINKING", False):
        thinking_budget = _int_env("LAZYMEM_ACTIVE_RL_JUDGE_THINKING_BUDGET", 512)
        payload["enable_thinking"] = True
        if thinking_budget > 0:
            payload["thinking_budget"] = thinking_budget
    return payload


def _post_chat_completion(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("chat completion content is not string")
    return content


def _openai_judge(
    *,
    query_time: str,
    query: str,
    gold_answer: str,
    required_evidence: str,
    source_messages: list[dict[str, Any]],
    constructed_memory: str,
) -> dict[str, Any]:
    base_url = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://localhost:8001/v1"
    api_key = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "empty"
    model = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_MODEL", "qwen-32b")
    timeout = _float_env("LAZYMEM_ACTIVE_RL_JUDGE_TIMEOUT", 60.0)
    retries = _int_env("LAZYMEM_ACTIVE_RL_JUDGE_RETRIES", 2)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_error = ""
    invalid_output = ""
    for attempt in range(retries + 1):
        try:
            payload = _judge_payload(
                model=model,
                query_time=query_time,
                query=query,
                gold_answer=gold_answer,
                required_evidence=required_evidence,
                source_messages=source_messages,
                constructed_memory=constructed_memory,
                repair_error=last_error or None,
                invalid_output=invalid_output or None,
            )
            content = _post_chat_completion(url, headers, payload, timeout)
            obj = _parse_judge_json(content)
            ok, error = _validate_judge_obj(obj)
            if not ok:
                invalid_output = content
                raise ValueError(error)
            judged = {
                "faithfulness": _score_level(obj.get("faithfulness")),
                "utility": _score_level(obj.get("utility")),
                "reason": str(obj.get("reason", ""))[:300],
                "mode": "openai",
                "judge_attempts": attempt + 1,
            }
            _maybe_log_judge_io(
                {
                    "time": time.time(),
                    "status": "ok",
                    "attempt": attempt + 1,
                    "url": url,
                    "model": model,
                    "input": payload.get("messages", []),
                    "request_payload": payload,
                    "response_text": content,
                    "parsed_judge": judged,
                }
            )
            return judged
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _maybe_log_judge_io(
                {
                    "time": time.time(),
                    "status": "error",
                    "attempt": attempt + 1,
                    "url": url,
                    "model": model,
                    "error": last_error,
                    "input": payload.get("messages", []) if "payload" in locals() else None,
                    "request_payload": payload if "payload" in locals() else None,
                    "response_text": invalid_output,
                }
            )
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return {
        "faithfulness": 0,
        "utility": 0,
        "reason": last_error[:300],
        "mode": "openai_error",
        "judge_attempts": retries + 1,
    }


def _heuristic_judge(
    *,
    query: str,
    gold_answer: str,
    required_evidence: str,
    source_messages: list[dict[str, Any]],
    constructed_memory: str,
) -> dict[str, Any]:
    if not constructed_memory.strip():
        return {
            "faithfulness": 0,
            "utility": 0,
            "reason": "empty constructed memory",
            "mode": "heuristic",
        }

    source_text = "\n".join(str(message.get("content", "")) for message in source_messages if isinstance(message, dict))
    memory_tokens = _tokens(constructed_memory)
    source_tokens = _tokens(source_text)
    query_tokens = _tokens(query)
    answer_tokens = _tokens(gold_answer)
    evidence_tokens = _tokens(required_evidence)

    source_overlap = _safe_div(len(memory_tokens & source_tokens), max(1, len(memory_tokens)))
    query_overlap = _safe_div(len(memory_tokens & query_tokens), max(1, len(query_tokens)))
    answer_overlap = _safe_div(len(memory_tokens & answer_tokens), max(1, len(answer_tokens)))
    evidence_overlap = _safe_div(len(memory_tokens & evidence_tokens), max(1, len(evidence_tokens)))

    faithfulness = 2 if source_overlap >= 0.55 or constructed_memory in source_text else 1 if source_overlap >= 0.30 else 0
    utility_signal = max(query_overlap, answer_overlap)
    utility = 2 if utility_signal >= 0.50 else 1 if utility_signal >= 0.15 else 0
    if faithfulness == 0:
        utility = min(utility, 1)

    return {
        "faithfulness": faithfulness,
        "utility": utility,
        "reason": "heuristic token overlap",
        "mode": "heuristic",
        "source_overlap": source_overlap,
        "query_overlap": query_overlap,
        "answer_overlap": answer_overlap,
        "evidence_overlap": evidence_overlap,
    }


def _judge_reward(pred_obj: dict[str, Any] | None, extra_info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    query_time = _query_time(extra_info)
    query = _extra_text(extra_info, ("query", "question", "stage2_question"))
    gold_answer = _extra_text(extra_info, ("gold_answer", "answer", "stage2_gold_answer"))
    required_evidence = _extra_text(
        extra_info,
        (
            "gold_reasoning_chain",
            "required_evidence",
            "gold_required_evidence",
            "stage2_answer_context",
            "gold_reasoning",
        ),
    )
    source_map = _source_by_index(extra_info)
    decisions = pred_obj.get("local_decisions") if isinstance(pred_obj, dict) else []
    weights = {
        "faithfulness": 0.5,
        "utility": 0.5,
    }
    keep_decisions = [
        decision
        for decision in (decisions if isinstance(decisions, list) else [])
        if isinstance(decision, dict) and decision.get("op") == "KEEP"
    ]
    if not keep_decisions:
        return 0.0, {
            "R_judge": 0.0,
            "R_quality": 0.0,
            "judge_weighted_level": 0.0,
            "faithfulness": 0.0,
            "utility": 0.0,
            "weights": weights,
            "keep_judged": 0,
            "judge_items": [],
            "judge_mode_counts": {"no_keep_penalty": 1},
            "constructed_memory_chars": 0,
        }

    cache_file = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_CACHE")
    _load_disk_cache(cache_file)
    mode = os.getenv("LAZYMEM_ACTIVE_RL_JUDGE_MODE", "heuristic").strip().lower()

    item_rewards: list[float] = []
    item_levels: dict[str, list[float]] = {"faithfulness": [], "utility": []}
    item_weighted_levels: list[float] = []
    judge_items: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    cache_hits = 0
    total_memory_chars = 0

    for decision in keep_decisions:
        idx = decision.get("global_message_index")
        idx_int = idx if isinstance(idx, int) else -1
        source_message = source_map.get(idx_int)
        judge_source = _judge_source_message(source_message)
        source_messages = [judge_source] if judge_source is not None else []
        content = decision.get("compressed_content")
        kept_memory = content if isinstance(content, str) else ""
        total_memory_chars += len(kept_memory)

        if not source_messages:
            judged = {
                "faithfulness": 0,
                "utility": 0,
                "reason": "missing source message for KEEP decision",
                "mode": "missing_source",
                "cache_hit": 0,
            }
        else:
            judge_input = {
                "query_time": query_time,
                "query": query,
                "gold_answer": gold_answer,
                "required_evidence": required_evidence,
                "source_messages": source_messages,
                "constructed_memory": kept_memory,
                "global_message_index": idx_int,
            }
            key = _judge_cache_key(judge_input)
            if key in JUDGE_CACHE:
                judged = dict(JUDGE_CACHE[key])
                judged["cache_hit"] = 1
            else:
                if mode in {"openai", "api", "llm"}:
                    judged = _openai_judge(
                        query_time=query_time,
                        query=query,
                        gold_answer=gold_answer,
                        required_evidence=required_evidence,
                        source_messages=source_messages,
                        constructed_memory=kept_memory,
                    )
                elif mode in {"none", "off", "disabled"}:
                    judged = {
                        "faithfulness": 1,
                        "utility": 1,
                        "reason": "judge disabled",
                        "mode": "none",
                    }
                else:
                    judged = _heuristic_judge(
                        query=query,
                        gold_answer=gold_answer,
                        required_evidence=required_evidence,
                        source_messages=source_messages,
                        constructed_memory=kept_memory,
                    )
                judged["cache_hit"] = 0
                JUDGE_CACHE[key] = judged
                _append_disk_cache(cache_file, key, judged)

        faithfulness = _score_level(judged.get("faithfulness"))
        utility = _score_level(judged.get("utility"))
        if faithfulness == 0:
            raw_level = 0.0
            item_reward = 0.0
        else:
            raw_level = 0.5 * faithfulness + 0.5 * utility
            # Judge levels are 0/1/2. Map the averaged level directly to [0, 1].
            item_reward = _clip(raw_level / 2.0, 0.0, 1.0)
        item_rewards.append(item_reward)
        item_weighted_levels.append(raw_level)
        for name in item_levels:
            item_levels[name].append(float(_score_level(judged.get(name))))
        mode_name = str(judged.get("mode", "unknown"))
        mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
        cache_hits += int(judged.get("cache_hit", 0))
        judge_items.append(
            {
                "global_message_index": idx_int,
                "R_judge_item": item_reward,
                "judge_weighted_level": raw_level,
                "faithfulness": faithfulness,
                "utility": utility,
                "compressed_content_chars": len(kept_memory),
                "judge": judged,
            }
        )

    reward = _mean(item_rewards)
    weighted_level_mean = _mean(item_weighted_levels)

    return reward, {
        "R_judge": reward,
        "R_quality": reward,
        "judge_weighted_level": weighted_level_mean,
        "faithfulness": _mean(item_levels["faithfulness"]),
        "utility": _mean(item_levels["utility"]),
        "weights": weights,
        "keep_judged": len(keep_decisions),
        "judge_items": judge_items,
        "judge_mode_counts": mode_counts,
        "judge_cache_hit_rate": _safe_div(cache_hits, len(keep_decisions)),
        "constructed_memory_chars": total_memory_chars,
    }


def _length_reward(pred_obj: dict[str, Any] | None, extra_info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    memory = _constructed_memory(pred_obj, extra_info)
    source_map = _source_by_index(extra_info)
    decisions = pred_obj.get("local_decisions") if isinstance(pred_obj, dict) else []
    keep_count = 0
    original_keep_chars = 0
    compressed_keep_chars = 0
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            if decision.get("op") != "KEEP":
                continue
            keep_count += 1
            compressed_content = decision.get("compressed_content")
            if isinstance(compressed_content, str):
                compressed_keep_chars += len(compressed_content.strip())
            idx = decision.get("global_message_index")
            idx_int = idx if isinstance(idx, int) else None
            source = source_map.get(idx_int) if idx_int is not None else None
            source_text = source.get("content") if isinstance(source, dict) else ""
            if isinstance(source_text, str):
                original_keep_chars += len(source_text.strip())

    max_compression_ratio = _float_env("LAZYMEM_ACTIVE_RL_MAX_COMPRESSION_RATIO", 0.8)
    max_compression_ratio = _clip(max_compression_ratio, 0.0, 1.0)
    allowed_chars = float(original_keep_chars) * max_compression_ratio
    overflow_chars = max(0.0, float(compressed_keep_chars) - allowed_chars)
    full_penalty_span = max(float(original_keep_chars) - allowed_chars, 1.0)
    penalty = _clip(_safe_div(overflow_chars, full_penalty_span), 0.0, 1.0)
    return penalty, {
        "R_length": penalty,
        "constructed_memory_chars": len(memory),
        "original_keep_chars": float(original_keep_chars),
        "compressed_keep_chars": float(compressed_keep_chars),
        "max_compression_ratio": float(max_compression_ratio),
        "allowed_compressed_chars": float(allowed_chars),
        "overflow_chars": float(overflow_chars),
        "keep_count": keep_count,
        "no_penalty_below_ratio": float(max_compression_ratio),
    }


def _step(extra_info: dict[str, Any], kwargs: dict[str, Any]) -> int:
    for source in (kwargs, extra_info, _maybe_json_dict(extra_info.get("metadata"))):
        for key in ("step", "global_step", "trainer_step", "epoch_step"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, bool):
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return 0


def _linear_schedule(name: str, step: int, default_start: float, default_end: float) -> float:
    start = _float_env(f"LAZYMEM_ACTIVE_RL_{name}_START", default_start)
    end = _float_env(f"LAZYMEM_ACTIVE_RL_{name}_END", default_end)
    warmup = _int_env(f"LAZYMEM_ACTIVE_RL_{name}_WARMUP_STEPS", 0)
    if warmup <= 0:
        return end
    ratio = _clip(step / warmup, 0.0, 1.0)
    return start + (end - start) * ratio


def _metric_value(extra_info: dict[str, Any], kwargs: dict[str, Any], name: str) -> float | None:
    env_name = f"LAZYMEM_ACTIVE_RL_METRIC_{name.upper()}"
    raw_env = os.getenv(env_name)
    if raw_env is not None:
        try:
            return float(raw_env)
        except ValueError:
            return None

    keys = (name, name.lower(), name.upper(), f"metric_{name}", f"{name}_metric")
    for source in (kwargs, extra_info, _maybe_json_dict(extra_info.get("metadata"))):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_from_mapping(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float_value(source.get(key))
        if value is not None:
            return value
    for group_key in ("eval", "val", "validation", "metrics", "latest"):
        nested = source.get(group_key)
        if isinstance(nested, dict):
            value = _metric_from_mapping(nested, keys)
            if value is not None:
                return value
    return None


def _metric_from_json_file(path: str | None, keys: tuple[str, ...]) -> float | None:
    if not path:
        return None
    metric_path = Path(path)
    try:
        stat = metric_path.stat()
    except OSError:
        return None

    cache_hit = (
        EVAL_METRIC_FILE_CACHE.get("path") == str(metric_path)
        and EVAL_METRIC_FILE_CACHE.get("mtime_ns") == stat.st_mtime_ns
        and EVAL_METRIC_FILE_CACHE.get("size") == stat.st_size
    )
    if cache_hit:
        candidates = EVAL_METRIC_FILE_CACHE.get("rows")
        if not isinstance(candidates, list):
            candidates = []
    else:
        try:
            text = metric_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

        candidates = []
        if text:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    candidates.append(obj)
                elif isinstance(obj, list):
                    candidates.extend(item for item in obj if isinstance(item, dict))
            except json.JSONDecodeError:
                for line in text.splitlines():
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        candidates.append(obj)
        EVAL_METRIC_FILE_CACHE.update(
            {
                "path": str(metric_path),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "rows": candidates,
            }
        )

    for obj in reversed(candidates):
        value = _metric_from_mapping(obj, keys)
        if value is not None:
            return value
    return None


def _split_name(extra_info: dict[str, Any], kwargs: dict[str, Any]) -> str:
    for source in (kwargs, extra_info, _maybe_json_dict(extra_info.get("metadata"))):
        if not isinstance(source, dict):
            continue
        value = source.get("split")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _is_eval_split(extra_info: dict[str, Any], kwargs: dict[str, Any]) -> bool:
    return _split_name(extra_info, kwargs) in {"val", "valid", "validation", "eval", "test"}


def _eval_metric_value(extra_info: dict[str, Any], kwargs: dict[str, Any], name: str) -> float | None:
    env_name = f"LAZYMEM_ACTIVE_RL_EVAL_METRIC_{name.upper()}"
    value = _float_value(os.getenv(env_name))
    if value is not None:
        return value

    keys = (
        f"eval_{name}",
        f"val_{name}",
        f"validation_{name}",
        f"eval_metric_{name}",
        f"metric_eval_{name}",
        f"eval/{name}",
        f"val/{name}",
    )
    value = _metric_from_json_file(os.getenv("LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH"), keys + (name,))
    if value is not None:
        return value

    for source in (kwargs, extra_info, _maybe_json_dict(extra_info.get("metadata"))):
        if not isinstance(source, dict):
            continue
        value = _metric_from_mapping(source, keys)
        if value is not None:
            return value
    return None


def _phase2_latch_path() -> Path | None:
    configured = os.getenv("LAZYMEM_ACTIVE_RL_PHASE2_LATCH_PATH")
    if configured is not None:
        configured = configured.strip()
        return Path(configured) if configured else None

    metrics_path = os.getenv("LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH")
    if not metrics_path:
        return None
    return Path(f"{metrics_path}.phase2_latched.json")


def _phase2_is_latched(path: Path | None) -> bool:
    key = str(path) if path is not None else "__memory__"
    if PHASE2_LATCH_STATE.get(key, False):
        return True
    if path is not None and path.is_file():
        PHASE2_LATCH_STATE[key] = True
        return True
    return False


def _persist_phase2_latch(path: Path | None, eval_g_format: float, threshold: float) -> None:
    key = str(path) if path is not None else "__memory__"
    PHASE2_LATCH_STATE[key] = True
    if path is None:
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    except OSError:
        return

    row = {
        "time": time.time(),
        "pid": os.getpid(),
        "phase2_latched": True,
        "trigger_eval_g_format": eval_g_format,
        "phase2_format_threshold": threshold,
    }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass


def _maybe_append_eval_metrics(scored: dict[str, Any], extra_info: dict[str, Any], kwargs: dict[str, Any]) -> None:
    if not _bool_env("LAZYMEM_ACTIVE_RL_AUTO_APPEND_EVAL_METRICS", False):
        return
    if not _is_eval_split(extra_info, kwargs):
        return
    path = os.getenv("LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH")
    if not path:
        return

    details = scored.get("details") if isinstance(scored.get("details"), dict) else {}
    action = details.get("action") if isinstance(details.get("action"), dict) else {}
    split = _split_name(extra_info, kwargs) or "eval"
    state = EVAL_METRIC_STATE.setdefault(
        split,
        {
            "samples": 0.0,
            "score_sum": 0.0,
            "g_format_sum": 0.0,
            "gold_keep": 0.0,
            "gold_drop": 0.0,
            "other_keep": 0.0,
            "other_drop": 0.0,
            "gold_count": 0.0,
            "other_count": 0.0,
        },
    )
    state["samples"] += 1.0
    state["score_sum"] += float(scored.get("score", 0.0))
    state["g_format_sum"] += float(scored.get("g_format", 0.0))
    for key in ("gold_keep", "gold_drop", "other_keep", "other_drop", "gold_count", "other_count"):
        state[key] += float(action.get(key, 0.0))

    row = {
        "time": time.time(),
        "pid": os.getpid(),
        "source": "active_rl_auto_eval",
        "split": split,
        "samples": int(state["samples"]),
        "score": _safe_div(state["score_sum"], state["samples"]),
        "eval_g_format": _safe_div(state["g_format_sum"], state["samples"]),
        "g_format": _safe_div(state["g_format_sum"], state["samples"]),
        "eval_gold_keep_rate": _safe_div(state["gold_keep"], state["gold_count"]),
        "eval_gold_drop_rate": _safe_div(state["gold_drop"], state["gold_count"]),
        "eval_other_keep_rate": _safe_div(state["other_keep"], state["other_count"]),
        "eval_other_drop_rate": _safe_div(state["other_drop"], state["other_count"]),
        "gold_count": state["gold_count"],
        "other_count": state["other_count"],
    }
    _append_jsonl(path, row)


def _lambda_weights(step: int, extra_info: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    del step
    configured_lambda_qual = _clip(_float_env("LAZYMEM_ACTIVE_RL_LAMBDA_QUAL", 0.5), 0.0, 1.0)
    eval_g_format = _eval_metric_value(extra_info, kwargs, "g_format")
    threshold = _clip(
        _float_env("LAZYMEM_ACTIVE_RL_PHASE2_FORMAT_THRESHOLD", 0.99),
        0.0,
        1.0,
    )
    latch_path = _phase2_latch_path()
    phase2_enabled = _phase2_is_latched(latch_path)
    phase2_triggered_now = False
    if not phase2_enabled and eval_g_format is not None and eval_g_format > threshold:
        _persist_phase2_latch(latch_path, eval_g_format, threshold)
        phase2_enabled = True
        phase2_triggered_now = True
    lambda_qual = configured_lambda_qual if phase2_enabled else 0.0
    return {
        "configured_lambda_qual": configured_lambda_qual,
        "lambda_qual": lambda_qual,
        "action_weight": 1.0 - lambda_qual,
        "judge_enabled": 1.0 if lambda_qual > 0.0 else 0.0,
        "reward_phase": "judge" if lambda_qual > 0.0 else "action",
        "phase2_enabled": 1.0 if phase2_enabled else 0.0,
        "phase2_latched": 1.0 if phase2_enabled else 0.0,
        "phase2_triggered_now": 1.0 if phase2_triggered_now else 0.0,
        "phase2_format_threshold": threshold,
        "eval_g_format": eval_g_format if eval_g_format is not None else -1.0,
    }


def _skipped_judge_info(reason: str, weights: dict[str, Any]) -> dict[str, Any]:
    return {
        "R_judge": 0.0,
        "R_quality": 0.0,
        "judge_weighted_level": 0.0,
        "faithfulness": 0.0,
        "utility": 0.0,
        "weights": {},
        "keep_judged": 0,
        "judge_items": [],
        "judge_mode_counts": {reason: 1},
        "judge_cache_hit_rate": 0.0,
        "constructed_memory_chars": 0,
        "skip_reason": reason,
        "judge_gate_gold_drop_rate": weights.get("judge_gate_gold_drop_rate", 0.0),
        "judge_gate_threshold": weights.get("judge_gate_threshold", 0.0),
        "judge_gate_source": weights.get("judge_gate_source", ""),
    }


def score_with_details(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str | dict[str, Any] | None = None,
    extra_info: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source, ground_truth
    extra = _maybe_json_dict(extra_info)

    pred_obj, strict_ok, recovered_ok = _parse_json(solution_str)
    pred_obj = _fill_decision_locators_from_sources(pred_obj, extra)
    expected_decisions = len(_source_messages_for_judge(extra))
    g_format, format_info = _format_gate(solution_str, pred_obj, recovered_ok, expected_decisions)
    action, action_info = _action_reward(pred_obj, extra, kwargs)
    length, length_info = _length_reward(pred_obj, extra)
    step = _step(extra, kwargs)
    weights = _lambda_weights(step, extra, kwargs)
    if not g_format:
        judge = 0.0
        judge_info = _skipped_judge_info("skip_invalid_format", weights)
    elif float(weights.get("lambda_qual", 0.0)) > 0.0:
        judge, judge_info = _judge_reward(pred_obj, extra)
    else:
        judge = 0.0
        judge_info = _skipped_judge_info("skip_lambda_qual_zero", weights)

    raw_inside = (
        weights["action_weight"] * action
        + weights["lambda_qual"] * judge
    )
    action_norm_score = _clip(float(action), 0.0, 1.0)
    judge_norm_score = _clip(float(judge), 0.0, 1.0)
    action_norm_info = {
        "action_normalized": action_norm_score,
        "action_norm_score": action_norm_score,
        "action_norm_enabled": 0.0,
        "action_norm_min": 0.0,
        "action_norm_max": 1.0,
        "action_norm_scale": 1.0,
    }
    judge_norm_info = {
        "judge_normalized": judge_norm_score,
        "judge_norm_score": judge_norm_score,
        "judge_norm_enabled": 0.0,
        "judge_norm_min": 0.0,
        "judge_norm_max": 1.0,
        "judge_norm_scale": 1.0,
    }
    component_norm_info = _merge_component_norm_info(action_norm_info, judge_norm_info)
    inside_score = (
        weights["action_weight"] * action_norm_score
        + weights["lambda_qual"] * judge_norm_score
    )
    score = float(g_format) * inside_score
    if not math.isfinite(score):
        score = 0.0

    scored = {
        "score": float(score),
        "g_format": g_format,
        "R_action": float(action),
        "R_judge": float(judge),
        "R_quality": float(judge),
        "R_length": float(length),
        "raw_inside_gate": float(raw_inside),
        "inside_gate_score": float(inside_score),
        "R_action_normalized": float(component_norm_info["action_normalized"]),
        "R_action_norm_score": float(component_norm_info["action_norm_score"]),
        "R_action_norm_min": float(component_norm_info["action_norm_min"]),
        "R_action_norm_max": float(component_norm_info["action_norm_max"]),
        "R_judge_normalized": float(component_norm_info["judge_normalized"]),
        "R_judge_norm_score": float(component_norm_info["judge_norm_score"]),
        "R_judge_norm_min": float(component_norm_info["judge_norm_min"]),
        "R_judge_norm_max": float(component_norm_info["judge_norm_max"]),
        "component_norm_enabled": float(component_norm_info["component_norm_enabled"]),
        "step": step,
        "strict_json": int(strict_ok),
        "details": {
            "format": format_info,
            "action": action_info,
            "judge": judge_info,
            "length": length_info,
            "weights": weights,
            "component_norm": component_norm_info,
        },
    }
    _maybe_append_eval_metrics(scored, extra, kwargs)
    return scored


def _flat_score_dict(scored: dict[str, Any]) -> dict[str, float]:
    details = scored.get("details") if isinstance(scored.get("details"), dict) else {}
    action = details.get("action") if isinstance(details.get("action"), dict) else {}
    judge = details.get("judge") if isinstance(details.get("judge"), dict) else {}
    length = details.get("length") if isinstance(details.get("length"), dict) else {}
    weights = details.get("weights") if isinstance(details.get("weights"), dict) else {}
    gold_keep = float(action.get("gold_keep", 0.0))
    gold_drop = float(action.get("gold_drop", 0.0))
    other_keep = float(action.get("other_keep", 0.0))
    other_drop = float(action.get("other_drop", 0.0))
    gold_count = float(action.get("gold_count", 0.0))
    other_count = float(action.get("other_count", 0.0))
    flat = {
        "score": float(scored.get("score", 0.0)),
        "acc": float(scored.get("score", 0.0)),
        "g_format": float(scored.get("g_format", 0.0)),
        "R_action": float(scored.get("R_action", 0.0)),
        "R_judge": float(scored.get("R_judge", 0.0)),
        "R_quality": float(scored.get("R_quality", 0.0)),
        "R_length": float(scored.get("R_length", 0.0)),
        "raw_inside_gate": float(scored.get("raw_inside_gate", 0.0)),
        "inside_gate_score": float(scored.get("inside_gate_score", 0.0)),
        "R_action_normalized": float(scored.get("R_action_normalized", 0.0)),
        "R_action_norm_score": float(scored.get("R_action_norm_score", 0.0)),
        "R_action_norm_min": float(scored.get("R_action_norm_min", 0.0)),
        "R_action_norm_max": float(scored.get("R_action_norm_max", 0.0)),
        "R_judge_normalized": float(scored.get("R_judge_normalized", 0.0)),
        "R_judge_norm_score": float(scored.get("R_judge_norm_score", 0.0)),
        "R_judge_norm_min": float(scored.get("R_judge_norm_min", 0.0)),
        "R_judge_norm_max": float(scored.get("R_judge_norm_max", 0.0)),
        "component_norm_enabled": float(scored.get("component_norm_enabled", 0.0)),
        "lambda_qual": float(weights.get("lambda_qual", 0.0)),
        "configured_lambda_qual": float(weights.get("configured_lambda_qual", 0.0)),
        "action_weight": float(weights.get("action_weight", 1.0)),
        "judge_enabled": float(weights.get("judge_enabled", 0.0)),
        "phase_is_judge": 1.0 if weights.get("reward_phase") == "judge" else 0.0,
        "phase2_enabled": float(weights.get("phase2_enabled", 0.0)),
        "phase2_latched": float(weights.get("phase2_latched", 0.0)),
        "phase2_triggered_now": float(weights.get("phase2_triggered_now", 0.0)),
        "phase2_format_threshold": float(weights.get("phase2_format_threshold", 0.99)),
        "eval_g_format": float(weights.get("eval_g_format", -1.0)),
        "action_eta": float(action.get("action_eta", 0.0)),
        "R_gold": float(action.get("R_gold", 0.0)),
        "R_other": float(action.get("R_other", 0.0)),
        "gold_count": gold_count,
        "other_count": other_count,
        "gold_keep_count": gold_keep,
        "gold_drop_count": gold_drop,
        "other_keep_count": other_keep,
        "other_drop_count": other_drop,
        "faithfulness": float(judge.get("faithfulness", 0.0)),
        "utility": float(judge.get("utility", 0.0)),
        "constructed_memory_chars": float(judge.get("constructed_memory_chars", 0.0)),
        "keep_rate": float(length.get("keep_rate", 0.0)),
        "over_keep": float(length.get("over_keep", 0.0)),
    }
    # VERL batches reward_extra_info by indexing every key from the first
    # sample across the batch, so these keys must be present for all samples.
    # Use -1.0 as an undefined-rate sentinel when the denominator is absent.
    flat["gold_keep_rate"] = _safe_div(gold_keep, gold_count) if gold_count > 0 else -1.0
    flat["gold_drop_rate"] = _safe_div(gold_drop, gold_count) if gold_count > 0 else -1.0
    flat["other_keep_rate"] = _safe_div(other_keep, other_count) if other_count > 0 else -1.0
    flat["other_drop_rate"] = _safe_div(other_drop, other_count) if other_count > 0 else -1.0
    return flat


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str | dict[str, Any] | None = None,
    extra_info: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    return _flat_score_dict(
        score_with_details(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug active RL reward for one prediction.")
    parser.add_argument("--solution-file", required=True)
    parser.add_argument("--extra-info-file", required=True)
    parser.add_argument("--ground-truth-file")
    args = parser.parse_args()

    solution = open(args.solution_file, "r", encoding="utf-8").read()
    extra_obj = json.load(open(args.extra_info_file, "r", encoding="utf-8"))
    ground_truth_obj = None
    if args.ground_truth_file:
        ground_truth_obj = json.load(open(args.ground_truth_file, "r", encoding="utf-8"))
    print(
        json.dumps(
            score_with_details(solution_str=solution, ground_truth=ground_truth_obj, extra_info=extra_obj),
            ensure_ascii=False,
            indent=2,
        )
    )
