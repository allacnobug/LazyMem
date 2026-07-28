"""Run a trained LazyMem SFT checkpoint on the window-level validation set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "outputs" / "sft" / "lazymem-qwen4b"
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "processed" / "lazymem_window_dft_val_balanced.json"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "outputs" / "predictions" / "sft_val.jsonl"


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected object row")
                rows.append(row)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return data


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sample_key(row: dict[str, Any], row_index: int) -> str:
    metadata = row.get("metadata") or {}
    parts = [
        metadata.get("question_id"),
        metadata.get("segment_index"),
        metadata.get("window_index"),
        metadata.get("sample_index"),
    ]
    if any(part is not None for part in parts):
        return "|".join("" if part is None else str(part) for part in parts)
    return f"row:{row_index}"


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skip invalid output line {path}:{line_no}")
                continue
            key = row.get("sample_key")
            if isinstance(key, str) and key:
                done.add(key)
    return done


def split_think_answer(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        inner_start = len("<think>")
        inner_end = stripped.find("</think>", inner_start)
        reasoning = stripped[inner_start:inner_end].strip()
        answer = stripped[inner_end + len("</think>") :].strip()
        return reasoning, answer
    return "", stripped


def input_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("row.messages must contain at least system/user messages")
    return messages[:2]


def gold_output(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 3 and isinstance(messages[2], dict):
        return str(messages[2].get("content") or "")
    return ""


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]], *, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3 window-DFT checkpoint on the balanced validation set and save raw IO."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    rows = load_json_or_jsonl(args.data_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()

    keyed_rows = [
        (sample_key(row, row_index), row_index, row)
        for row_index, row in enumerate(rows)
        if row_index % args.num_shards == args.shard_index
    ]
    done = load_done_keys(args.output_file)
    pending = [(key, row_index, row) for key, row_index, row in keyed_rows if key not in done]

    print(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "data_file": str(args.data_file),
                "output_file": str(args.output_file),
                "total_rows": len(rows),
                "shard_rows": len(keyed_rows),
                "completed_rows": len(done),
                "pending_rows": len(pending),
                "enable_thinking": args.enable_thinking,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
            },
            ensure_ascii=False,
        )
    )

    for batch in tqdm(list(batched(pending, args.batch_size)), desc="infer"):
        keys = [item[0] for item in batch]
        row_indexes = [item[1] for item in batch]
        batch_rows = [item[2] for item in batch]
        batch_messages = [input_messages(row) for row in batch_rows]
        prompts = [
            render_prompt(tokenizer, messages, enable_thinking=args.enable_thinking)
            for messages in batch_messages
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_width = inputs["input_ids"].shape[1]
        output_rows: list[dict[str, Any]] = []
        for offset, (row, output_ids) in enumerate(zip(batch_rows, generated)):
            generated_ids = output_ids[prompt_width:]
            prediction_raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            reasoning, answer = split_think_answer(prediction_raw)
            gold_reasoning, gold_answer = split_think_answer(gold_output(row))
            output_rows.append(
                {
                    "sample_key": keys[offset],
                    "row_index": row_indexes[offset],
                    "metadata": row.get("metadata") or {},
                    "input_messages": batch_messages[offset],
                    "prompt_text": prompts[offset],
                    "prediction_raw": prediction_raw,
                    "prediction_reasoning": reasoning,
                    "prediction_answer": answer,
                    "gold_output": gold_output(row),
                    "gold_reasoning": gold_reasoning,
                    "gold_answer": gold_answer,
                    "infer_meta": {
                        "model_path": str(args.model_path),
                        "enable_thinking": args.enable_thinking,
                        "prompt_tokens": int(inputs["attention_mask"][offset].sum().item()),
                        "completion_tokens": int(generated_ids.numel()),
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
            )
        append_jsonl(args.output_file, output_rows)

    print(json.dumps({"output_file": str(args.output_file), "rows": len(load_done_keys(args.output_file))}))


if __name__ == "__main__":
    main()
