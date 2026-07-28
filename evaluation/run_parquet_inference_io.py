#!/usr/bin/env python3
"""Run local vLLM inference on VERL Parquet prompts while preserving public metadata."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-sample vLLM inference on parquet prompts and save input/output JSONL."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--job",
        action="append",
        nargs=2,
        metavar=("PARQUET", "OUTPUT_JSONL"),
        required=True,
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def normalize_messages(value):
    messages = value.tolist() if hasattr(value, "tolist") else value
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def to_builtin(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def public_metadata(extra_info: Any) -> dict[str, Any]:
    extra = to_builtin(extra_info)
    if not isinstance(extra, dict):
        return {}
    metadata = extra.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    question_id = metadata.get("question_id") or extra.get("question_id")
    result = {
        "question_id": question_id,
        "query_id": metadata.get("query_id") or question_id,
        "sample_id": metadata.get("sample_id"),
        "qa_idx": metadata.get("qa_idx"),
        "query": extra.get("query"),
        "query_time": metadata.get("query_timestamp") or extra.get("query_time"),
        "question_type": metadata.get("question_type"),
        "category": metadata.get("category"),
    }
    return {key: value for key, value in result.items() if value not in (None, "")}


def readable_input(messages):
    return "\n".join(
        f"{message['role']}\n{message['content']}" for message in messages
    )


def main():
    args = parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for parquet_path, output_path in args.job:
        frame = pd.read_parquet(parquet_path)
        if "prompt" not in frame.columns:
            raise ValueError(f"{parquet_path} has no prompt column")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Starting {parquet_path}: rows={len(frame)}, output={destination}",
            flush=True,
        )

        with destination.open("w", encoding="utf-8") as output_file:
            for start in range(0, len(frame), args.batch_size):
                stop = start + args.batch_size
                values = frame["prompt"].iloc[start:stop]
                messages_batch = [normalize_messages(value) for value in values]
                if "extra_info" in frame.columns:
                    metadata_batch = [
                        public_metadata(value)
                        for value in frame["extra_info"].iloc[start:stop]
                    ]
                else:
                    metadata_batch = [{} for _ in messages_batch]
                prompts = [
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for messages in messages_batch
                ]
                results = llm.generate(prompts, sampling_params)
                for messages, result, metadata in zip(
                    messages_batch,
                    results,
                    metadata_batch,
                ):
                    record = {
                        **metadata,
                        "input": readable_input(messages),
                        "output": result.outputs[0].text,
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"Saved {min(start + args.batch_size, len(frame))}/{len(frame)}", flush=True)

        print(f"Finished {parquet_path}: output={destination}", flush=True)


if __name__ == "__main__":
    main()
