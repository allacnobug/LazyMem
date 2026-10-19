# Data layout

The curated SFT train/validation files are included under `sft/data/`.
Raw benchmark datasets and generated RL data are not included. Scripts accept
explicit CLI paths; their relative defaults use the following local layout:

```text
data/
├── longmemeval_s_cleaned.json
├── locomo10.json
├── processed/
├── reward/
└── splits/
```

Generated files are written below `outputs/` and are ignored by Git.

Question-level reasoning evidence is generated directly from the raw
LongMemEval answer sessions and the split ID files:

```bash
python data_pipeline/generate_reasoning_evidence.py \
  --dataset-file data/longmemeval_s_cleaned.json \
  --split-dir data/splits \
  --output-dir data/reward \
  --splits train val test
```

The resulting
`data/reward/<split>/reasoning_qwen32b.split.jsonl` contains one row per
question. Its `reasoning_answer` is copied to `extra_info.required_evidence`
when the GRPO parquet is built.

The GRPO launcher consumes VERL-compatible parquet files. Each row must provide
the prompt plus reward metadata in `extra_info`, including:

- `source_messages`
- `gold_global_message_idxs`
- `query`
- `gold_answer`
- `required_evidence`
- `split`

Use `data_pipeline/prepare_window_rl_data.py --help` for the available input and
output path overrides.
