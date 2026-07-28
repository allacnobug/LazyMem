# Reward

## Output format

The final model answer must be a JSON array, optionally preceded by a closed
`<think>...</think>` block:

```json
[
  {"op": "KEEP", "compressed_content": "...", "reason": "..."},
  {"op": "DROP", "compressed_content": "", "reason": "..."}
]
```

The array length and order must match `extra_info.source_messages`. Locator
fields are restored from the source messages. Markdown code fences, invalid
operations, non-empty DROP content, or a length mismatch fail the binary format
gate.

## Action reward

Let `A_g` be KEEP accuracy on gold messages and `A_n` be DROP accuracy on
non-gold messages. The final configuration fixes:

```text
eta = 0.915
R_action = eta * A_g + (1 - eta) * A_n
```

If a sample contains only one category, its category accuracy is used directly.

## Quality reward

Phase 2 judges each KEEP result for faithfulness and utility using levels
`0/1/2`. A faithfulness level of zero maps to zero reward. Otherwise the mean
level is divided by two, producing a value in `[0, 1]`. Item rewards are
averaged across KEEP decisions. Samples with no KEEP decisions receive zero
quality reward without an LLM request.

Invalidly formatted samples never call the judge.

## Permanent phase switch

Phase 1 uses only `R_action`. When the global validation format accuracy is
strictly greater than `0.99`, the reward creates a latch file and enters Phase 2:

```text
score = (1 - lambda_qual) * R_action + lambda_qual * R_quality
lambda_qual = 0.5
```

Once created, the latch file keeps Phase 2 enabled across later validation
drops and process restarts. Do not copy an old latch file into a fresh run.

`R_length` is still reported as a diagnostic metric, but it is not part of the
final reward scalar.

The global validation metric used by this switch requires the included VERL
0.7.1 patches.
