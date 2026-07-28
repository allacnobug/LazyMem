"""Evaluate LongMemEval predictions with a benchmark-compatible LLM judge."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-judge evaluation for Module-2 predictions on LongMemEval.")
    p.add_argument("--pred-file", default="", help="Single prediction file path")
    p.add_argument("--pred-files", default="", help="Comma-separated prediction file paths")
    p.add_argument(
        "--pred-glob",
        default="",
        help="Glob for multiple prediction files, e.g. checkpoints/module2/m2-*/predictions_clean.jsonl",
    )
    p.add_argument(
        "--ref-file",
        default="data/Benchmark/LongMemEval/data/longmemeval_s_cleaned.json",
        help="Reference file containing question/answer/question_type",
    )
    p.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    p.add_argument(
        "--judge-base-urls",
        default="",
        help="comma-separated judge base urls, e.g. http://localhost:8001/v1,http://localhost:8002/v1",
    )
    p.add_argument("--judge-api-key", default="")
    p.add_argument("--judge-model", default="deepseek-v4-pro")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument(
        "--thinking-mode",
        default="none",
        choices=["none", "qwen_enable", "qwen_disable"],
        help="For Qwen3-style APIs: pass chat_template_kwargs enable_thinking true/false.",
    )
    p.add_argument("--workers", type=int, default=1, help="Concurrent judge workers")
    p.add_argument("--limit", type=int, default=0, help="0 means evaluate all")
    p.add_argument("--out-file", default=None, help="default: <pred-file>.llm_judge.jsonl")
    p.add_argument("--summary-file", default=None, help="default: <pred-file>.llm_judge.summary.json")
    p.add_argument(
        "--batch-summary-file",
        default="",
        help="When evaluating multiple pred files, write combined summary to this JSON file",
    )
    return p.parse_args()


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def parse_base_urls(base_url: str, base_urls: str) -> List[str]:
    urls = []
    if base_urls:
        urls.extend([x.strip() for x in base_urls.split(",") if x.strip()])
    if not urls and base_url:
        urls.append(base_url.strip())
    if not urls:
        raise ValueError("No judge base url provided.")
    return urls


def collect_pred_files(args) -> List[Path]:
    files = []
    if args.pred_file.strip():
        files.append(Path(args.pred_file.strip()))
    if args.pred_files.strip():
        files.extend([Path(x.strip()) for x in args.pred_files.split(",") if x.strip()])
    if args.pred_glob.strip():
        files.extend(sorted(Path(".").glob(args.pred_glob.strip())))
    # dedup keep order
    out = []
    seen = set()
    for fp in files:
        key = str(fp)
        if key in seen:
            continue
        seen.add(key)
        out.append(fp)
    if not out:
        raise ValueError("No prediction files found. Use --pred-file or --pred-files or --pred-glob.")
    return out


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors, "
                "the model's response is still correct.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all points in the rubric. "
                "The response is correct as long as it recalls and uses personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise ValueError(f"Unsupported question_type: {task}")

    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say the information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def judge_once(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_mode: str,
) -> str:
    extra_body = None
    if thinking_mode == "qwen_enable":
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    elif thinking_mode == "qwen_disable":
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return (resp.choices[0].message.content or "").strip()


def safe_mean(xs: List[int]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def prediction_text(pred: dict) -> str:
    for key in ("prediction", "answer", "response", "model_answer"):
        value = pred.get(key)
        if value not in (None, ""):
            return str(value)
    raw_response = pred.get("raw_response")
    return "" if raw_response is None else str(raw_response)


def eval_one_pred_file(args, pred_file: Path, refs: List[dict]):
    from openai import OpenAI

    ref_file = Path(args.ref_file)
    out_file = Path(args.out_file) if args.out_file else pred_file.with_suffix(pred_file.suffix + ".llm_judge.jsonl")
    summary_file = (
        Path(args.summary_file)
        if args.summary_file
        else pred_file.with_suffix(pred_file.suffix + ".llm_judge.summary.json")
    )

    preds = load_json_or_jsonl(pred_file)
    qid2ref: Dict[str, dict] = {x["question_id"]: x for x in refs if "question_id" in x}

    base_urls = parse_base_urls(args.judge_base_url, args.judge_base_urls)
    url_n = len(base_urls)

    rows = []
    by_type: Dict[str, List[int]] = {}
    all_labels: List[int] = []
    skipped = 0
    pred_items = list(enumerate(preds))
    if args.limit > 0:
        pred_items = pred_items[: args.limit]

    def process_one(item):
        i, pred = item
        qid = pred.get("question_id") or pred.get("conversation_id")
        ref = qid2ref.get(qid)
        if not ref:
            return {"skip": True}
        qtype = ref.get("question_type", "unknown")
        question = ref.get("question", pred.get("question", ""))
        answer = ref.get("answer", pred.get("gold_answer", ""))
        response = prediction_text(pred)
        abstention = "_abs" in str(qid)
        prompt = get_anscheck_prompt(
            task=qtype,
            question=str(question),
            answer=str(answer),
            response=str(response),
            abstention=abstention,
        )
        base_url = base_urls[i % url_n]
        client = OpenAI(api_key=args.judge_api_key, base_url=base_url)
        judge_raw = judge_once(
            client=client,
            model=args.judge_model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_mode=args.thinking_mode,
        )
        label = 1 if "yes" in judge_raw.lower() else 0
        row = dict(pred)
        row["llm_judge"] = {
            "judge_model": args.judge_model,
            "judge_base_url": base_url,
            "thinking_mode": args.thinking_mode,
            "judge_raw": judge_raw,
            "label": bool(label),
        }
        return {"skip": False, "row": row, "qtype": qtype, "label": label}

    if args.workers <= 1:
        iterable = pred_items
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(pred_items), desc=f"LLM Judge {pred_file.parent.name}", unit="sample")
        for item in iterable:
            out = process_one(item)
            if out["skip"]:
                skipped += 1
                continue
            rows.append(out["row"])
            all_labels.append(out["label"])
            by_type.setdefault(out["qtype"], []).append(out["label"])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            mapped = ex.map(process_one, pred_items)
            if tqdm is not None:
                mapped = tqdm(mapped, total=len(pred_items), desc=f"LLM Judge {pred_file.parent.name}", unit="sample")
            for out in mapped:
                if out["skip"]:
                    skipped += 1
                    continue
                rows.append(out["row"])
                all_labels.append(out["label"])
                by_type.setdefault(out["qtype"], []).append(out["label"])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_acc = {k: round(safe_mean(v), 6) for k, v in sorted(by_type.items())}
    summary = {
        "pred_file": str(pred_file),
        "ref_file": str(ref_file),
        "judge_model": args.judge_model,
        "total_evaluated": len(rows),
        "skipped_missing_ref": skipped,
        "overall_acc": round(safe_mean(all_labels), 6),
        "acc_by_question_type": type_acc,
        "out_file": str(out_file),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    pred_files = collect_pred_files(args)
    refs = load_json_or_jsonl(Path(args.ref_file))

    all_summaries = []
    for pred_file in pred_files:
        summary = eval_one_pred_file(args, pred_file=pred_file, refs=refs)
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    if len(all_summaries) > 1:
        overall = {
            "num_experiments": len(all_summaries),
            "judge_model": args.judge_model,
            "mean_overall_acc": round(safe_mean([s["overall_acc"] for s in all_summaries]), 6),
            "experiments": all_summaries,
        }
        if args.batch_summary_file:
            fp = Path(args.batch_summary_file)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved batch summary: {fp}")
        else:
            print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
