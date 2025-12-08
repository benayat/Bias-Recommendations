#!/usr/bin/env python3
"""
Run a single condition (no personas) against questions in batch.

Modes:
- Canonical-only (default): each qid once.
- With paraphrases (--include-paraphrases): each qid gets:
    canonical + ALL paraphrases (5 variants total in your setup).

Robustness knobs:
- --n and --seed (vLLM SamplingParams.n/seed)
- --seeds for multi-pass runs with different seeds

Assumes your LLMClient supports SamplingConfig fields:
  temperature, top_p, max_tokens, n, seed

And LLMClient returns:
- response: str when n=1
- response: List[str] when n>1

Output filename defaults to:
  data/responses_<model>[_paraphrases].json
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from constants import HOME_CONFIG_SMALL_RECOMMENDATIONS, QUESTIONS, QID_ORDER
from llm import LLMClient, SamplingConfig


LIST_ONLY_SYSTEM_PROMPT = (
    "Output exactly 5 recommendations as a numbered list (1-5). "
    "Each item must be formatted as: '<label> — <1–2 sentence rationale>'. "
    "The label should be 2–8 words (not a single word). "
    # "Do not mention being an AI or the prompt. "
    "No text before item 1 and no text after item 5."
)


def parse_args():
    p = argparse.ArgumentParser(description="Run recommendations prompts (no personas)")

    # model / config
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--scale-for-model-size", action="store_true", help="Scale LLM config for model size")

    # question variants
    p.add_argument(
        "--include-paraphrases",
        action="store_true",
        help="Include canonical + ALL paraphrases for each question.",
    )

    # decoding
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="Nucleus sampling top_p")
    p.add_argument("--max-tokens", type=int, default=900, dest="max_tokens", help="Max new tokens")

    # robustness
    p.add_argument("--n", type=int, default=1, help="Number of samples per prompt (SamplingParams.n)")
    p.add_argument("--seed", type=int, default=12345, help="Seed for sampling (SamplingParams.seed)")
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds. If provided, overrides --seed and runs multiple passes.",
    )

    # output
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output path (.json). If empty, defaults to data/responses_<model>[_paraphrases].json",
    )

    return p.parse_args()


def _parse_seeds(seed: int, seeds_csv: str) -> List[int]:
    s = seeds_csv.strip()
    if not s:
        return [int(seed)]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--seeds was provided but no valid seeds were parsed.")
    return out


def _question_variants(qid: str, include_paraphrases: bool) -> List[Tuple[str, str]]:
    q = QUESTIONS[qid]
    variants: List[Tuple[str, str]] = [("canonical", q["canonical"])]
    if include_paraphrases:
        for i, para in enumerate(q.get("paraphrases", []), start=1):
            variants.append((f"paraphrase_{i}", para))
    return variants


def build_batch_prompts(include_paraphrases: bool) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for qid in QID_ORDER:
        for variant_id, question_text in _question_variants(qid, include_paraphrases):
            prompts.append(
                {
                    "messages": [
                        {"role": "system", "content": LIST_ONLY_SYSTEM_PROMPT},
                        {"role": "user", "content": question_text},
                    ],
                    "metadata": {
                        "qid": qid,
                        "variant": variant_id,
                        "subject": QUESTIONS[qid]["subject"],
                        "group": QUESTIONS[qid]["group"],
                    },
                }
            )
    return prompts


def _flatten_response(response: Any) -> List[str]:
    if response is None:
        return []
    if isinstance(response, str):
        return [response]
    if isinstance(response, list):
        return [str(x) for x in response]
    return [str(response)]


def main():
    args = parse_args()
    model_id = args.model
    seeds = _parse_seeds(args.seed, args.seeds)

    if args.n < 1:
        raise ValueError("--n must be >= 1")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")

    print(f"Model: {model_id}")
    print(f"Questions: {len(QID_ORDER)}")
    print(f"Include paraphrases: {args.include_paraphrases}")
    print(f"Decoding: temperature={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens} n={args.n}")
    print(f"Seeds: {seeds}")

    prompts = build_batch_prompts(args.include_paraphrases)
    print(f"Total prompts per pass: {len(prompts)}")

    # Setup LLM client
    model_size_match = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model_id)
    model_size_b = float(model_size_match.group(1)) if model_size_match else None

    config = HOME_CONFIG_SMALL_RECOMMENDATIONS
    if args.scale_for_model_size and model_size_b is not None:
        print(f"Scaling LLM config for model size: {model_size_b}B")
        config.scale_for_model_size(model_size_b)

    llm = LLMClient(model_name=model_id, config=config)

    rows: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            sampling_params = SamplingConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=args.n,
                seed=seed,
            )

            results = llm.run_batch(prompts, sampling_params, output_field="response")

            for r in results:
                responses = _flatten_response(r.get("response"))
                if args.n == 1:
                    # keep one row
                    rows.append(
                        {
                            "model": model_id,
                            "seed": seed,
                            "n": args.n,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                            "question_id": r.get("qid"),
                            "variant": r.get("variant"),
                            "subject": r.get("subject"),
                            "group": r.get("group"),
                            "response": responses[0] if responses else "",
                        }
                    )
                else:
                    # one row per sample
                    for j, txt in enumerate(responses, start=1):
                        rows.append(
                            {
                                "model": model_id,
                                "seed": seed,
                                "n": args.n,
                                "sample_idx": j,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "max_tokens": args.max_tokens,
                                "question_id": r.get("qid"),
                                "variant": r.get("variant"),
                                "subject": r.get("subject"),
                                "group": r.get("group"),
                                "response": txt,
                            }
                        )
    finally:
        llm.delete_client()

    sanitized_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    suffix = "_paraphrases" if args.include_paraphrases else ""
    default_out = f"data/responses_{sanitized_model}{suffix}.json"
    out_path = Path(args.out.strip() or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()
