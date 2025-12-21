#!/usr/bin/env python3
"""
Run recommendation prompts (no personas) against a model using OpenAI-compatible API.

This is the OpenAI equivalent of main.py - same interface, same outputs, but uses
OpenAI API instead of vLLM.

Supports:
- canonical only (default)
- canonical + ALL paraphrases (--include-paraphrases)
- robustness via multi-seed (--seeds) and/or multi-sample per prompt (--n)  [requires temp>0 to matter]
- flattening outputs: one JSON row per (prompt, seed, sample_idx)

Output rows look like:
{
  "model": "...",
  "seed": 123,
  "n": 5,
  "sample_idx": 0,
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 512,
  "question_id": "...",
  "variant": "canonical" | "paraphrase_1" | ...,
  "subject": "...",
  "group": "...",
  "response": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai_llm import LLMClient, OpenAIConfig, SamplingConfig
from constants import QUESTIONS


DEFAULT_SYSTEM_PROMPT = (
    "Output exactly 5 recommendations as a numbered list (1-5). "
    "Each item must be formatted as: '<label> — <1–2 sentence rationale>'. "
    "The label should be 2–8 words (not a single word). "
    "No text before item 1 and no text after item 5."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run recommendation questions via OpenAI API")

    # model / config
    p.add_argument("--model", required=True, help="OpenAI model name (e.g. gpt-4, gpt-3.5-turbo)")
    p.add_argument("--api-key", type=str, default="",
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--base-url", type=str, default="https://api.openai.com/v1",
                   help="API base URL (for OpenAI-compatible endpoints)")

    # variants
    p.add_argument("--include-paraphrases", action="store_true",
                   help="Include canonical + ALL paraphrases for each question")

    # decoding
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="Nucleus sampling top_p")
    p.add_argument("--max-tokens", type=int, default=512, dest="max_tokens", help="Max new tokens")

    # robustness
    p.add_argument("--n", type=int, default=1, help="Number of samples per prompt")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--seeds", type=str, default="",
                   help="Comma-separated list of seeds. If provided, overrides --seed and runs multiple passes.")

    # prompt
    p.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                   help="System prompt enforcing list format")

    # output
    p.add_argument("--out", type=str, default="",
                   help="Output path (.json). If empty, defaults to data/responses_<model>[_paraphrases].json")

    # retry
    p.add_argument("--no-retry", action="store_true",
                   help="Disable automatic retry with exponential backoff")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Maximum retry attempts per request (default: 5)")

    return p.parse_args()


def _parse_seeds(seed: int, seeds_csv: str) -> List[int]:
    s = (seeds_csv or "").strip()
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


def build_batch_prompts(system_prompt: str, include_paraphrases: bool) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []

    # stable order
    qids = list(QUESTIONS.keys())

    for qid in qids:
        q = QUESTIONS[qid]
        for variant_id, question_text in _question_variants(qid, include_paraphrases):
            prompts.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_text},
                ],
                "metadata": {
                    "qid": qid,
                    "variant": variant_id,
                    "subject": q.get("subject", ""),
                    "group": q.get("group", ""),
                }
            })

    return prompts


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def main() -> None:
    args = parse_args()

    if args.n < 1:
        raise ValueError("--n must be >= 1")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")

    # Get API key from args or environment
    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No API key provided. Use --api-key or set OPENAI_API_KEY environment variable.")

    model_id = args.model
    seeds = _parse_seeds(args.seed, args.seeds)

    prompts = build_batch_prompts(args.system_prompt, args.include_paraphrases)

    print(f"Model: {model_id}")
    print(f"API Base URL: {args.base_url}")
    print(f"Questions: {len(QUESTIONS)} | Include paraphrases: {args.include_paraphrases}")
    print(f"Total prompts per pass: {len(prompts)}")
    print(f"Decoding: temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens} n={args.n}")
    print(f"Seeds: {seeds}")

    # Determine output path early for resume functionality
    suffix = "_paraphrases" if args.include_paraphrases else ""
    default_out = f"data/responses_{_sanitize_model_id(model_id)}{suffix}.json"
    out_path = Path(args.out.strip() or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create OpenAI config
    config = OpenAIConfig(
        api_key=api_key,
        base_url=args.base_url,
        enable_retry=not args.no_retry,
        max_retries=args.max_retries,
    )
    llm = LLMClient(model_name=model_id, config=config)

    all_rows: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            print(f"\nProcessing with seed: {seed}")
            
            sampling = SamplingConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=args.n,
                seed=seed,
            )

            results = llm.run_batch(prompts, sampling, output_field="response", output_path=str(out_path))

            # Flatten: one row per sample_idx
            for r in results:
                base = {
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
                }

                resp = r.get("response", "")
                if isinstance(resp, list):
                    # n > 1: multiple samples
                    for j, txt in enumerate(resp):
                        all_rows.append({**base, "sample_idx": j, "response": (txt or "").strip()})
                else:
                    # n = 1: single sample
                    all_rows.append({**base, "sample_idx": 0, "response": (resp or "").strip()})
    finally:
        llm.delete_client()

    # Results already saved by run_batch, just load and return final count
    print(f"\nCompleted: {len(all_rows)} total rows in {out_path}")


if __name__ == "__main__":
    main()

