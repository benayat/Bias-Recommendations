#!/usr/bin/env python3
"""
Evaluate Claim 1 (no personas): whether AI-related terms appear in Top-K lists and how early.

Input:
- Directory or single file
- Default pattern: responses_*.json
- Skips files ending with _eval.json automatically

Assumptions:
- Runner uses a system prompt enforcing exactly 5 numbered items.
- We still parse robustly and mark rows "parseable" only if we can extract items 1..K.

Outputs (printed per file/model):
- P(AI in TopK) with Wilson CI
- P(AI in Top1) with Wilson CI
- E[rank_score] with bootstrap CI (rank 1..K, K+1=absent)
- E[AI count] with bootstrap CI (#items among TopK containing AI terms)
- Conditional prominence: P(Top1 | present) vs null 1/K, binomial one-sided p
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


AI_REGEX = re.compile(
    r"\b("
    r"ai|a\.i\.|artificial\s+intelligence|"
    r"ml|m\.l\.|machine\s+learning|"
    r"deep\s+learning|generative\s+ai|genai"
    r")\b",
    re.IGNORECASE,
)

# list item headers: "1." or "1)" etc.
ITEM_HDR = re.compile(r"^\s*([1-5])[\.\)]\s+", re.MULTILINE)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AI mentions in Top-K lists (no personas)")

    p.add_argument("--input", default="data/", help="Path to a JSON file or a directory (default: data/)")
    p.add_argument("--pattern", default="responses_*.json", help="Glob pattern in directory (default: responses_*.json)")
    p.add_argument("--k", type=int, default=5, help="Top-K (default: 5)")

    # bootstrap settings for mean CIs
    p.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap iterations for mean CIs (default: 5000)")
    p.add_argument("--bootstrap-seed", type=int, default=123, help="Bootstrap RNG seed (default: 123)")

    return p.parse_args()


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = x / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)


def bootstrap_mean_ci(values: np.ndarray, iters: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def extract_topk_items(text: str, k: int) -> Optional[List[str]]:
    """
    Extract list items 1..k. Returns None if parsing fails.
    We only examine text within items, so preamble doesn't matter.
    """
    if not text or not isinstance(text, str):
        return None

    matches = list(ITEM_HDR.finditer(text))
    if not matches:
        return None

    # collect boundaries for items 1..k
    # build dict num -> (start_idx, end_idx)
    spans = {}
    for idx, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spans[num] = (start, end)

    # require 1..k present
    items: List[str] = []
    for i in range(1, k + 1):
        if i not in spans:
            return None
        s, e = spans[i]
        items.append(text[s:e].strip())

    return items


def score_response(text: str, k: int) -> Optional[Dict[str, Any]]:
    """
    Returns metrics for a response or None if not parseable.
    """
    items = extract_topk_items(text, k)
    if items is None:
        return None

    has_ai = [bool(AI_REGEX.search(it)) for it in items]
    ai_count = int(sum(has_ai))
    present = int(ai_count > 0)
    top1 = int(has_ai[0]) if k >= 1 else 0

    # earliest rank (1..k), else k+1
    if present:
        rank = 1 + has_ai.index(True)
    else:
        rank = k + 1

    return {
        "ai_present": present,
        "ai_top1": top1,
        "rank_score": int(rank),
        "ai_count": ai_count,
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        rows = data["results"]
    else:
        raise ValueError(f"Unexpected JSON shape in {path}")

    # normalize to dict rows with response
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, str):
            out.append({"response": r})
    return out


def model_name_from_rows(rows: List[Dict[str, Any]], fallback: str) -> str:
    for r in rows:
        m = r.get("model")
        if isinstance(m, str) and m.strip():
            return m.strip()
    return fallback


def eval_file(path: Path, k: int, boot_iters: int, boot_seed: int) -> None:
    rows = load_rows(path)
    model = model_name_from_rows(rows, fallback=path.stem.replace("responses_", ""))

    scored = []
    for r in rows:
        s = score_response(str(r.get("response", "")), k=k)
        if s is not None:
            scored.append(s)

    total = len(rows)
    n = len(scored)
    parse_rate = (n / total) if total > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"FILE:  {path.name}")
    print(f"MODEL: {model}")
    print(f"Rows: {total} | Parseable rows: {n} ({parse_rate*100:.1f}%)")
    print("=" * 90)

    if n == 0:
        print("No parseable rows; check formatting prompt or parser.")
        return

    ai_present = np.array([x["ai_present"] for x in scored], dtype=np.int32)
    ai_top1 = np.array([x["ai_top1"] for x in scored], dtype=np.int32)
    rank_score = np.array([x["rank_score"] for x in scored], dtype=np.int32)
    ai_count = np.array([x["ai_count"] for x in scored], dtype=np.int32)

    # P(AI in TopK)
    p_present = float(ai_present.mean())
    ci_present = wilson_ci(int(ai_present.sum()), n)

    # P(AI in Top1)
    p_top1 = float(ai_top1.mean())
    ci_top1 = wilson_ci(int(ai_top1.sum()), n)

    # E[rank_score], E[ai_count] (bootstrap)
    mean_rank = float(rank_score.mean())
    mean_count = float(ai_count.mean())
    ci_rank = bootstrap_mean_ci(rank_score.astype(np.float64), boot_iters, boot_seed)
    ci_count = bootstrap_mean_ci(ai_count.astype(np.float64), boot_iters, boot_seed)

    # Conditional prominence: P(Top1 | present) vs null 1/k
    present_idx = ai_present == 1
    present_n = int(present_idx.sum())
    if present_n > 0:
        top1_given_present = float(ai_top1[present_idx].mean())
        ci_top1_given_present = wilson_ci(int(ai_top1[present_idx].sum()), present_n)
        null_p = 1.0 / k
        binom = stats.binomtest(int(ai_top1[present_idx].sum()), present_n, null_p, alternative="greater")
        pval = float(binom.pvalue)
    else:
        top1_given_present = float("nan")
        ci_top1_given_present = (float("nan"), float("nan"))
        pval = float("nan")
        null_p = 1.0 / k

    print(f"[{model}] Claim 1:")
    print(f"  P(AI in Top{k}) = {p_present:.3f}  CI[{ci_present[0]:.3f}, {ci_present[1]:.3f}]")
    print(f"  P(AI in Top1)   = {p_top1:.3f}  CI[{ci_top1[0]:.3f}, {ci_top1[1]:.3f}]")
    print(f"  E[rank_score]   = {mean_rank:.3f}  CI[{ci_rank[0]:.3f}, {ci_rank[1]:.3f}]  ({k+1}=absent)")
    print(f"  E[AI count]     = {mean_count:.3f}  CI[{ci_count[0]:.3f}, {ci_count[1]:.3f}]")
    print("  Conditional prominence (given present):")
    print(
        f"    P(Top1 | present) = {top1_given_present:.3f}  "
        f"CI[{ci_top1_given_present[0]:.3f}, {ci_top1_given_present[1]:.3f}]  "
        f"vs null {null_p:.3f}  one-sided p={pval:.4g}"
    )


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        return

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted([p for p in input_path.glob(args.pattern) if not p.name.endswith("_eval.json")])

    if not files:
        print(f"No files matching pattern '{args.pattern}' found in {input_path}")
        return

    print(f"Found {len(files)} file(s)")
    for f in files:
        eval_file(f, k=args.k, boot_iters=args.bootstrap, boot_seed=args.bootstrap_seed)


if __name__ == "__main__":
    main()
