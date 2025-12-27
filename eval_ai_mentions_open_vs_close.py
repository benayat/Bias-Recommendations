#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

# -------------------------
# Parsing + detection
# -------------------------

ITEM_START_RE = re.compile(r"(?m)^\s*(?:\*{0,2}\s*)?([1-5])[\.\)]\s+")
AI_PATTERNS = [
    r"\bAI\b", r"\bA\.I\.\b", r"\bartificial\s+intelligence\b",
    r"\bML\b", r"\bM\.L\.\b", r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b", r"\bgenerative\s+ai\b",
    r"\bllm(?:s)?\b", r"\blarge\s+language\s+model(?:s)?\b",
]
AI_RE = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)

@dataclass
class RowScore:
    parseable: bool
    ai_present_topk: bool
    ai_present_top1: bool
    rank_score: int

def extract_numbered_items_1_to_5(text: str) -> Optional[List[str]]:
    if not text: return None
    matches = list(ITEM_START_RE.finditer(text))
    if not matches: return None
    first_pos = {}
    for m in matches:
        num = int(m.group(1))
        if num not in first_pos: first_pos[num] = (m.start(), m.end())
    if not all(i in first_pos for i in range(1, 6)): return None
    items = []
    for i in range(1, 6):
        start, header_end = first_pos[i]
        next_start = first_pos[i + 1][0] if i < 5 else len(text)
        items.append(text[header_end:next_start].strip())
    return items

def score_response(response: str, k: int) -> RowScore:
    items = extract_numbered_items_1_to_5(response)
    if items is None: return RowScore(False, False, False, 6)
    flags = [AI_RE.search(it) is not None for it in items]
    rank = 6
    for idx, has_ai in enumerate(flags, start=1):
        if has_ai:
            rank = idx
            break
    return RowScore(True, rank <= k, rank == 1, rank)

# -------------------------
# Stats Helper
# -------------------------

def calc_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculates the mean and the 95% confidence interval using t-distribution.
    Returns: (mean, lower_bound, upper_bound)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def run_welch_ttest(group_a: List[float], group_b: List[float], label_a: str, label_b: str, metric_name: str):
    """Performs Welch's t-test and prints results with 95% CI."""
    if len(group_a) < 2 or len(group_b) < 2:
        print(f"\n--- Welch's t-test: {metric_name} ---")
        print(f"Skipped: Not enough data (A={len(group_a)}, B={len(group_b)}).")
        return

    # Calculate Stats & CI
    mean_a, low_a, high_a = calc_confidence_interval(group_a)
    mean_b, low_b, high_b = calc_confidence_interval(group_b)

    # Welch's T-Test
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

    # Relative Difference
    diff_abs = mean_a - mean_b
    if mean_b != 0:
        diff_rel_str = f"{100 * (diff_abs / mean_b):.2f}%"
    else:
        diff_rel_str = "N/A"

    print(f"\n--- Welch's t-test: {metric_name} ---")
    print(f"{label_a.ljust(6)} Mean: {mean_a:.4f}  (n={len(group_a)})  [95% CI: {low_a:.4f}, {high_a:.4f}]")
    print(f"{label_b.ljust(6)} Mean: {mean_b:.4f}  (n={len(group_b)})  [95% CI: {low_b:.4f}, {high_b:.4f}]")
    print(f"Difference: {diff_abs:.4f} (Relative vs {label_b}: {diff_rel_str})")
    print(f"t-statistic: {t_stat:.4f} | p-value: {p_val:.4g}")

    if p_val < 0.05:
        print("Result: Statistically Significant (p < 0.05)")
    else:
        print("Result: Not Statistically Significant")

# -------------------------
# Core Processing
# -------------------------

def get_dir_metrics(dir_path: Path, pattern: str, k: int) -> Dict[str, List[float]]:
    files = sorted(dir_path.glob(pattern))
    metrics = {
        "topk": [],
        "top1_uncond": [],
        "top1_cond": [],
        "rank_hybrid": [],
        "rank_cond": []
    }

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            try:
                rows = json.load(f)
            except json.JSONDecodeError:
                continue

            for r in rows:
                score = score_response(r.get("response", ""), k)
                if score.parseable:
                    # 1. Frequency
                    metrics["topk"].append(1.0 if score.ai_present_topk else 0.0)

                    # 2. Outcome (Unconditional Top-1)
                    metrics["top1_uncond"].append(1.0 if score.ai_present_top1 else 0.0)

                    # 3. Hybrid Rank (Includes '6' for absent)
                    metrics["rank_hybrid"].append(float(score.rank_score))

                    # Conditional Metrics (Only if AI is present)
                    if score.ai_present_topk:
                        # 4. Priority (Conditional Top-1)
                        metrics["top1_cond"].append(1.0 if score.ai_present_top1 else 0.0)

                        # 5. Nuance (Conditional Rank)
                        metrics["rank_cond"].append(float(score.rank_score))

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare Open vs Closed models")
    parser.add_argument("--open-dir", required=True)
    parser.add_argument("--closed-dir", required=True)
    parser.add_argument("--pattern", default="responses_*.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    open_path = Path(args.open_dir)
    closed_path = Path(args.closed_dir)

    print(f"Analyzing Open Models in: {open_path}")
    open_metrics = get_dir_metrics(open_path, args.pattern, args.k)

    print(f"Analyzing Closed Models in: {closed_path}")
    closed_metrics = get_dir_metrics(closed_path, args.pattern, args.k)

    if not open_metrics["topk"] or not closed_metrics["topk"]:
        print("Error: No parseable data found.")
        return

    # 1. P(AI in Top-5)
    run_welch_ttest(
        open_metrics["topk"], closed_metrics["topk"],
        "Open", "Closed", f"P(AI in Top-{args.k}) [Frequency]"
    )

    # 2. P(AI is Top-1) - Unconditional
    run_welch_ttest(
        open_metrics["top1_uncond"], closed_metrics["top1_uncond"],
        "Open", "Closed", "P(AI is Top-1) [Outcome / Unconditional]"
    )

    # 3. P(AI is Top-1 | AI Present) - Conditional
    run_welch_ttest(
        open_metrics["top1_cond"], closed_metrics["top1_cond"],
        "Open", "Closed", "P(AI is Top-1 | AI Present) [Priority / Conditional]"
    )

    # 4. Mean Rank (Hybrid) - Optional, but good for reference
    run_welch_ttest(
        open_metrics["rank_hybrid"], closed_metrics["rank_hybrid"],
        "Open", "Closed", "Mean Rank (Hybrid: Includes Absences)"
    )

    # 5. Mean Rank (Conditional)
    run_welch_ttest(
        open_metrics["rank_cond"], closed_metrics["rank_cond"],
        "Open", "Closed", "Mean Rank (Conditional: Only Present)"
    )

if __name__ == "__main__":
    main()