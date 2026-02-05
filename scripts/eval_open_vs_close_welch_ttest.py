#!/usr/bin/env python3
"""
Welch's t-test analysis: Compare open vs closed models at the response level.

This script treats each individual response as an independent observation:
- Open models: ~1300 responses (13 models × 100 prompts)
- Closed models: ~400 responses (4 models × 100 prompts)

Performs Welch's t-test comparing two groups of responses directly.
Note: This assumes each response is independent (ignores within-model correlation).
"""

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

def get_prompt_id(row: Dict[str, Any]) -> str:
    """
    Extract unique prompt identifier from row.
    Combines question_id and variant to uniquely identify each prompt.
    """
    qid = row.get("question_id") or row.get("qid", "")
    variant = row.get("variant", "canonical")
    return f"{qid}_{variant}"

# -------------------------
# Core Processing
# -------------------------

def collect_all_responses(dir_path: Path, pattern: str, k: int) -> List[RowScore]:
    """
    Collect all individual response scores from all models.

    Returns:
        List of RowScore objects (one per response)
    """
    files = sorted(dir_path.glob(pattern))
    all_scores = []

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            try:
                rows = json.load(f)
            except json.JSONDecodeError:
                continue

        for row in rows:
            response = row.get("response", "")
            score = score_response(response, k)

            if score.parseable:
                all_scores.append(score)

    return all_scores

def run_welch_ttest(
    open_scores: List[RowScore],
    closed_scores: List[RowScore],
    metric_name: str,
    metric_type: str  # "binary" or "continuous"
):
    """
    Perform Welch's t-test on response-level data.

    Args:
        open_scores: All scores from open models
        closed_scores: All scores from closed models
        metric_name: Description of the metric
        metric_type: "binary" for proportions, "continuous" for ranks
    """
    if metric_type == "binary":
        # Binary outcome: AI in top-5 (0 or 1)
        open_values = [1.0 if s.ai_present_topk else 0.0 for s in open_scores]
        closed_values = [1.0 if s.ai_present_topk else 0.0 for s in closed_scores]
    elif metric_type == "continuous":
        # Continuous outcome: rank (conditional on AI present)
        open_values = [float(s.rank_score) for s in open_scores if s.ai_present_topk]
        closed_values = [float(s.rank_score) for s in closed_scores if s.ai_present_topk]
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    n_open = len(open_values)
    n_closed = len(closed_values)

    if n_open < 2 or n_closed < 2:
        print(f"\n{'='*80}")
        print(f"{metric_name}")
        print(f"{'='*80}")
        print(f"Skipped: Not enough data (open={n_open}, closed={n_closed})")
        return

    # Calculate descriptive statistics
    mean_open = np.mean(open_values)
    mean_closed = np.mean(closed_values)
    std_open = np.std(open_values, ddof=1)
    std_closed = np.std(closed_values, ddof=1)

    # Welch's t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(open_values, closed_values, equal_var=False)

    # Calculate Welch-Satterthwaite degrees of freedom
    var_open = std_open ** 2
    var_closed = std_closed ** 2
    se_open_sq = var_open / n_open
    se_closed_sq = var_closed / n_closed

    df = (se_open_sq + se_closed_sq)**2 / (se_open_sq**2 / (n_open - 1) + se_closed_sq**2 / (n_closed - 1))

    # Calculate Cohen's d (using pooled standard deviation)
    pooled_std = np.sqrt(((n_open - 1) * var_open + (n_closed - 1) * var_closed) / (n_open + n_closed - 2))
    cohens_d = (mean_closed - mean_open) / pooled_std if pooled_std > 0 else 0.0

    # Confidence intervals for means
    se_open = std_open / np.sqrt(n_open)
    se_closed = std_closed / np.sqrt(n_closed)

    t_crit_open = stats.t.ppf(0.975, n_open - 1)
    t_crit_closed = stats.t.ppf(0.975, n_closed - 1)

    ci_open_lower = mean_open - t_crit_open * se_open
    ci_open_upper = mean_open + t_crit_open * se_open
    ci_closed_lower = mean_closed - t_crit_closed * se_closed
    ci_closed_upper = mean_closed + t_crit_closed * se_closed

    # Mean difference and its CI
    mean_diff = mean_closed - mean_open
    se_diff = np.sqrt(se_open**2 + se_closed**2)

    # CI for the difference using Welch-Satterthwaite df
    t_crit_diff = stats.t.ppf(0.975, df)
    ci_diff_lower = mean_diff - t_crit_diff * se_diff
    ci_diff_upper = mean_diff + t_crit_diff * se_diff

    if mean_open != 0:
        rel_diff_pct = 100 * mean_diff / mean_open
    else:
        rel_diff_pct = float('nan')

    print(f"\n{'='*80}")
    print(f"{metric_name}")
    print(f"{'='*80}")
    print(f"Open models:   n={n_open:>6}  Mean={mean_open:.4f}  [95% CI: {ci_open_lower:.4f}, {ci_open_upper:.4f}]")
    print(f"Closed models: n={n_closed:>6}  Mean={mean_closed:.4f}  [95% CI: {ci_closed_lower:.4f}, {ci_closed_upper:.4f}]")
    print(f"Difference (Closed - Open): {mean_diff:.4f}  [95% CI: {ci_diff_lower:.4f}, {ci_diff_upper:.4f}]")
    print(f"  Relative change: {rel_diff_pct:.2f}%")
    print()
    print(f"Welch's t-test:")
    print(f"  t({df:.1f}) = {t_stat:.4f}, p = {p_val:.4g}")
    print(f"  Cohen's d = {cohens_d:.3f}")
    print()

    if p_val < 0.05:
        print("Result: Statistically Significant (p < 0.05)")
    else:
        print("Result: Not Statistically Significant")


def main():
    parser = argparse.ArgumentParser(
        description="Welch's t-test: Compare open vs closed models at the response level"
    )
    parser.add_argument("--open-dir", required=True)
    parser.add_argument("--closed-dir", required=True)
    parser.add_argument("--pattern", default="responses_*.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    open_path = Path(args.open_dir)
    closed_path = Path(args.closed_dir)

    print("=" * 80)
    print("OPEN VS CLOSED MODELS COMPARISON (WELCH'S T-TEST)")
    print("Response-level analysis (each response = 1 independent observation)")
    print("WARNING: This assumes responses are independent (ignores within-model correlation)")
    print("=" * 80)
    print()

    # Collect all individual responses
    print(f"Analyzing Open Models in: {open_path}")
    open_scores = collect_all_responses(open_path, args.pattern, args.k)
    print(f"Collected {len(open_scores)} parseable responses from open models")

    print(f"\nAnalyzing Closed Models in: {closed_path}")
    closed_scores = collect_all_responses(closed_path, args.pattern, args.k)
    print(f"Collected {len(closed_scores)} parseable responses from closed models")

    if not open_scores or not closed_scores:
        print("\nError: No parseable responses found.")
        return

    print()
    print("=" * 80)
    print("STATISTICAL TESTS (WELCH'S T-TEST)")
    print("Treating each response as an independent observation")
    print("=" * 80)

    # Test 1: P(AI in Top-K) - binary outcome
    run_welch_ttest(
        open_scores,
        closed_scores,
        f"P(AI in Top-{args.k}) — Response-Level Comparison",
        "binary"
    )

    # Test 2: E[Rank | AI present] - continuous outcome
    run_welch_ttest(
        open_scores,
        closed_scores,
        "E[Rank | AI in Top-5] — Response-Level Comparison (Conditional)",
        "continuous"
    )

    print("\n" + "=" * 80)
    print("IMPORTANT NOTES")
    print("=" * 80)
    print()
    print("This analysis treats each response as independent, which gives:")
    print(f"  - Large sample sizes (n≈{len(open_scores)} open, n≈{len(closed_scores)} closed)")
    print("  - High statistical power")
    print()
    print("HOWEVER, this approach has a critical limitation:")
    print("  - Responses from the same model are NOT independent")
    print("  - This inflates statistical power (Type I error risk)")
    print("  - Results should be interpreted with caution")
    print()
    print("For more valid inference, use eval_ai_mentions_open_vs_close.py")
    print("which aggregates to the model level (n=13 open, n=4 closed).")

if __name__ == "__main__":
    main()

