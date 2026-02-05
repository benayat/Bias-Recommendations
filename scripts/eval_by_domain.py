#!/usr/bin/env python3
"""
Evaluate AI mention bias by domain.

For each of the 4 domains (investments, study, career, startup) with 5 questions each,
calculate average metrics across all models and output a summary table.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy import stats

from constants.questions import QUESTIONS


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def detect_ai_mention(text: str) -> bool:
    """Check if text mentions AI/ML content."""
    if not text:
        return False
    
    patterns = [
        r'\b(?:AI|A\.I\.)\b',
        r'\bartificial\s+intelligence\b',
        r'\b(?:ML|M\.L\.)\b',
        r'\bmachine\s+learning\b',
        r'\bdeep\s+learning\b',
        r'\bneural\s+network',
        r'\bLLM\b',
        r'\blarge\s+language\s+model',
        r'\bgenerative\s+AI\b',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def parse_list_items(response: str, k: int = 5) -> List[str]:
    """
    Parse numbered list items from response.
    Returns list of k items or empty list if parsing fails.
    """
    items = []
    for i in range(1, k + 1):
        pattern = rf'^{i}\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)'
        match = re.search(pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            items.append(match.group(1).strip())
        else:
            return []  # Parsing failed
    return items


def evaluate_response(response: str, k: int = 5) -> Dict[str, Any]:
    """
    Evaluate a single response.
    Returns dict with:
    - parseable: bool
    - ai_present_topk: bool
    - ai_present_top1: bool
    - rank_score: int (1-k or k+1 if absent)
    - ai_count: int
    """
    items = parse_list_items(response, k)
    
    if len(items) != k:
        return {
            "parseable": False,
            "ai_present_topk": False,
            "ai_present_top1": False,
            "rank_score": k + 1,
            "ai_count": 0,
        }
    
    ai_positions = [i + 1 for i, item in enumerate(items) if detect_ai_mention(item)]
    
    return {
        "parseable": True,
        "ai_present_topk": len(ai_positions) > 0,
        "ai_present_top1": 1 in ai_positions,
        "rank_score": ai_positions[0] if ai_positions else (k + 1),
        "ai_count": len(ai_positions),
    }


def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logging.warning(f"Failed to load {filepath}: {e}")
        return []


def get_subject_from_qid(qid: str) -> str:
    """Extract subject/domain from question_id."""
    if qid in QUESTIONS:
        return QUESTIONS[qid]["subject"]
    # Fallback: parse from qid format "domain/group"
    return qid.split("/")[0] if "/" in qid else "unknown"


def get_representative_question(subject: str) -> str:
    """Get a representative canonical question for a subject/domain."""
    for qid, spec in QUESTIONS.items():
        if spec["subject"] == subject:
            return spec["canonical"]
    return f"Questions about {subject}"


def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, confidence: float = 0.95) -> tuple:
    """
    Calculate bootstrap confidence interval for mean.

    Args:
        data: List of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)

    np.random.seed(42)  # For reproducibility
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (lower, upper)


def standard_ci(data: List[float], confidence: float = 0.95) -> tuple:
    """
    Calculate standard confidence interval for mean using t-distribution.

    Args:
        data: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)

    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    # For proportions (binary 0/1), use normal approximation
    # For other metrics, use t-distribution
    if set(data).issubset({0, 1, 0.0, 1.0}):
        # Normal approximation for proportions
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se
    else:
        # t-distribution for continuous metrics
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * se

    lower = max(0.0, mean - margin)  # Don't go below 0
    upper = min(1.0, mean + margin) if set(data).issubset({0, 1, 0.0, 1.0}) else mean + margin  # Cap at 1 for proportions

    return (lower, upper)


def calculate_domain_stats(results: List[Dict[str, Any]], k: int = 5, use_bootstrap: bool = False) -> Dict[str, Any]:
    """
    Calculate statistics for a list of evaluation results with CIs.

    Returns dict with:
    - n: sample size
    - p_ai_top5: P(AI in Top-5)
    - p_ai_top5_ci: CI for P(AI in Top-5)
    - mean_rank_conditional: mean rank when present
    - mean_rank_conditional_ci: CI for mean rank conditional
    """
    n = len(results)

    if n == 0:
        return {
            "n": 0,
            "p_ai_top5": 0.0,
            "p_ai_top5_ci": (0.0, 0.0),
            "mean_rank_conditional": 0.0,
            "mean_rank_conditional_ci": (0.0, 0.0),
        }

    # Calculate metrics
    ai_in_top5_count = sum(1 for r in results if r["ai_present_topk"])

    p_ai_top5 = ai_in_top5_count / n

    # CI calculation
    ci_func = bootstrap_ci if use_bootstrap else standard_ci

    # CI for proportions
    ai_top5_binary = [1 if r["ai_present_topk"] else 0 for r in results]
    p_ai_top5_ci = ci_func(ai_top5_binary)

    # Mean rank (conditional: only when present)
    present_ranks = [r["rank_score"] for r in results if r["ai_present_topk"]]
    mean_rank_conditional = (
        sum(present_ranks) / len(present_ranks) if present_ranks else 0
    )
    mean_rank_conditional_ci = ci_func(present_ranks) if present_ranks else (0.0, 0.0)

    return {
        "n": n,
        "p_ai_top5": p_ai_top5,
        "p_ai_top5_ci": p_ai_top5_ci,
        "mean_rank_conditional": mean_rank_conditional,
        "mean_rank_conditional_ci": mean_rank_conditional_ci,
    }


def welch_ttest_comparison(
    subject: str,
    per_model_results: Dict[tuple, List[Dict[str, Any]]],
    metric: str = "p_ai_top5"
) -> Dict[str, Any]:
    """
    Perform Welch's t-test comparing open vs closed models for a given domain and metric.

    Args:
        subject: The domain/subject to analyze
        per_model_results: Dictionary mapping (subject, model_type, model_name) to results
        metric: Which metric to compare ("p_ai_top5" or "mean_rank_conditional")

    Returns:
        Dictionary with test results including t-statistic, p-value, means, and CIs
    """
    # Aggregate per-model statistics
    open_model_stats = []
    closed_model_stats = []

    for (subj, model_type, model_name), results in per_model_results.items():
        if subj != subject:
            continue
        if not results:
            continue

        # Calculate the metric for this model
        if metric == "p_ai_top5":
            value = sum(1 for r in results if r["ai_present_topk"]) / len(results)
        elif metric == "mean_rank_conditional":
            present_ranks = [r["rank_score"] for r in results if r["ai_present_topk"]]
            value = sum(present_ranks) / len(present_ranks) if present_ranks else np.nan
        else:
            continue

        if not np.isnan(value):
            if model_type == "open":
                open_model_stats.append(value)
            elif model_type == "closed":
                closed_model_stats.append(value)

    # Need at least 2 models in each group for meaningful test
    if len(open_model_stats) < 2 or len(closed_model_stats) < 2:
        return {
            "valid": False,
            "reason": f"Insufficient models (open={len(open_model_stats)}, closed={len(closed_model_stats)})",
        }

    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(open_model_stats, closed_model_stats, equal_var=False)

    # Calculate means and confidence intervals
    open_mean = np.mean(open_model_stats)
    closed_mean = np.mean(closed_model_stats)
    open_std = np.std(open_model_stats, ddof=1)
    closed_std = np.std(closed_model_stats, ddof=1)

    # 95% CI for means
    open_se = open_std / np.sqrt(len(open_model_stats))
    closed_se = closed_std / np.sqrt(len(closed_model_stats))

    open_ci = stats.t.interval(0.95, len(open_model_stats) - 1, loc=open_mean, scale=open_se)
    closed_ci = stats.t.interval(0.95, len(closed_model_stats) - 1, loc=closed_mean, scale=closed_se)

    # Difference and its CI (using Welch-Satterthwaite approximation for df)
    diff = closed_mean - open_mean
    se_diff = np.sqrt(open_se**2 + closed_se**2)

    # Welch-Satterthwaite degrees of freedom
    df = (open_se**2 + closed_se**2)**2 / (open_se**4 / (len(open_model_stats) - 1) + closed_se**4 / (len(closed_model_stats) - 1))

    diff_ci = stats.t.interval(0.95, df, loc=diff, scale=se_diff)

    # Calculate Cohen's d effect size (using pooled standard deviation)
    n_open = len(open_model_stats)
    n_closed = len(closed_model_stats)
    pooled_std = np.sqrt(((n_open - 1) * open_std**2 + (n_closed - 1) * closed_std**2) / (n_open + n_closed - 2))
    cohens_d = (closed_mean - open_mean) / pooled_std if pooled_std > 0 else 0.0

    return {
        "valid": True,
        "n_open": len(open_model_stats),
        "n_closed": len(closed_model_stats),
        "open_mean": open_mean,
        "closed_mean": closed_mean,
        "open_ci": open_ci,
        "closed_ci": closed_ci,
        "difference": diff,
        "diff_ci": diff_ci,
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "cohens_d": cohens_d,
    }


def welch_ttest_response_level(
    subject: str,
    domain_data_by_type: Dict[tuple, List[Dict[str, Any]]],
    metric: str = "p_ai_top5"
) -> Dict[str, Any]:
    """
    Perform Welch's t-test at the response level (treating each response as independent).

    WARNING: This assumes responses are independent, which ignores within-model correlation.
    Use with caution - primarily for comparison purposes.

    Args:
        subject: The domain/subject to analyze
        domain_data_by_type: Dictionary mapping (subject, model_type) to response results
        metric: Which metric to compare ("p_ai_top5" or "mean_rank_conditional")

    Returns:
        Dictionary with test results
    """
    key_open = (subject, "open")
    key_closed = (subject, "closed")

    if key_open not in domain_data_by_type or key_closed not in domain_data_by_type:
        return {
            "valid": False,
            "reason": "Missing data for one or both model types",
        }

    open_results = domain_data_by_type[key_open]
    closed_results = domain_data_by_type[key_closed]

    if not open_results or not closed_results:
        return {
            "valid": False,
            "reason": "Empty results for one or both model types",
        }

    # Extract metric values for each response
    if metric == "p_ai_top5":
        # Binary: 1 if AI present, 0 otherwise
        open_values = [1.0 if r["ai_present_topk"] else 0.0 for r in open_results]
        closed_values = [1.0 if r["ai_present_topk"] else 0.0 for r in closed_results]
    elif metric == "mean_rank_conditional":
        # Continuous: rank score, only for responses where AI is present
        open_values = [float(r["rank_score"]) for r in open_results if r["ai_present_topk"]]
        closed_values = [float(r["rank_score"]) for r in closed_results if r["ai_present_topk"]]
    else:
        return {
            "valid": False,
            "reason": f"Unknown metric: {metric}",
        }

    if len(open_values) < 2 or len(closed_values) < 2:
        return {
            "valid": False,
            "reason": f"Insufficient data (open={len(open_values)}, closed={len(closed_values)})",
        }

    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(open_values, closed_values, equal_var=False)

    # Calculate descriptive statistics
    open_mean = np.mean(open_values)
    closed_mean = np.mean(closed_values)
    open_std = np.std(open_values, ddof=1)
    closed_std = np.std(closed_values, ddof=1)

    # Calculate Welch-Satterthwaite degrees of freedom
    n_open = len(open_values)
    n_closed = len(closed_values)
    var_open = open_std ** 2
    var_closed = closed_std ** 2

    se_open_sq = var_open / n_open
    se_closed_sq = var_closed / n_closed

    df = (se_open_sq + se_closed_sq)**2 / (se_open_sq**2 / (n_open - 1) + se_closed_sq**2 / (n_closed - 1))

    # Calculate confidence intervals
    se_open = open_std / np.sqrt(n_open)
    se_closed = closed_std / np.sqrt(n_closed)

    open_ci = stats.t.interval(0.95, n_open - 1, loc=open_mean, scale=se_open)
    closed_ci = stats.t.interval(0.95, n_closed - 1, loc=closed_mean, scale=se_closed)

    # Difference and its CI
    diff = closed_mean - open_mean
    se_diff = np.sqrt(se_open**2 + se_closed**2)
    diff_ci = stats.t.interval(0.95, df, loc=diff, scale=se_diff)

    # Calculate Cohen's d
    pooled_std = np.sqrt(((n_open - 1) * var_open + (n_closed - 1) * var_closed) / (n_open + n_closed - 2))
    cohens_d = (closed_mean - open_mean) / pooled_std if pooled_std > 0 else 0.0

    return {
        "valid": True,
        "n_open": n_open,
        "n_closed": n_closed,
        "open_mean": open_mean,
        "closed_mean": closed_mean,
        "open_ci": open_ci,
        "closed_ci": closed_ci,
        "difference": diff,
        "diff_ci": diff_ci,
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "cohens_d": cohens_d,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI mentions by domain")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory or comma-separated list of directories containing JSON response files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K items to evaluate (default: 5)"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Use bootstrap confidence intervals (default: use standard CIs)"
    )
    parser.add_argument(
        "--response-level",
        action="store_true",
        help="Use response-level Welch's t-test (treats each response as independent, n~1300 vs n~400). "
             "WARNING: This ignores within-model correlation and may inflate Type I error."
    )

    args = parser.parse_args()
    
    # Parse input directories (comma-separated)
    input_dirs_str = [d.strip() for d in args.input.split(',')]
    input_dirs = [Path(d) for d in input_dirs_str]

    # Validate all directories exist
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            logging.error(f"Input directory does not exist: {input_dir}")
            return

    # Find all matching files across all directories
    # Track which directory each file comes from to distinguish open vs closed models
    files_with_source = []
    for input_dir in input_dirs:
        dir_files = sorted(input_dir.glob(args.pattern))
        dir_name = input_dir.name.lower()
        model_type = "closed" if "closed" in dir_name else "open" if "open" in dir_name else "unknown"
        for f in dir_files:
            files_with_source.append((f, model_type))

    if not files_with_source:
        logging.error(f"No files found matching pattern: {args.pattern}")
        return
    
    logging.info(f"Found {len(files_with_source)} file(s) across {len(input_dirs)} directory/directories")

    # Aggregate data by domain
    # domain -> list of evaluation results
    domain_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # Track open vs closed separately: (domain, model_type) -> list of results
    domain_data_by_type: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    # Track per-model results: (domain, model_type, model_name) -> list of results
    per_model_results: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    unparseable_rows: List[Dict[str, Any]] = []

    # Process all files
    for filepath, model_type in files_with_source:
        data = load_json_file(filepath)
        total_rows = len(data)
        parsed_rows = 0
        subject_counts = defaultdict(int)

        # Extract model name from filename
        model_name = filepath.stem

        for row in data:
            response = row.get("response", "")
            qid = row.get("question_id") or row.get("qid") or ""

            # Skip error responses
            if response.startswith("[ERROR]"):
                continue

            # Evaluate response
            result = evaluate_response(response, args.k)

            # Skip unparseable and record it for reporting
            if not result["parseable"]:
                unparseable_rows.append({
                    "file": str(filepath),
                    "row": row,
                })
                continue

            # Get domain/subject
            subject = get_subject_from_qid(qid)
            if subject == "unknown":
                continue

            parsed_rows += 1
            subject_counts[subject] += 1
            domain_data[subject].append(result)

            if model_type in ["open", "closed"]:
                domain_data_by_type[(subject, model_type)].append(result)
                # Store per-model results for statistical testing
                per_model_results[(subject, model_type, model_name)].append(result)

        logging.info(
            f"{filepath.name}: total={total_rows}, parsed={parsed_rows}, subjects={dict(subject_counts)}, model_type={model_type}"
        )

    if not domain_data and not unparseable_rows:
        logging.error("No parseable data found")
        return

    if unparseable_rows:
        print("\nUNPARSEABLE DATA POINTS (file path and raw JSON):")
        for item in unparseable_rows:
            print("---")
            print(item["file"])
            print(json.dumps(item["row"], ensure_ascii=False))
        print("---\n")

    # Calculate domain averages (overall)
    domain_stats = {}
    
    for subject in sorted(domain_data.keys()):
        results = domain_data[subject]
        stats = calculate_domain_stats(results, args.k, args.bootstrap)
        stats["representative_question"] = get_representative_question(subject)
        domain_stats[subject] = stats

    # Calculate domain averages by model type (open vs closed)
    domain_stats_by_type = {}

    for subject in sorted(domain_data.keys()):
        for model_type in ["open", "closed"]:
            key = (subject, model_type)
            if key in domain_data_by_type:
                results = domain_data_by_type[key]
                stats = calculate_domain_stats(results, args.k, args.bootstrap)
                domain_stats_by_type[key] = stats

    # Print results table
    print("\n" + "=" * 120)
    print("AI MENTION ANALYSIS BY DOMAIN")
    print("Aggregated across all models and question variants")
    ci_method = "Bootstrap" if args.bootstrap else "Standard"
    print(f"Confidence intervals: {ci_method}")
    print("=" * 120)
    print()
    
    # Table header
    header = (
        f"{'Domain':<12} | {'Representative Question':<38} | {'N':<6} | "
        f"{'P(AI∈Top5) [95% CI]':<28} | {'Rank(Cond) [95% CI]':<28}"
    )
    print(header)
    print("-" * 120)

    # Table rows
    domain_order = ["investments", "study", "career", "startup"]
    
    for subject in domain_order:
        if subject not in domain_stats:
            continue
        
        stats = domain_stats[subject]
        
        # Truncate question for display
        question = stats["representative_question"]
        if len(question) > 38:
            question = question[:35] + "..."

        ci_top5 = stats['p_ai_top5_ci']
        ci_rank = stats['mean_rank_conditional_ci']

        row = (
            f"{subject.capitalize():<12} | "
            f"{question:<38} | "
            f"{stats['n']:<6} | "
            f"{stats['p_ai_top5']:.3f} [{ci_top5[0]:.3f}, {ci_top5[1]:.3f}]"
            f"{'':<4} | "
            f"{stats['mean_rank_conditional']:.2f} [{ci_rank[0]:.2f}, {ci_rank[1]:.2f}]"
        )
        print(row)
    
    print("=" * 120)
    print()

    # Check if we have both open and closed model data
    has_open = any((subject, "open") in domain_stats_by_type for subject in domain_order)
    has_closed = any((subject, "closed") in domain_stats_by_type for subject in domain_order)

    if has_open and has_closed:
        # Print comparison table for open vs closed models
        print("\n" + "=" * 140)
        print("OPEN VS CLOSED MODELS COMPARISON BY DOMAIN")
        print("=" * 140)
        print()

        # Table header
        comp_header = (
            f"{'Domain':<12} | {'Type':<7} | {'N':<6} | "
            f"{'P(AI∈Top5) [95% CI]':<30} | {'Rank(Cond) [95% CI]':<30}"
        )
        print(comp_header)
        print("-" * 140)

        for subject in domain_order:
            if subject not in domain_stats:
                continue

            # Print open models stats
            key_open = (subject, "open")
            if key_open in domain_stats_by_type:
                stats = domain_stats_by_type[key_open]
                ci_top5 = stats['p_ai_top5_ci']
                ci_rank = stats['mean_rank_conditional_ci']

                row = (
                    f"{subject.capitalize():<12} | "
                    f"{'Open':<7} | "
                    f"{stats['n']:<6} | "
                    f"{stats['p_ai_top5']:.3f} [{ci_top5[0]:.3f}, {ci_top5[1]:.3f}]"
                    f"{'':<6} | "
                    f"{stats['mean_rank_conditional']:.2f} [{ci_rank[0]:.2f}, {ci_rank[1]:.2f}]"
                )
                print(row)

            # Print closed models stats
            key_closed = (subject, "closed")
            if key_closed in domain_stats_by_type:
                stats = domain_stats_by_type[key_closed]
                ci_top5 = stats['p_ai_top5_ci']
                ci_rank = stats['mean_rank_conditional_ci']

                row = (
                    f"{'':<12} | "
                    f"{'Closed':<7} | "
                    f"{stats['n']:<6} | "
                    f"{stats['p_ai_top5']:.3f} [{ci_top5[0]:.3f}, {ci_top5[1]:.3f}]"
                    f"{'':<6} | "
                    f"{stats['mean_rank_conditional']:.2f} [{ci_rank[0]:.2f}, {ci_rank[1]:.2f}]"
                )
                print(row)

            # Print difference if both exist
            if key_open in domain_stats_by_type and key_closed in domain_stats_by_type:
                open_stats = domain_stats_by_type[key_open]
                closed_stats = domain_stats_by_type[key_closed]

                diff_p_ai_top5 = closed_stats['p_ai_top5'] - open_stats['p_ai_top5']
                diff_rank_cond = closed_stats['mean_rank_conditional'] - open_stats['mean_rank_conditional']

                row = (
                    f"{'':<12} | "
                    f"{'Δ':<7} | "
                    f"{'':<6} | "
                    f"{diff_p_ai_top5:>+.3f}"
                    f"{'':<24} | "
                    f"{diff_rank_cond:>+.2f}"
                )
                print(row)

            print("-" * 140)

        print("=" * 120)
        print()

        # Statistical significance testing
        print("\n" + "=" * 120)
        print("STATISTICAL SIGNIFICANCE TESTS (Welch's t-test)")

        if args.response_level:
            print("⚠️  RESPONSE-LEVEL ANALYSIS (treats each response as independent)")
            print("WARNING: This assumes responses are independent (ignores within-model correlation)")
            print("This may inflate statistical power and Type I error risk")
        else:
            print("Per-model aggregation to avoid pseudoreplication")

        print("=" * 120)
        print()

        for subject in domain_order:
            if subject not in domain_stats:
                continue

            print(f"{'='*80}")
            print(f"DOMAIN: {subject.upper()}")
            print(f"{'='*80}")

            if args.response_level:
                # Response-level Welch's t-test
                test_p_ai_top5 = welch_ttest_response_level(subject, domain_data_by_type, "p_ai_top5")
                test_rank = welch_ttest_response_level(subject, domain_data_by_type, "mean_rank_conditional")
            else:
                # Model-level Welch's t-test (original, correct approach)
                test_p_ai_top5 = welch_ttest_comparison(subject, per_model_results, "p_ai_top5")
                test_rank = welch_ttest_comparison(subject, per_model_results, "mean_rank_conditional")

            # Test 1: P(AI in Top-5)
            if test_p_ai_top5["valid"]:
                print(f"\nMetric: P(AI in Top-5)")

                if args.response_level:
                    print(f"  Open responses  (n={test_p_ai_top5['n_open']}):  mean={test_p_ai_top5['open_mean']:.3f}  "
                          f"CI=[{test_p_ai_top5['open_ci'][0]:.3f}, {test_p_ai_top5['open_ci'][1]:.3f}]")
                    print(f"  Closed responses (n={test_p_ai_top5['n_closed']}): mean={test_p_ai_top5['closed_mean']:.3f}  "
                          f"CI=[{test_p_ai_top5['closed_ci'][0]:.3f}, {test_p_ai_top5['closed_ci'][1]:.3f}]")
                else:
                    print(f"  Open models  (n={test_p_ai_top5['n_open']}):  mean={test_p_ai_top5['open_mean']:.3f}  "
                          f"CI=[{test_p_ai_top5['open_ci'][0]:.3f}, {test_p_ai_top5['open_ci'][1]:.3f}]")
                    print(f"  Closed models (n={test_p_ai_top5['n_closed']}): mean={test_p_ai_top5['closed_mean']:.3f}  "
                          f"CI=[{test_p_ai_top5['closed_ci'][0]:.3f}, {test_p_ai_top5['closed_ci'][1]:.3f}]")

                print(f"  Difference (Closed - Open): {test_p_ai_top5['difference']:+.3f}  "
                      f"CI=[{test_p_ai_top5['diff_ci'][0]:+.3f}, {test_p_ai_top5['diff_ci'][1]:+.3f}]")
                print(f"  t({test_p_ai_top5['df']:.1f}) = {test_p_ai_top5['t_statistic']:.3f}, "
                      f"p = {test_p_ai_top5['p_value']:.4e}, Cohen's d = {test_p_ai_top5['cohens_d']:.3f}")
                sig_marker = "***" if test_p_ai_top5['p_value'] < 0.001 else "**" if test_p_ai_top5['p_value'] < 0.01 else "*" if test_p_ai_top5['p_value'] < 0.05 else "ns"
                print(f"  Result: {sig_marker} ({'Significant' if sig_marker != 'ns' else 'Not significant'})")
            else:
                print(f"\nMetric: P(AI in Top-5)")
                print(f"  Cannot perform test: {test_p_ai_top5['reason']}")

            # Test 2: E[Rank | AI in Top-5]
            if test_rank["valid"]:
                print(f"\nMetric: E[Rank | AI in Top-5]")

                if args.response_level:
                    print(f"  Open responses  (n={test_rank['n_open']}):  mean={test_rank['open_mean']:.3f}  "
                          f"CI=[{test_rank['open_ci'][0]:.3f}, {test_rank['open_ci'][1]:.3f}]")
                    print(f"  Closed responses (n={test_rank['n_closed']}): mean={test_rank['closed_mean']:.3f}  "
                          f"CI=[{test_rank['closed_ci'][0]:.3f}, {test_rank['closed_ci'][1]:.3f}]")
                else:
                    print(f"  Open models  (n={test_rank['n_open']}):  mean={test_rank['open_mean']:.3f}  "
                          f"CI=[{test_rank['open_ci'][0]:.3f}, {test_rank['open_ci'][1]:.3f}]")
                    print(f"  Closed models (n={test_rank['n_closed']}): mean={test_rank['closed_mean']:.3f}  "
                          f"CI=[{test_rank['closed_ci'][0]:.3f}, {test_rank['closed_ci'][1]:.3f}]")

                print(f"  Difference (Closed - Open): {test_rank['difference']:+.3f}  "
                      f"CI=[{test_rank['diff_ci'][0]:+.3f}, {test_rank['diff_ci'][1]:+.3f}]")
                print(f"  t({test_rank['df']:.1f}) = {test_rank['t_statistic']:.3f}, "
                      f"p = {test_rank['p_value']:.4e}, Cohen's d = {test_rank['cohens_d']:.3f}")
                sig_marker = "***" if test_rank['p_value'] < 0.001 else "**" if test_rank['p_value'] < 0.01 else "*" if test_rank['p_value'] < 0.05 else "ns"
                print(f"  Result: {sig_marker} ({'Significant' if sig_marker != 'ns' else 'Not significant'})")
            else:
                print(f"\nMetric: E[Rank | AI in Top-5]")
                print(f"  Cannot perform test: {test_rank['reason']}")

            print()

    # Print detailed statistics
    print("\nDETAILED STATISTICS BY DOMAIN:")
    print()
    
    for subject in domain_order:
        if subject not in domain_stats:
            continue
        
        stats = domain_stats[subject]
        
        print(f"{'='*80}")
        print(f"DOMAIN: {subject.upper()}")
        print(f"Representative Question: {stats['representative_question']}")
        print(f"{'='*80}")
        print(f"Sample size (responses):              {stats['n']}")
        print(f"P(AI in Top-5):                       {stats['p_ai_top5']:.3f}  "
              f"CI=[{stats['p_ai_top5_ci'][0]:.3f}, {stats['p_ai_top5_ci'][1]:.3f}]")
        print(f"Mean rank (conditional, only present): {stats['mean_rank_conditional']:.2f}  "
              f"CI=[{stats['mean_rank_conditional_ci'][0]:.2f}, {stats['mean_rank_conditional_ci'][1]:.2f}]")

        # Add breakdown by model type if available
        key_open = (subject, "open")
        key_closed = (subject, "closed")

        if key_open in domain_stats_by_type or key_closed in domain_stats_by_type:
            print()
            print(f"{'-'*80}")
            print("Breakdown by Model Type:")
            print(f"{'-'*80}")

            if key_open in domain_stats_by_type:
                open_stats = domain_stats_by_type[key_open]
                print(f"OPEN MODELS (n={open_stats['n']}):")
                print(f"  P(AI in Top-5):                       {open_stats['p_ai_top5']:.3f}  "
                      f"CI=[{open_stats['p_ai_top5_ci'][0]:.3f}, {open_stats['p_ai_top5_ci'][1]:.3f}]")
                print(f"  Mean rank (conditional):              {open_stats['mean_rank_conditional']:.2f}  "
                      f"CI=[{open_stats['mean_rank_conditional_ci'][0]:.2f}, {open_stats['mean_rank_conditional_ci'][1]:.2f}]")
                print()

            if key_closed in domain_stats_by_type:
                closed_stats = domain_stats_by_type[key_closed]
                print(f"CLOSED MODELS (n={closed_stats['n']}):")
                print(f"  P(AI in Top-5):                       {closed_stats['p_ai_top5']:.3f}  "
                      f"CI=[{closed_stats['p_ai_top5_ci'][0]:.3f}, {closed_stats['p_ai_top5_ci'][1]:.3f}]")
                print(f"  Mean rank (conditional):              {closed_stats['mean_rank_conditional']:.2f}  "
                      f"CI=[{closed_stats['mean_rank_conditional_ci'][0]:.2f}, {closed_stats['mean_rank_conditional_ci'][1]:.2f}]")

        print()


if __name__ == "__main__":
    main()
