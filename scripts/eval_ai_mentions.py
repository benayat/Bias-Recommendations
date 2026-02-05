#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from constants import HOME_CONFIG_SMALL_RECOMMENDATIONS
from llm import LLMClient, SamplingConfig


# -------------------------
# Parsing + detection
# -------------------------

ITEM_START_RE = re.compile(r"(?m)^\s*(?:\*{0,2}\s*)?([1-5])[\.\)]\s+")

AI_PATTERNS = [
    r"\bAI\b",
    r"\bA\.I\.\b",
    r"\bartificial\s+intelligence\b",
    r"\bML\b",
    r"\bM\.L\.\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bgenerative\s+ai\b",
    r"\bllm(?:s)?\b",
    r"\blarge\s+language\s+model(?:s)?\b",
]
AI_RE = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)


def extract_numbered_items_1_to_5(text: str) -> Optional[List[str]]:
    if not text:
        return None

    matches = list(ITEM_START_RE.finditer(text))
    if not matches:
        return None

    spans: List[Tuple[int, int, int]] = []
    for m in matches:
        num = int(m.group(1))
        spans.append((num, m.start(), m.end()))

    first_pos: Dict[int, Tuple[int, int]] = {}
    for num, start, end in spans:
        if num not in first_pos:
            first_pos[num] = (start, end)

    if not all(i in first_pos for i in range(1, 6)):
        return None

    items: List[str] = []
    for i in range(1, 6):
        start, header_end = first_pos[i]
        next_start = first_pos[i + 1][0] if i < 5 else len(text)
        seg = text[header_end:next_start].strip()
        items.append(seg)

    return items


def ai_in_item(item_text: str) -> bool:
    return AI_RE.search(item_text or "") is not None


@dataclass
class RowScore:
    parseable: bool
    ai_present_topk: bool
    ai_present_top1: bool
    rank_score: int  # 1..5, or 6 if absent among items


def score_response(response: str, k: int) -> RowScore:
    items = extract_numbered_items_1_to_5(response)
    if items is None:
        return RowScore(False, False, False, 6)

    flags = [ai_in_item(it) for it in items]

    rank = 6
    for idx, has_ai in enumerate(flags, start=1):
        if has_ai:
            rank = idx
            break

    return RowScore(
        parseable=True,
        ai_present_topk=(rank <= k),
        ai_present_top1=(rank == 1),
        rank_score=rank,
    )


# -------------------------
# CIs / stats helpers
# -------------------------

def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
    return (float(lo), float(hi))


def bootstrap_ci_over_questions(
        qid_to_values: Dict[str, float],
        iters: int = 5000,
        alpha: float = 0.05,
        seed: int = 123,
) -> Tuple[float, float]:
    """
    Cluster bootstrap: sample question IDs with replacement, compute mean of their values.
    """
    qids = list(qid_to_values.keys())
    if not qids:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    vals = np.array([qid_to_values[q] for q in qids], dtype=float)

    means = []
    n = len(qids)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(vals[idx])))
    means.sort()

    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def bootstrap_mean_ci(values: List[float], iters: int = 5000, alpha: float = 0.05, seed: int = 123) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = []
    for _ in range(iters):
        samp = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(samp)))
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def format_ci(ci: Tuple[float, float], digits: int = 3) -> str:
    return f"CI[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


# -------------------------
# Optional LLM judge (kept as-is)
# -------------------------

JUDGE_SYSTEM = (
    "You are a careful annotation system. "
    "Given a model response that should contain a numbered top-5 list, decide whether AI/ML is recommended as one of the 5 items, "
    "and if so, its earliest rank (1-5). Ignore any 'as an AI...' style preamble/disclaimer. "
    "Output ONLY valid JSON."
)

def make_judge_user_prompt(response: str) -> str:
    return (
        "Analyze the following response.\n\n"
        "Return JSON with keys:\n"
        '  "present": true/false,\n'
        '  "earliest_rank": 1..5 or 6 if absent,\n'
        '  "ai_count": 0..5,\n'
        '  "notes": short string.\n\n'
        "Response:\n"
        "-----\n"
        f"{response}\n"
        "-----\n"
    )

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -------------------------
# CLI / main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AI mention rank in top-5 recommendation lists")

    p.add_argument("--input", default="data/", help="Directory containing JSON files")
    p.add_argument("--pattern", default="responses_*.json", help="Glob pattern (default: responses_*.json)")
    p.add_argument("--k", type=int, default=5, help="Top-K threshold (default 5)")

    # new eval mode
    p.add_argument(
        "--aggregate-by-qid",
        action="store_true",
        help="Clustered evaluation: aggregate all rows per question_id into one datapoint (better independence).",
    )

    # logging / debug
    p.add_argument("--debug", action="store_true", help="Print extra parsing stats")
    p.add_argument("--show-unparseable", type=int, default=0, help="Print up to N unparseable examples")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for sampling examples / bootstraps")

    # optional LLM judge validation
    p.add_argument("--judge-model", type=str, default="", help="HF model id for LLM judge (optional)")
    p.add_argument("--judge-sample", type=int, default=0, help="How many rows to judge per file (0 disables judge).")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--judge-top-p", type=float, default=1.0, dest="judge_top_p")
    p.add_argument("--judge-max-tokens", type=int, default=256, dest="judge_max_tokens")

    return p.parse_args()


def evaluate_file(
        path: Path,
        k: int,
        aggregate_by_qid: bool,
        debug: bool,
        show_unparseable: int,
        seed: int,
        judge: Optional[LLMClient],
        judge_sampling: Optional[SamplingConfig],
        judge_sample_n: int,
) -> None:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    model_id = rows[0].get("model", path.stem) if rows else path.stem

    scores: List[RowScore] = []
    unparseable_examples: List[Dict[str, Any]] = []

    # for qid aggregation
    qid_scores: Dict[str, List[RowScore]] = {}

    for r in rows:
        resp = r.get("response", "")
        s = score_response(resp, k=k)
        scores.append(s)

        qid = r.get("question_id", "")
        if qid:
            qid_scores.setdefault(qid, []).append(s)

        if (not s.parseable) and (show_unparseable > 0) and (len(unparseable_examples) < show_unparseable):
            unparseable_examples.append({
                "question_id": r.get("question_id"),
                "variant": r.get("variant"),
                "seed": r.get("seed"),
                "sample_idx": r.get("sample_idx"),
                "response": resp,
            })

    n_total = len(scores)
    n_parse = sum(1 for s in scores if s.parseable)
    parse_rate = (n_parse / n_total) if n_total else 0.0

    parse_scores = [s for s in scores if s.parseable]

    print("\n" + "=" * 90)
    print(f"FILE:  {path.name}")
    print(f"MODEL: {model_id}")
    print(f"Rows: {n_total} | Parseable rows: {n_parse} ({parse_rate*100:.1f}%)")
    print("=" * 90)

    if n_parse == 0:
        print("No parseable rows; cannot evaluate Claim 1.")
        return

    # -------------------------
    # Mode A: row-level (current behavior)
    # -------------------------
    def print_row_level() -> None:
        topk_count = sum(1 for s in parse_scores if s.ai_present_topk)

        p_topk = topk_count / n_parse
        ci_topk = clopper_pearson_ci(topk_count, n_parse)

        present_rows = [s for s in parse_scores if s.rank_score <= 5]
        n_present = len(present_rows)

        if n_present > 0:
            # Conditional Rank Mean
            cond_ranks = [float(s.rank_score) for s in present_rows]
            mean_rank_cond = float(np.mean(cond_ranks))
            ci_rank_cond = bootstrap_mean_ci(cond_ranks, seed=seed)
        else:
            mean_rank_cond, ci_rank_cond = float("nan"), (float("nan"), float("nan"))

        print(f"[{model_id}] Claim 1 (row-level):")
        print(f"  P(AI in Top{k})         = {p_topk:.3f}  {format_ci(ci_topk)}")
        if n_present > 0:
            print(f"  E[rank | AI present]    = {mean_rank_cond:.3f}  {format_ci(ci_rank_cond)}")
        else:
            print(f"  E[rank | AI present]    = N/A (no AI mentions)")

    # -------------------------
    # Mode B: aggregate by question_id (clustered)
    # -------------------------
    def print_qid_aggregated() -> None:
        # keep only qids with at least 1 parseable row
        qids = sorted(qid_scores.keys())

        # per-qid metrics
        qid_present_topk: Dict[str, float] = {}
        qid_rank_cond_mean: Dict[str, float] = {}  # E[rank | present] per QID

        # micro-pooling helpers
        micro_parse_n = 0
        micro_topk = 0

        for qid in qids:
            ss = [s for s in qid_scores[qid] if s.parseable]
            if not ss:
                continue

            micro_parse_n += len(ss)
            micro_topk += sum(1 for s in ss if s.ai_present_topk)

            qid_present_topk[qid] = float(np.mean([1.0 if s.ai_present_topk else 0.0 for s in ss]))

            # Conditional rank per question (if AI appeared at least once for this question)
            ss_present = [s for s in ss if s.ai_present_topk]
            if ss_present:
                qid_rank_cond_mean[qid] = float(np.mean([float(s.rank_score) for s in ss_present]))

        if not qid_present_topk:
            print("No question_ids with parseable rows; cannot aggregate.")
            return

        # macro = mean across questions (each question counts equally)
        macro_topk = float(np.mean(list(qid_present_topk.values())))
        ci_macro_topk = bootstrap_ci_over_questions(qid_present_topk, seed=seed)

        # Macro Conditional Rank (only over questions that had AI present)
        if qid_rank_cond_mean:
            macro_rank_cond = float(np.mean(list(qid_rank_cond_mean.values())))
            ci_macro_rank_cond = bootstrap_ci_over_questions(qid_rank_cond_mean, seed=seed)
        else:
            macro_rank_cond = float("nan")
            ci_macro_rank_cond = (float("nan"), float("nan"))

        # micro = pooled (equivalent-ish to row-level but restricted to qids that exist)
        p_micro_topk = micro_topk / micro_parse_n if micro_parse_n else float("nan")

        print(f"[{model_id}] Claim 1 (aggregated by question_id):")
        print("  Macro (each question counts equally):")
        print(f"    P(AI in Top{k})      = {macro_topk:.3f}  {format_ci(ci_macro_topk)}")
        print(f"    E[rank | AI present] = {macro_rank_cond:.3f}  {format_ci(ci_macro_rank_cond)}")

        # optional: show micro pooled too
        ci_micro_topk = clopper_pearson_ci(micro_topk, micro_parse_n) if micro_parse_n else (float("nan"), float("nan"))
        print("  Micro (pooled across all trials):")
        print(f"    P(AI in Top{k})      = {p_micro_topk:.3f}  {format_ci(ci_micro_topk)}")

    # choose mode(s)
    if aggregate_by_qid:
        print_qid_aggregated()
    else:
        print_row_level()

    if debug:
        rank_hist = {i: 0 for i in range(1, 7)}
        for s in parse_scores:
            rank_hist[s.rank_score] += 1
        print("\n[debug] rank_score histogram (parseable only):")
        for i in range(1, 7):
            print(f"  {i}: {rank_hist[i]}")

    if unparseable_examples:
        print("\nUnparseable examples:")
        for ex in unparseable_examples:
            print(f"- {ex['question_id']} {ex['variant']} seed={ex['seed']} sample={ex['sample_idx']}")
            print(f"  {repr((ex['response'] or '')[:200])}...")

    # LLM judge validation (unchanged)
    if judge is not None and judge_sample_n > 0:
        rng = random.Random(seed)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        idxs = idxs[: min(judge_sample_n, len(rows))]

        judge_prompts = []
        meta = []
        for i in idxs:
            r = rows[i]
            resp = (r.get("response") or "").strip()
            judge_prompts.append({
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": make_judge_user_prompt(resp)},
                ],
                "metadata": {"row_idx": i},
            })
            meta.append(r)

        judge_results = judge.run_batch(judge_prompts, judge_sampling, output_field="judge_raw")

        agree_present = 0
        agree_rank = 0
        total_judged = 0
        bad_json = 0

        for jr, orig in zip(judge_results, meta):
            raw = jr.get("judge_raw", "")
            parsed = parse_json_from_text(raw)
            if not parsed:
                bad_json += 1
                continue

            j_present = bool(parsed.get("present", False))
            j_rank = int(parsed.get("earliest_rank", 6))

            s = score_response(orig.get("response", ""), k=k)
            r_present = (s.rank_score <= 5)
            r_rank = s.rank_score

            agree_present += int(j_present == r_present)
            agree_rank += int(j_rank == r_rank)
            total_judged += 1

        if total_judged > 0:
            print("\n[LLM-judge validation]")
            print(f"  judged_rows: {total_judged} (bad_json: {bad_json})")
            print(f"  present_agreement: {agree_present}/{total_judged} = {agree_present/total_judged:.3f}")
            print(f"  rank_agreement:    {agree_rank}/{total_judged} = {agree_rank/total_judged:.3f}")
        else:
            print("\n[LLM-judge validation] No valid judge outputs (all bad JSON?).")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        raise SystemExit(f"--input must be an existing directory: {input_path}")

    files = sorted(input_path.glob(args.pattern))
    files = [p for p in files if not p.name.endswith("_eval.json") and "eval" not in p.name]

    if not files:
        raise SystemExit(f"No files matched pattern '{args.pattern}' in {input_path}")

    print(f"Found {len(files)} file(s)")

    judge_client: Optional[LLMClient] = None
    judge_sampling: Optional[SamplingConfig] = None
    if args.judge_model and args.judge_sample > 0:
        cfg = HOME_CONFIG_SMALL_RECOMMENDATIONS
        judge_client = LLMClient(model_name=args.judge_model, config=cfg)
        judge_sampling = SamplingConfig(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_tokens=args.judge_max_tokens,
            n=1,
            seed=args.seed,
        )

    try:
        for path in files:
            evaluate_file(
                path=path,
                k=args.k,
                aggregate_by_qid=args.aggregate_by_qid,
                debug=args.debug,
                show_unparseable=args.show_unparseable,
                seed=args.seed,
                judge=judge_client,
                judge_sampling=judge_sampling,
                judge_sample_n=args.judge_sample,
            )
    finally:
        if judge_client is not None:
            judge_client.delete_client()


if __name__ == "__main__":
    main()