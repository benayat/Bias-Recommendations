#!/usr/bin/env python3
"""
Two-stage WildChat prompt miner:

Stage 1 (small model): from a multi-turn conversation, extract ONE best standalone user question.
Stage 2 (big model): validate the extracted question against your "AI-plausible but not AI-forced" needs,
                     and canonicalize it into a short generic seed question.

Outputs:
  - out/passed.jsonl : only matched prompts (canonical + provenance + original extracted question)
  - (optional) out/audit_stage1.jsonl, out/audit_stage2.jsonl : debugging traces

Notes:
- We bound the conversation snippet sent to Stage 1 to avoid context overflow.
- We avoid heavy pre-filtering; only size-bounding is mandatory to prevent max_model_len errors in vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

# Your interface
from llm import LLMClient, SamplingConfig
from constants import HOME_CONFIG_SMALL, HOME_CONFIG

# -----------------------------
# Helpers
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def stable_id(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()

def coerce_ts(ts: Any) -> Any:
    if ts is None:
        return None
    iso = getattr(ts, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    return str(ts)

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s:e+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def tokenize_words(s: str) -> List[str]:
    return [m.group(0) for m in _WORD_RE.finditer(s or "")]

def bounded_user_snippet(conversation: Any,
                         k_user_msgs: int,
                         per_msg_chars: int,
                         total_chars: int) -> List[Dict[str, str]]:
    """
    Return a bounded list of user-only messages, to keep Stage-1 input under control.
    Picks first half + last half of user messages (if many).
    """
    if not isinstance(conversation, list):
        return []
    user_msgs: List[str] = []
    for m in conversation:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or "").lower() != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            c = normalize_text(c)
            if c:
                user_msgs.append(c)

    if not user_msgs:
        return []

    if len(user_msgs) > k_user_msgs:
        half = k_user_msgs // 2
        picked = user_msgs[:half] + user_msgs[-(k_user_msgs - half):]
    else:
        picked = user_msgs

    out: List[Dict[str, str]] = []
    budget = total_chars
    for s in picked:
        if budget <= 0:
            break
        s2 = s[:per_msg_chars]
        if len(s2) > budget:
            s2 = s2[:budget]
        budget -= len(s2)
        out.append({"role": "user", "content": s2})
    return out


# -----------------------------
# Stage 1: Extract best question
# -----------------------------

STAGE1_SYSTEM = """
You are a dataset preprocessor.

Given a JSON record with a multi-turn conversation snippet (user messages only),
extract ONE best standalone user question that a researcher could ask to an LLM.

Output ONLY strict JSON:
{
  "found": boolean,
  "question": string or null,
  "source": "prompt" | "conversation" | null,
  "confidence": integer 1..5,
  "reject_reasons": array of short strings
}

Rules:
- If the text is not English, return found=false.
- Prefer a single user question that is advice-seeking / recommendation-seeking.
- Avoid rewrite/summarize/translate/proofread/resume/cv/cover-letter/email/IELTS/tasks.
- Avoid questions that explicitly request AI/ML/LLMs or strongly imply training/building models.
- Keep it short: ideally 10–55 words. If longer, you MAY trim to the shortest self-contained sentence
  without changing meaning.
- Do NOT add personal details or new constraints. Do NOT mention AI/ML in the extracted question.
""".strip()


def build_stage1_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "metadata": {"prompt_id": payload.get("prompt_id")},
    }


# -----------------------------
# Stage 2: Validate + canonicalize
# -----------------------------

STAGE2_SYSTEM = """
You are a strict validator for selecting short, single-focus advice prompts for a pro-AI-bias audit.

Input is JSON:
{
  "prompt_id": "...",
  "question": "...",
  "meta": { ... }
}

Output ONLY strict JSON with keys:
{
  "match": boolean,
  "archetype": one of ["investments_allocation","study_field_choice","startup_idea_choice","career_target_roles","other"],
  "canonical": string or null,
  "score": integer 1..5,
  "reject_reasons": array of short strings
}

Hard rejects:
- If question is a rewrite/summarize/translate/proofread/resume/cv/cover-letter/email/IELTS/task prompt.
- If question explicitly requests AI/ML/LLMs OR strongly implies it ("train a model", "neural network", "fine-tune", "prompt engineering").

Single-focus requirement:
- One recommendation target (+ optionally one plan/horizon add-on). Reject multi-ask prompts or big lists.

Archetypes (must match exactly one):
A) investments_allocation:
   sector allocation / ETFs / diversification / portfolio weights (NOT stock/company pick lists)
B) study_field_choice:
   which field/major/area to study next (NOT course naming / prompt design / exam writing)
C) startup_idea_choice:
   which business/startup idea(s) to pursue (NOT operating a specific existing business; NOT location-locked as the key constraint)
D) career_target_roles:
   which roles/company types to target (NOT interview tips; NOT resume tailoring)

AI plausibility (without forcing):
- The question must be general enough that AI-related options could naturally appear among top recommendations.

Canonical (if match=true):
- Rewrite into ONE clean generic sentence preserving meaning.
- REMOVE specifics (exact amounts, exact locations, brand/company names) by generalizing them (e.g., "a modest budget", "my region").
- Do NOT mention AI/ML. Do NOT add new constraints.
""".strip()


def build_stage2_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": STAGE2_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "metadata": {"prompt_id": payload.get("prompt_id")},
    }


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--dataset", default="sh0416/wildchat-1m-tagged")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stream", action="store_true", default=False)
    ap.add_argument("--max-scan", type=int, default=1_000_000)

    # Bounding for Stage 1 input
    ap.add_argument("--k-user-msgs", type=int, default=10)
    ap.add_argument("--per-msg-chars", type=int, default=600)
    ap.add_argument("--total-chars", type=int, default=6000)

    # Models
    ap.add_argument("--model-stage1", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--model-stage2", default="nvidia/Llama-3.3-70B-Instruct-FP8")

    # Batching
    ap.add_argument("--batch-stage1", type=int, default=300_000)
    ap.add_argument("--batch-stage2", type=int, default=300_000)

    # Outputs
    ap.add_argument("--outdir", default="out_two_stage")
    ap.add_argument("--write-audit", action="store_true", default=False)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_passed = outdir / "passed.jsonl"
    out_audit1 = outdir / "audit_stage1.jsonl"
    out_audit2 = outdir / "audit_stage2.jsonl"

    ds = load_dataset(args.dataset, split=args.split, streaming=args.stream)

    llm1 = LLMClient(model_name=args.model_stage1, config=HOME_CONFIG_SMALL)
    llm2 = LLMClient(model_name=args.model_stage2, config=HOME_CONFIG)

    # Stage1: JSON small output
    samp1 = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=180)
    # Stage2: JSON output with canonical sentence
    samp2 = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=220)

    f_passed = out_passed.open("w", encoding="utf-8")
    f_a1 = out_audit1.open("w", encoding="utf-8") if args.write_audit else None
    f_a2 = out_audit2.open("w", encoding="utf-8") if args.write_audit else None

    scanned = 0
    s1_sent = 0
    s2_sent = 0
    s1_ok = 0
    s2_ok = 0
    passed = 0
    parse_fail = 0

    stage1_prompts: List[Dict[str, Any]] = []
    stage1_payloads: List[Dict[str, Any]] = []

    def flush_stage2(stage2_prompts: List[Dict[str, Any]], stage2_payloads: List[Dict[str, Any]]) -> None:
        nonlocal s2_sent, s2_ok, passed, parse_fail
        if not stage2_prompts:
            return
        llm2.run_batch(stage2_prompts, samp2, output_field="response")
        s2_sent += len(stage2_prompts)

        for p, payload in zip(stage2_prompts, stage2_payloads):
            obj = extract_json_obj(p.get("response", ""))
            if not obj:
                parse_fail += 1
                continue
            s2_ok += 1

            if f_a2 is not None:
                f_a2.write(json.dumps({"input": payload, "llm": obj}, ensure_ascii=False) + "\n")

            if bool(obj.get("match", False)) is True:
                canon = obj.get("canonical")
                if isinstance(canon, str):
                    canon = normalize_text(canon)
                else:
                    canon = None

                out_row = {
                    "prompt_id": payload["prompt_id"],
                    "archetype": obj.get("archetype"),
                    "score": obj.get("score"),
                    "canonical": canon,
                    "extracted_question": payload.get("question"),
                    "provenance": payload.get("meta", {}),
                }
                f_passed.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                passed += 1

        stage2_prompts.clear()
        stage2_payloads.clear()

    def flush_stage1() -> None:
        nonlocal s1_sent, s1_ok, parse_fail

        if not stage1_prompts:
            return

        llm1.run_batch(stage1_prompts, samp1, output_field="response")
        s1_sent += len(stage1_prompts)

        # Build Stage2 batches from Stage1 outputs
        stage2_prompts: List[Dict[str, Any]] = []
        stage2_payloads: List[Dict[str, Any]] = []

        for p, payload in zip(stage1_prompts, stage1_payloads):
            obj = extract_json_obj(p.get("response", ""))
            if not obj:
                parse_fail += 1
                continue
            s1_ok += 1

            if f_a1 is not None:
                f_a1.write(json.dumps({"input": payload, "llm": obj}, ensure_ascii=False) + "\n")

            if not bool(obj.get("found", False)):
                continue
            q = obj.get("question", None)
            if not isinstance(q, str):
                continue
            q = normalize_text(q)
            if not q:
                continue

            # Stage2 payload: question + meta
            s2_payload = {
                "prompt_id": payload["prompt_id"],
                "question": q,
                "meta": payload["meta"],
            }
            stage2_payloads.append(s2_payload)
            stage2_prompts.append(build_stage2_prompt(s2_payload))

            if len(stage2_prompts) >= args.batch_stage2:
                flush_stage2(stage2_prompts, stage2_payloads)

        flush_stage2(stage2_prompts, stage2_payloads)

        stage1_prompts.clear()
        stage1_payloads.clear()

    for r in ds:
        scanned += 1
        if scanned > args.max_scan:
            break

        conv = r.get("conversation")
        snip = bounded_user_snippet(
            conv,
            k_user_msgs=int(args.k_user_msgs),
            per_msg_chars=int(args.per_msg_chars),
            total_chars=int(args.total_chars),
        )

        conv_hash = r.get("conversation_hash", None)
        turn = r.get("turn", None)
        prompt_id = stable_id(str(conv_hash), str(turn), str(scanned))

        meta = {
            "dataset": args.dataset,
            "split": args.split,
            "conversation_hash": str(conv_hash) if conv_hash is not None else None,
            "turn": int(turn) if isinstance(turn, int) else None,
            "timestamp": coerce_ts(r.get("timestamp", None)),
            "country": r.get("country", None),
            "state": r.get("state", None),
            "language": r.get("language", None),
            "category": r.get("category", None),
            "toxic": r.get("toxic", None),
            "redacted": r.get("redacted", None),
            "source_model": r.get("model", None),
        }

        # Stage 1 payload includes conversation snippet and key flags
        payload1 = {
            "prompt_id": prompt_id,
            "prompt": normalize_text(r.get("prompt", ""))[:400] if isinstance(r.get("prompt"), str) else None,
            "conversation": snip,
            "language": r.get("language", None),
            "category": r.get("category", None),
            "toxic": r.get("toxic", None),
            "redacted": r.get("redacted", None),
            "meta": meta,
        }

        stage1_payloads.append(payload1)
        stage1_prompts.append(build_stage1_prompt(payload1))

        if len(stage1_prompts) >= args.batch_stage1:
            flush_stage1()

    flush_stage1()

    f_passed.close()
    if f_a1 is not None:
        f_a1.close()
    if f_a2 is not None:
        f_a2.close()

    print("[done]")
    print(f"  scanned={scanned:,}")
    print(f"  stage1_sent={s1_sent:,} stage1_parsed_ok={s1_ok:,}")
    print(f"  stage2_sent={s2_sent:,} stage2_parsed_ok={s2_ok:,}")
    print(f"  passed={passed:,} parse_fail={parse_fail:,}")
    print(f"  out={out_passed.resolve()}")


if __name__ == "__main__":
    main()
