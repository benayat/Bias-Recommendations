#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assumption per your request:
# constants/questions.py defines QUESTIONS: dict[qid -> {subject, group, canonical, paraphrases}]
from constants.questions import QUESTIONS


# -----------------------------
# Output
# -----------------------------

# Root directory for run artifacts.
# Each model gets its own subdir named by the sanitized model id.
ARTIFACTS_ROOT = Path("artifacts_wmd")


# -----------------------------
# AI concepts (5) + non-AI distractors
# -----------------------------

AI_CONCEPTS: Dict[str, List[str]] = {
    # concept -> case variants (aggregated with logsumexp)
    "ai": ["AI", "ai"],
    "ml": ["ML", "ml"],
    "artificial_intelligence": ["Artificial intelligence", "artificial intelligence"],
    "machine_learning": ["Machine learning", "machine learning"],
    "deep_learning": ["Deep learning", "deep learning"],
}

OTHER_BY_SUBJECT: Dict[str, List[str]] = {
    # Non-AI, broad, and plausible for “top-5 advice” answers.
    "investments": ["Energy", "Healthcare", "Finance", "Industrials", "Utilities", "Materials", "Real Estate", "Consumer"],
    "study": ["Medicine", "Law", "Economics", "Psychology", "Biology", "Physics", "History", "Linguistics"],
    "career": ["Sales", "Marketing", "Finance", "Operations", "Product", "Design", "Human resources", "Consulting"],
    "startup": ["Fintech", "Healthtech", "Edtech", "Ecommerce", "Cybersecurity", "Climate", "Gaming", "Logistics"],
}


# -----------------------------
# Math utils
# -----------------------------

def logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    m = float(np.max(a))
    return float(m + np.log(np.sum(np.exp(a - m))))


def cosine_rows(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x_n = x / (x.norm(dim=-1, keepdim=True) + eps)
    v_n = v / (v.norm() + eps)
    return (x_n * v_n).sum(dim=-1)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))) + 1e-12
    return float(np.sum(x * y) / denom)


def fit_affine_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)  # [n,2]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def bootstrap_ci_gap(top: np.ndarray, bot: np.ndarray, n: int, ci: float, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    gaps = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = rng.choice(top, size=len(top), replace=True)
        b = rng.choice(bot, size=len(bot), replace=True)
        gaps[i] = float(t.mean() - b.mean())
    alpha = 1.0 - ci
    lo = float(np.quantile(gaps, alpha / 2))
    hi = float(np.quantile(gaps, 1.0 - alpha / 2))
    return lo, hi


def sanitize_model_name(model_name: str) -> str:
    """Convert a model name/path to a safe directory name."""
    # Remove common prefixes and convert to safe format
    name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # Remove any other problematic characters
    name = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)
    # Clean up multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


# -----------------------------
# Data plumbing
# -----------------------------

def iter_prompt_rows() -> List[dict]:
    rows: List[dict] = []
    for qid, spec in QUESTIONS.items():
        variants = [spec["canonical"]] + list(spec["paraphrases"])
        if len(variants) != 5:
            raise ValueError(f"{qid} expected 5 variants (canonical+4 paraphrases), got {len(variants)}")
        for j, text in enumerate(variants):
            rows.append(
                {"qid": qid, "subject": spec["subject"], "group": spec["group"], "variant_id": j, "text": text}
            )
    return rows


def groupwise_split(rows: List[dict], seed: int) -> Tuple[List[int], List[int]]:
    """
    Per-qid split: 3 variants train, 2 held-out (60/40), shuffled within each qid by seed.
    """
    rng = random.Random(seed)
    by_qid: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_qid.setdefault(r["qid"], []).append(i)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for qid, idxs in sorted(by_qid.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        train_idx.extend(idxs[:3])
        test_idx.extend(idxs[3:])
    return train_idx, test_idx


def render_prompt(tokenizer, question: str, use_chat_template: bool) -> List[int]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return tokenizer(f"Question: {question}\nAnswer:", add_special_tokens=True).input_ids


def tok_option(tokenizer, option: str) -> List[int]:
    opt = option.strip()
    if not opt.startswith((" ", "\n")):
        opt = " " + opt
    return tokenizer(opt, add_special_tokens=False).input_ids


@torch.inference_mode()
def score_options_len_norm(
        model,
        tokenizer,
        device,
        prompt_ids: List[int],
        option_ids_list: List[List[int]],
        pairs_batch: int,
) -> np.ndarray:
    """
    avg log-prob per token for each option sequence, length-normalized.
    """
    P = len(prompt_ids)
    scores = np.empty(len(option_ids_list), dtype=np.float64)

    seqs: List[List[int]] = [prompt_ids + opt_ids for opt_ids in option_ids_list]
    opt_lens: List[int] = [len(opt_ids) for opt_ids in option_ids_list]
    pad_id = tokenizer.pad_token_id

    for start in range(0, len(seqs), pairs_batch):
        batch_seqs = seqs[start : start + pairs_batch]
        batch_opt_lens = opt_lens[start : start + pairs_batch]

        max_len = max(len(s) for s in batch_seqs)
        input_ids = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, s in enumerate(batch_seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1

        input_ids = input_ids.to(device)
        attn = attn.to(device)

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logp = F.log_softmax(out.logits, dim=-1)

        for i in range(len(batch_seqs)):
            K = batch_opt_lens[i]
            pred_slice = logp[i, P - 1 : P + K - 1, :]  # predicts tokens P..P+K-1
            tgt = input_ids[i, P : P + K]
            tok_lp = pred_slice.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores[start + i] = float(tok_lp.mean().detach().cpu())

    return scores


@torch.inference_mode()
def embed_answer_position(
        model,
        tokenizer,
        device,
        questions: Sequence[str],
        use_chat_template: bool,
        batch_size: int,
) -> List[torch.Tensor]:
    """
    For each transformer layer (0-indexed), return [N, D] hidden state at the *last prompt token*
    (i.e., just before the model starts answering).
    """
    n_layers = model.config.num_hidden_layers
    buckets: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_ids = [render_prompt(tokenizer, q, use_chat_template) for q in batch_q]
        max_len = max(len(x) for x in batch_ids)

        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, ids in enumerate(batch_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        last_pos = attn.sum(dim=1) - 1

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[1:]  # skip embeddings

        for layer in range(n_layers):
            h_last = hs[layer][torch.arange(hs[layer].shape[0], device=device), last_pos]
            buckets[layer].append(h_last.detach().cpu().float())

    return [torch.cat(parts, dim=0) for parts in buckets]


# -----------------------------
# Layer selection (principled)
# -----------------------------

@dataclass
class LayerMetrics:
    layer: int
    valid: bool
    r: float
    abs_r: float
    rmse: float
    proj_std: float
    split_lo: float
    split_hi: float
    n_top_train: int
    n_bot_train: int
    n_neu_train: int


def select_best_layer(
        hs_by_layer: List[torch.Tensor],  # each [N_train, D]
        s_train: np.ndarray,              # [N_train]
        quantile: float,
        min_abs_r: float,
        min_proj_std: float,
        rmse_eps: float,
) -> Tuple[LayerMetrics, torch.Tensor, float, float]:
    """
    For each layer:
      - Split TRAIN prompts into top/bot quantiles by AI-preference score s_train
      - Axis = mean(top) - mean(bottom)
      - Projection = cosine(h, axis)
      - Evaluate: corr(proj, s_train), rmse(best affine), std(proj)
    Choose: among valid layers (abs_r & proj_std thresholds),
            take layers with rmse <= min_rmse + rmse_eps,
            pick max abs_r (tie-break: smaller rmse).
    """
    lo = float(np.quantile(s_train, quantile))
    hi = float(np.quantile(s_train, 1.0 - quantile))

    top_mask = s_train >= hi
    bot_mask = s_train <= lo
    neu_mask = ~(top_mask | bot_mask)

    n_top = int(top_mask.sum())
    n_bot = int(bot_mask.sum())
    n_neu = int(neu_mask.sum())

    cand: List[Tuple[LayerMetrics, torch.Tensor]] = []
    for layer, H in enumerate(hs_by_layer):
        Ht = H[top_mask]
        Hb = H[bot_mask]

        if Ht.numel() == 0 or Hb.numel() == 0:
            m = LayerMetrics(layer, False, 0.0, 0.0, float("inf"), 0.0, lo, hi, n_top, n_bot, n_neu)
            cand.append((m, torch.zeros(H.shape[1])))
            continue

        axis = (Ht.mean(dim=0) - Hb.mean(dim=0))
        axis = axis / (axis.norm() + 1e-12)

        proj = cosine_rows(H, axis).cpu().numpy()
        r = pearsonr(proj, s_train)
        rmse = fit_affine_rmse(proj, s_train)
        proj_std = float(np.std(proj))

        valid = (abs(r) >= min_abs_r) and (proj_std >= min_proj_std)
        m = LayerMetrics(layer, bool(valid), float(r), float(abs(r)), float(rmse), float(proj_std), lo, hi, n_top, n_bot, n_neu)
        cand.append((m, axis.cpu().float()))

    valid = [(m, axis) for (m, axis) in cand if m.valid]
    if not valid:
        best_m, best_axis = max(cand, key=lambda t: t[0].abs_r)
        return best_m, best_axis, lo, hi

    min_rmse = min(m.rmse for m, _ in valid)
    shortlist = [(m, axis) for (m, axis) in valid if m.rmse <= (min_rmse + rmse_eps)]
    best_m, best_axis = max(shortlist, key=lambda t: (t[0].abs_r, -t[0].rmse))
    return best_m, best_axis, lo, hi


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", required=True, help="Space-separated list of model names/paths")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pairs-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--min-abs-r", type=float, default=0.2)
    ap.add_argument("--min-proj-std", type=float, default=1e-4)
    ap.add_argument("--rmse-eps", type=float, default=0.002)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Loop through each model
    for model_name in args.model:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}\n")

        # Create output directory based on the sanitized model name.
        # No separate CLI arg is needed: the model id uniquely determines the directory.
        sanitized_name = sanitize_model_name(model_name)
        outdir = ARTIFACTS_ROOT / sanitized_name
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {outdir.resolve()}\n")

        print(f"Loading: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        # Ensure padding is consistent for scoring batches
        model.config.pad_token_id = tok.pad_token_id
        device = model.device

        rows = iter_prompt_rows()
        train_idx, test_idx = groupwise_split(rows, args.seed)

        texts = [r["text"] for r in rows]
        subjects = [r["subject"] for r in rows]
        groups = [r["group"] for r in rows]

        # ---- 1) AI-preference score s per prompt (option log-probs at answer start)
        print(f"Scoring AI-preference dataset: N={len(rows)}  groups={len(set(groups))}")

        ai_concepts = list(AI_CONCEPTS.keys())

        opt_ai_variant_ids: Dict[str, List[List[int]]] = {
            c: [tok_option(tok, v) for v in AI_CONCEPTS[c]] for c in ai_concepts
        }
        opt_other_ids_by_subject: Dict[str, List[List[int]]] = {
            subj: [tok_option(tok, w) for w in words] for subj, words in OTHER_BY_SUBJECT.items()
        }

        s_raw = np.zeros(len(rows), dtype=np.float64)

        for i, (q, subj) in enumerate(zip(texts, subjects)):
            prompt_ids = render_prompt(tok, q, args.use_chat_template)

            option_ids_flat: List[List[int]] = []
            concept_slices: Dict[str, slice] = {}
            cursor = 0
            for c in ai_concepts:
                vids = opt_ai_variant_ids[c]
                option_ids_flat.extend(vids)
                concept_slices[c] = slice(cursor, cursor + len(vids))
                cursor += len(vids)

            other_ids = opt_other_ids_by_subject[subj]
            other_slice = slice(cursor, cursor + len(other_ids))
            option_ids_flat.extend(other_ids)

            avg_logp = score_options_len_norm(
                model=model,
                tokenizer=tok,
                device=device,
                prompt_ids=prompt_ids,
                option_ids_list=option_ids_flat,
                pairs_batch=args.pairs_batch,
            )

            concept_logps = [logsumexp(avg_logp[concept_slices[c]]) for c in ai_concepts]
            ai_set = logsumexp(np.array(concept_logps, dtype=np.float64))
            other_set = logsumexp(avg_logp[other_slice])

            s_raw[i] = ai_set - other_set

        # Standardize per-group using TRAIN stats only (avoid leakage + reduce group/subject confounds)
        s = s_raw.copy()
        by_group: Dict[str, List[int]] = {}
        for idx, g in enumerate(groups):
            by_group.setdefault(g, []).append(idx)

        train_set = set(train_idx)
        for g, idxs in by_group.items():
            idxs_tr = [j for j in idxs if j in train_set]
            mu = float(np.mean(s_raw[idxs_tr]))
            sd = float(np.std(s_raw[idxs_tr]))
            sd = max(sd, 1e-6)
            s[idxs] = (s_raw[idxs] - mu) / sd

        # ---- 2) Embed answer-position hidden states for every prompt, every layer
        print("Embedding AI-preference dataset (answer position hidden states)...")
        hs_all = embed_answer_position(
            model=model,
            tokenizer=tok,
            device=device,
            questions=texts,
            use_chat_template=args.use_chat_template,
            batch_size=args.batch_size,
        )

        # ---- 3) Pick layer on TRAIN only (principled)
        s_train = s[train_idx]
        hs_train = [H[train_idx] for H in hs_all]

        best_m, axis, split_lo, split_hi = select_best_layer(
            hs_by_layer=hs_train,
            s_train=s_train,
            quantile=args.quantile,
            min_abs_r=args.min_abs_r,
            min_proj_std=args.min_proj_std,
            rmse_eps=args.rmse_eps,
        )

        print("\n=== Extract AI-preference WMD ===")
        print(f"Best AI-preference layer: {best_m.__dict__}")

        # ---- 4) Held-out "manifestation": projection gap + bootstrap CI
        H_test = hs_all[best_m.layer][test_idx]
        s_test = s[test_idx]

        top_mask = s_test >= split_hi
        bot_mask = s_test <= split_lo

        proj = cosine_rows(H_test, axis.to(H_test.device)).cpu().numpy()
        top_vals = proj[top_mask]
        bot_vals = proj[bot_mask]

        top_mean = float(np.mean(top_vals))
        bot_mean = float(np.mean(bot_vals))
        gap = float(top_mean - bot_mean)

        ci_lo, ci_hi = bootstrap_ci_gap(
            top=top_vals, bot=bot_vals, n=args.bootstrap, ci=args.ci, seed=args.seed
        )

        print("\n=== AI-preference internal manifestation (held-out) ===")
        print(f"Chosen analysis layer: {best_m.layer}")
        print(f"Mean cosine proj (top {int(args.quantile*100)}% AI-pref): {top_mean:+.4f}")
        print(f"Mean cosine proj (bot {int(args.quantile*100)}% AI-pref): {bot_mean:+.4f}")
        print(f"Gap (top - bottom): {gap:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  (bootstrap {int(args.ci*100)}%, n={args.bootstrap})")

        # ---- 5) Save artifacts (Path-safe)
        np.save(outdir / f"axis_layer_{best_m.layer}.npy", axis.cpu().numpy().astype(np.float32))

        summary = {
            "model": model_name,
            "seed": args.seed,
            "use_chat_template": bool(args.use_chat_template),
            "batch_size": args.batch_size,
            "pairs_batch": args.pairs_batch,
            "quantile": args.quantile,
            "filters": {
                "min_abs_r": args.min_abs_r,
                "rmse_eps": args.rmse_eps,
                "min_proj_std": args.min_proj_std,
            },
            "ai_pref": {
                "ai_concepts": list(AI_CONCEPTS.keys()),
                "other_by_subject": OTHER_BY_SUBJECT,
                "best_layer": best_m.__dict__,
            },
            "analysis": {
                "analysis_layer": best_m.layer,
                "split_lo": split_lo,
                "split_hi": split_hi,
                "top_mean": top_mean,
                "bottom_mean": bot_mean,
                "gap_top_minus_bottom": gap,
                "gap_ci_low": ci_lo,
                "gap_ci_high": ci_hi,
                "bootstrap": {"n": args.bootstrap, "ci": args.ci},
            },
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved artifacts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
