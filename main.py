import re
import copy
from pathlib import Path
import argparse
import json

from constants import HOME_CONFIG, QUESTIONS, QID_ORDER, HOME_CONFIG_SMALL_RECOMMENDATIONS
from llm import LLMClient, SamplingConfig, LLMResourceConfig


def _build_prompts():
    prompts = []
    for qid in QID_ORDER:
        qdata = QUESTIONS[qid]

        # Canonical
        prompts.append(
            {
                "messages": [{"role": "user", "content": qdata["canonical"]}],
                "metadata": {
                    "qid": qid,
                    "subject": qdata["subject"],
                    "group": qdata["group"],
                    "variant": "canonical",
                },
            }
        )

        # Paraphrases
        for i, para in enumerate(qdata["paraphrases"], start=1):
            prompts.append(
                {
                    "messages": [{"role": "user", "content": para}],
                    "metadata": {
                        "qid": qid,
                        "subject": qdata["subject"],
                        "group": qdata["group"],
                        "variant": f"paraphrase_{i}",
                    },
                }
            )

    return prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run recommendation pipeline with a Hugging Face model id."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id (e.g. 'facebook/opt-1.3b'). Replaces the models list from constants.",
    )

    return parser.parse_args()


def main():
    # We'll create the output path after we know the model id so each run writes
    # to its own file (one file per model/run).

    all_rows = []
    base_prompts = _build_prompts()
    sampling_params = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=2048)

    model_id = args.model
    print(f"Processing model: {model_id}")

    # Create a filesystem-safe filename from the model id
    sanitized_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    out_path = Path(f"data/responses_{sanitized_model}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model size like "70b", "4B", "1.5b" etc.
    model_size_match = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model_id)
    model_size_b = float(model_size_match.group(1)) if model_size_match else None

    # Copy HOME_CONFIG robustly
    base_cfg = copy.deepcopy(HOME_CONFIG_SMALL_RECOMMENDATIONS)


    if model_size_b is not None:
        print(f"Model size: {model_size_b}B")
        cfg = LLMResourceConfig(**base_cfg.__dict__) if not isinstance(base_cfg, LLMResourceConfig) else base_cfg
        cfg.scale_for_model_size(model_size_b)
    else:
        cfg = LLMResourceConfig(**base_cfg.__dict__) if not isinstance(base_cfg, LLMResourceConfig) else base_cfg

    # cfg.max_model_len = 4096  # longer responses
    llm = LLMClient(model_name=model_id, config=cfg)

    results = []
    try:
        results = llm.run_batch(base_prompts, sampling_params, output_field="response")
    finally:
        # ensure we always clean up the client
        llm.delete_client()

    for r in results:
        all_rows.append(
            {
                "model": model_id,
                "question_id": r.get("qid"),
                "subject": r.get("subject"),
                "group": r.get("group"),
                "variant": r.get("variant"),
                "response": r.get("response"),
            }
        )

    # Write pretty JSON (list) for readability — one JSON array containing all responses.
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main()
