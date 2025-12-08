import re
import copy
from pathlib import Path

import pandas as pd

from constants import MODELS_LIST, HOME_CONFIG, QUESTIONS, QID_ORDER
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


def main():
    out_path = Path("data/responses.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    base_prompts = _build_prompts()
    sampling_params = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=2048)

    for model in MODELS_LIST:
        print(f"Processing model: {model}")

        # Extract model size like "70b", "4B", "1.5b" etc.
        model_size_match = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model)
        model_size_b = float(model_size_match.group(1)) if model_size_match else None

        # Copy HOME_CONFIG robustly
        try:
            base_cfg = copy.deepcopy(HOME_CONFIG)
        except Exception:
            base_cfg = LLMResourceConfig(**HOME_CONFIG.__dict__)  # fallback

        if model_size_b is not None:
            print(f"Model size: {model_size_b}B")
            cfg = LLMResourceConfig(**base_cfg.__dict__) if not isinstance(base_cfg, LLMResourceConfig) else base_cfg
            cfg.scale_for_model_size(model_size_b)
        else:
            cfg = LLMResourceConfig(**base_cfg.__dict__) if not isinstance(base_cfg, LLMResourceConfig) else base_cfg

        cfg.max_model_len = 4096  # longer responses

        llm = LLMClient(model_name=model, config=cfg)

        try:
            results = llm.run_batch(base_prompts, sampling_params, output_field="response")
        finally:
            llm.delete_client()

        for r in results:
            all_rows.append(
                {
                    "model": model,
                    "question_id": r.get("qid"),
                    "subject": r.get("subject"),
                    "group": r.get("group"),
                    "variant": r.get("variant"),
                    "response": r.get("response"),
                }
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
