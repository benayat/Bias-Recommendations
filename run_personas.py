"""
Simple script to test all personas against questions in a single batch LLM call.
Each persona gets each question (canonical only, no paraphrases).
"""
import re
import json
import argparse
from pathlib import Path

from constants import HOME_CONFIG_SMALL_RECOMMENDATIONS, QUESTIONS, QID_ORDER, PERSONAS, PERSONA_COMPLETION_SUFFIX
from llm import LLMClient, SamplingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run all personas against questions")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    return parser.parse_args()


def build_batch_prompts():
    """Build one prompt for each (persona, question) combination."""
    prompts = []

    for persona_id, persona_text in PERSONAS.items():
        system_message = persona_text + PERSONA_COMPLETION_SUFFIX

        for qid in QID_ORDER:
            question = QUESTIONS[qid]["canonical"]

            prompts.append({
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                "metadata": {
                    "persona_id": persona_id,
                    "qid": qid,
                    "subject": QUESTIONS[qid]["subject"],
                    "group": QUESTIONS[qid]["group"],
                }
            })

    return prompts


def main():
    args = parse_args()
    model_id = args.model

    print(f"Processing model: {model_id}")
    print(f"Personas: {len(PERSONAS)}")
    print(f"Questions: {len(QID_ORDER)}")

    # Build all prompts (personas × questions)
    prompts = build_batch_prompts()
    print(f"Total prompts in batch: {len(prompts)}")

    # Setup LLM client
    config = HOME_CONFIG_SMALL_RECOMMENDATIONS
    llm = LLMClient(model_name=model_id, config=config)

    # Single batch call
    sampling_params = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=2048)
    results = []

    try:
        results = llm.run_batch(prompts, sampling_params, output_field="response")
    finally:
        llm.delete_client()

    # Format results
    all_rows = []
    for r in results:
        all_rows.append({
            "model": model_id,
            "persona_id": r.get("persona_id"),
            "question_id": r.get("qid"),
            "subject": r.get("subject"),
            "group": r.get("group"),
            "response": r.get("response"),
        })

    # Write output
    sanitized_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    out_path = Path(f"data/personas_{sanitized_model}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_rows)} results to: {out_path}")


if __name__ == "__main__":
    main()

