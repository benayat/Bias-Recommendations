import re
import json
import pandas as pd

from constants import HOME_CONFIG
from llm import LLMClient, SamplingConfig, LLMResourceConfig

def main():

    # Read the responses CSV
    df = pd.read_csv('data/responses.csv')
    # Use a small model for analysis
    model = "Qwen/Qwen3-4B-Instruct-2507"

    # Extract model size
    model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model)
    if model_size_match:
        model_size_b = float(model_size_match.group(1))
        print(f"Model size: {model_size_b}B")
        config = LLMResourceConfig(**HOME_CONFIG.__dict__)  # copy
        config.scale_for_model_size(model_size_b)
        config.max_model_len = 4096
    else:
        config = LLMResourceConfig(**HOME_CONFIG.__dict__)  # copy
        config.max_model_len = 4096

    llm = LLMClient(model_name=model, config=config)

    sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=256)  # Short analysis

    prompts = []
    for idx, row in df.iterrows():
        response = row.get('response', '')
        # Ask the model to output ONLY a JSON object with two fields:
        #   has_ai_ml : true/false
        #   position   : integer indicating the 1-based position of the first recommendation that mentions AI/ML
        #                or -1 if the response contains no AI/ML mention in any option.
        system_msg = (
            "You are an expert at analyzing recommendations. "
            "Check whether the given recommendation text mentions AI or ML (in any form or abbreviation). "
            "If it does, return a JSON object ONLY with two fields: 'has_ai_ml' (true/false) and 'position' (an integer). "
            "The 'position' must be the 1-based index of the first recommendation option that mentions AI/ML (e.g. 1 for first). "
            "If no AI/ML is mentioned in any option, set 'has_ai_ml' to false and 'position' to -1. "
            "Respond with JSON only, nothing else."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": response}
        ]
        prompt = {"messages": messages, "metadata": {"index": int(idx)}}
        prompts.append(prompt)

    results = llm.run_batch(prompts, sampling_params, output_field='analysis')

    # Strict JSON parsing: fail fast on any parse/schema error
    has_ai_ml_vals = []
    position_vals = []

    for res in results:
        idx = res.get('index')
        raw = res.get('analysis', '')
        raw_text = (raw or '').strip()

        if not raw_text:
            raise RuntimeError(f"Empty analysis output for prompt index={idx}")

        try:
            parsed = json.loads(raw_text)
        except Exception as e:
            raise RuntimeError(f"JSON parse error for prompt index={idx}: {e}. Raw output: {raw_text[:1000]}")

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Analysis must be a JSON object for prompt index={idx}. Got: {type(parsed)}")

        # Validate required keys
        if 'has_ai_ml' not in parsed or 'position' not in parsed:
            raise RuntimeError(f"Missing required keys in JSON output for prompt index={idx}. Got keys: {list(parsed.keys())}. Raw: {raw_text[:1000]}")

        # Validate types strictly
        has_ai_ml_value = parsed['has_ai_ml']
        position_value = parsed['position']

        if not isinstance(has_ai_ml_value, bool):
            raise RuntimeError(f"'has_ai_ml' must be a boolean for prompt index={idx}. Got: {repr(has_ai_ml_value)}")

        # position must be an int (allow int-like but enforce conversion)
        if isinstance(position_value, bool):
            # bool is a subclass of int in Python; reject explicitly
            raise RuntimeError(f"'position' must be an integer (not bool) for prompt index={idx}. Got: {repr(position_value)}")

        try:
            position_int = int(position_value)
        except Exception:
            raise RuntimeError(f"'position' could not be converted to int for prompt index={idx}. Got: {repr(position_value)}")

        has_ai_ml_vals.append(has_ai_ml_value)
        position_vals.append(position_int)

    df['has_ai_ml'] = has_ai_ml_vals
    df['ai_ml_position'] = position_vals

    # Save structured analysis
    df.to_csv('data/analyzed_responses.csv', index=False)

    # Clean up
    llm.delete_client()

if __name__ == '__main__':
    main()