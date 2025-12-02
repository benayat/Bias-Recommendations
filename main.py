import re
import pandas as pd

from constants import MODELS_LIST, QUESTIONS, HOME_CONFIG
from llm import LLMClient, SamplingConfig, LLMResourceConfig

def main():
    data = []
    for model in MODELS_LIST:
        print(f"Processing model: {model}")
        # Extract model size
        model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model)
        if model_size_match:
            model_size_b = float(model_size_match.group(1))
            print(f"Model size: {model_size_b}B")
            config = LLMResourceConfig(**HOME_CONFIG.__dict__)  # copy
            config.scale_for_model_size(model_size_b)
            config.max_model_len = 4096  # Adjust for longer responses
        else:
            config = LLMResourceConfig(**HOME_CONFIG.__dict__)  # copy
            config.max_model_len = 4096

        llm = LLMClient(model_name=model, config=config)

        sampling_params = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=2048)

        prompts = []
        for qid, qdata in QUESTIONS.items():
            # Canonical question
            question = qdata["canonical"]
            messages = [
                {"role": "user", "content": question}
            ]
            prompt = {"messages": messages, "metadata": {"qid": qid, "variant": "canonical"}}
            prompts.append(prompt)

            # Paraphrases
            for i, para in enumerate(qdata["paraphrases"]):
                question = para
                messages = [
                    {"role": "user", "content": question}
                ]
                prompt = {"messages": messages, "metadata": {"qid": qid, "variant": f"paraphrase_{i+1}"}}
                prompts.append(prompt)

        results = llm.run_batch(prompts, sampling_params, output_field='response')

        for result in results:
            qid = result["qid"]
            variant = result["variant"]
            response = result["response"]
            data.append({"model": model, "question_id": qid, "variant": variant, "response": response})

        # Clean up the client after processing the model
        llm.delete_client()

    df = pd.DataFrame(data)
    df.to_csv('data/responses.csv', index=False)

if __name__ == '__main__':
    main()
