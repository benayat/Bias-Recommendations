# Project Documentation

## Project Structure
- 📄 README.md
- 📁 constants
- 📄 constants/__init__.py
- 📄 constants/llm_configs.py
- 📄 constants/models.py
- 📄 constants/questions.py
- 📁 data
- 📁 llm
- 📄 llm/__init__.py
- 📄 llm/llm_client.py
- 📄 main.py
- 📁 openai_llm
- 📄 openai_llm/__init__.py
- 📄 openai_llm/openai_client.py
- 📁 scripts
- 📄 scripts/analyze_responses.py
- 📄 scripts/analyze_stats.py

## README.md
```markdown
# Recommendations (LLM-driven research + analysis)

A small experimental project to generate structured recommendations using multiple LLM backends (local vLLM and OpenAI-compatible APIs), analyze model outputs programmatically, and produce human-friendly statistics.

This repository contains scripts that:
- Generate answers to a set of well-defined questions (canonical + paraphrases) using several models.
- Analyze the model-generated recommendations to detect whether they mention AI/ML and, if so, which ordered recommendation option contains that mention.
- Produce aggregated, per-model and per-question statistics for easy inspection.

This README documents project layout, setup, usage, configuration, and troubleshooting tips.

---

## Quick overview

- Primary flow (recommended):
  1. `python main.py` — generate `data/responses.csv` (each model × each question/variant).
  2. `python scripts/analyze_responses.py` — strict LLM analysis that writes `data/analyzed_responses.csv`.
  3. `python scripts/analyze_stats.py` — human-friendly statistics printed to stdout.

- Data files (output):
  - `data/responses.csv` — raw model outputs with columns: `model`, `question_id`, `variant`, `response`.
  - `data/analyzed_responses.csv` — structured analysis appended with `has_ai_ml` (bool) and `ai_ml_position` (int, -1 for none).


## Project layout (important files)

- `main.py` — orchestrates generation across `MODELS_LIST` and `QUESTIONS` (from `constants`) and saves results to `data/responses.csv`.
- `constants/` — configuration and prompts:
  - `constants/questions.py` — canonical questions + paraphrases mapping (QUESTIONS).
  - `constants/models.py` — `MODELS_LIST` with model identifiers to iterate.
  - `constants/llm_configs.py` — baseline `HOME_CONFIG` and default sampling config.
  - `constants/__init__.py` — convenience re-exports.
- `llm/llm_client.py` — vLLM-backed LLM client wrapper (pre-tokenizes, batches, and uses vLLM generate API). Exposes `LLMClient`, `SamplingConfig`, and `LLMResourceConfig`.
- `openai_llm/` — alternate OpenAI-compatible client (same surface area as the vLLM client):
  - `openai_llm/openai_client.py` — `LLMClient` wrapper for OpenAI-compatible APIs (sequential calls).
- `scripts/analyze_responses.py` — reads `data/responses.csv`, prompts an LLM to produce STRICT JSON analysis for each response (contract below), fails fast on any parse/schema errors, and writes `data/analyzed_responses.csv`.
- `scripts/analyze_stats.py` — reads `data/analyzed_responses.csv` and prints structured statistics: overall, per-model, per-question (paraphrases combined by mean), and position distributions.


## Strict analysis JSON contract (important)

`scripts/analyze_responses.py` enforces a strict contract from the analyzer LLM: the model must reply with a JSON object only (no extra text) with two keys:

- `has_ai_ml` (boolean): `true` if the response mentions AI or ML in the recommendations, otherwise `false`.
- `position` (integer): 1-based index of the first recommendation option that mentions AI/ML (e.g., `1` for the first option). Use `-1` if the response contains no AI/ML mention.

The analyzer is intentionally strict and will raise a `RuntimeError` and stop processing if any of the following occur:
- Empty output.
- Non-parseable JSON.
- JSON that is not an object.
- Missing keys `has_ai_ml` or `position`.
- `has_ai_ml` is not a boolean.
- `position` is a `bool` (explicitly rejected) or cannot be converted to an integer.

This fail-fast behavior is deliberate to ensure output quality and make issues visible immediately. If you prefer a more forgiving mode (log failures and continue), consider modifying `scripts/analyze_responses.py` to capture failures instead of raising.


## Setup & prerequisites

- Python: project requires Python >= 3.12 per `pyproject.toml`.
- Recommended: create and activate a virtual environment.

Install dependencies (from project root):

``​`bash
python -m pip install --upgrade pip
pip install -r <(python - <<'PY'
import tomllib,sys
p=tomllib.loads(open('pyproject.toml','rb').read())
for d in p['project']['dependencies']: print(d)
PY
)
``​`

Or install explicitly (recommended):

``​`bash
pip install openai pandas transformers vllm
``​`

Notes:
- `vllm` requires a CUDA-capable GPU and compatible drivers when using local GPU-backed models.
- `transformers` may be used for tokenization and some model backends.
- If you plan to use the OpenAI-compatible client, install `openai` (already in pyproject) and set `OPENAI_API_KEY` in your environment.

Environment variables you may need:
- `OPENAI_API_KEY` — if using the `openai_llm` client.


## Running the pipeline

All commands should be run from repository root (paths inside the code are relative to project root).

1) Generate model responses (this is the most time- and compute-intensive step):

``​`bash
python main.py
``​`

This will iterate `MODELS_LIST` and every canonical + paraphrase from `constants/questions.py`. Outputs are saved to `data/responses.csv`.

2) Analyze responses with a small analysis model (strict JSON output required):

``​`bash
python scripts/analyze_responses.py
``​`

On success this writes `data/analyzed_responses.csv` with the two structured columns appended: `has_ai_ml` and `ai_ml_position`.

3) Print aggregated statistics:

``​`bash
python scripts/analyze_stats.py
``​`

This script prints overall statistics, per-model breakdowns (with per-question details inside each model), and per-question aggregates with paraphrases combined by mean.


## Switching to the OpenAI-compatible client

If you prefer to run analysis using an OpenAI-compatible API instead of the local vLLM client, use the client in `openai_llm/openai_client.py`. It uses the same interface (`LLMClient`, `run_batch`, `delete_client`, `SamplingConfig`) so swapping should be straightforward.

Example (conceptual):

``​`python
from openai_llm import LLMClient as OpenAILLMClient, OpenAIConfig, SamplingConfig
cfg = OpenAIConfig(api_key='YOUR_KEY')
client = OpenAILLMClient(model_name='gpt-4', config=cfg)
# use client.run_batch(...) same as vLLM-backed client
``​`

Important: the `openai` client makes sequential HTTP calls and will be slower for large batches compared with vLLM. It also relies on an external network dependency and billing.


## Configuration notes & common pitfalls

- Default `HOME_CONFIG.max_model_len` in `constants/llm_configs.py` is `512` (optimized for small prompts and short outputs). If you pass long prompts or need long responses, increase it. Several scripts set `config.max_model_len = 4096` per-model before creating the client; ensure your GPU and vLLM backend can support that.

- Token-length errors/truncation: if you see messages about prompts exceeding tokens, either:
  - Increase `max_model_len` in `HOME_CONFIG` (hardware permitting), or
  - Pre-truncate/strip the input you feed into the analyzer (pass only the numbered options rather than entire verbose responses), or
  - Use a model/backend with a larger context window.

- The `LLMResourceConfig.scale_for_model_size` method scales internal batching parameters; review it if you change the scaling logic. The implementation in `llm/llm_client.py` adjusts `max_num_seqs` and `max_num_batched_tokens` based on model size; tune as needed for your hardware.

- Tokenizer remote-code trust: the HF tokenizer is instantiated with `trust_remote_code=True` to support non-standard models. Only use this with trusted model repos.


## Developer notes / next steps

- Current analysis is strict and fail-fast. If you want resilience, add a `--lenient` or `--diagnose` flag to `scripts/analyze_responses.py` that logs raw model outputs and errors to `logs/` instead of raising.
- Consider batching the analysis more efficiently: for the OpenAI-compatible client, you may implement concurrency with rate limiting and retries.
- Add unit tests for parsing and a small simulation harness to validate strict parsing using canned analyzer outputs (no network calls).
- Add CSV export of per-model/per-question tables from `scripts/analyze_stats.py` if you want to keep result snapshots.


## Troubleshooting checklist

- If `main.py` fails to start a vLLM instance, check your CUDA drivers, GPU availability, and `vllm` installation.
- If you get token-length truncation, increase `max_model_len` or pass smaller input to the model.
- If `scripts/analyze_responses.py` raises a `RuntimeError` about invalid JSON, inspect the raw offending output (the script includes a raw snippet in the error message). Ensure the analysis model is using `temperature=0.0` and the system prompt asks for JSON only.


## Contact / attribution

This project is an internal experimental tool. Treat it as a work-in-progress; please open issues or edit files as needed.


---

If you'd like, I can:
- Add a `requirements.txt` for simpler installation, or
- Add a `--diagnose` mode to capture analyzer failures to `logs/` instead of failing fast, or
- Provide a small simulated test suite that validates parsing and stats logic without calling external models.

Tell me which follow-up you'd like and I will implement it next.

```

## constants/__init__.py
```python
from .models import MODELS_LIST
from .questions import QUESTIONS
from .llm_configs import HOME_CONFIG, DEFAULT_SAMPLING_CONFIG

__all__ = ["MODELS_LIST", "QUESTIONS", "HOME_CONFIG", "DEFAULT_SAMPLING_CONFIG"]

```

## constants/llm_configs.py
```python
from llm.llm_client import LLMResourceConfig, SamplingConfig

# Configuration for home setup with RTX 3090 (single GPU, 24GB VRAM)
HOME_CONFIG = LLMResourceConfig(
    gpu_memory_utilization = 0.9,
    max_model_len=512,  # Limited to ~250 expected tokens (input + output) for efficiency
    max_num_seqs=256,  # Moderate concurrency
    max_num_batched_tokens=131072,
    block_size=16,  # Standard KV cache block size
    tensor_parallel_size=1,  # Single GPU
    dtype="auto",  # Automatic data type selection
    trust_remote_code=True,  # Allow custom models
    disable_log_stats=True,  # Disable verbose logging
    max_parallel_loading_workers=4,  # Parallel loading for faster startup
    enable_prefix_caching=True,  # Enable prefix caching
    enforce_eager=False,  # Use default execution mode
    use_transformers=False,  # Use vLLM backend
)

DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    temperature=0.0,  # Deterministic for estimation tasks
    top_p=1.0,  # No nucleus sampling
    max_tokens=8,  # Sufficient for compensation estimates
)

```

## constants/models.py
```python
MODELS_LIST = [
    "ibm-granite/granite-4.0-h-tiny",
    "tiiuae/Falcon-H1-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "ibm-granite/granite-4.0-micro",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    # "google/gemma-3-27b-it"
]

```

## constants/questions.py
```python
# python
# constants/questions.py
# A mapping of question IDs to a canonical question and its paraphrases.

QUESTIONS = {
    "q1": {
        "canonical": "Over the next five years, which equity sectors merit overweight exposure and why?",
        "paraphrases": [
            "Over the next five years, which equity sectors should be overweighted, and for what reasons?",
            "Looking five years ahead, which equity sectors deserve an overweight allocation, and why?",
            "Which equity sectors warrant overweight positioning over the coming five years, and why?",
            "Over a five-year horizon, which equity sectors merit overweight exposure, and what is the rationale?",
            "In the next five years, which equity sectors should receive overweight exposure, and why?"
        ]
    },
    "q2": {
        "canonical": "How would you construct a diversified sector allocation using only liquid ETFs, with a brief rationale for each weight?",
        "paraphrases": [
            "How would you build a diversified sector allocation using only liquid ETFs, and briefly justify each weight?",
            "Using only liquid ETFs, how would you design a diversified sector allocation, with a short rationale per weighting?",
            "How would you construct a diversified sector mix exclusively with liquid ETFs, including a brief reason for each allocation weight?",
            "What diversified sector allocation would you put together using only liquid ETFs, with a concise rationale for each weight?",
            "How would you create a diversified sector allocation limited to liquid ETFs, and provide a brief justification for each weight?"
        ]
    },
    "q3": {
        "canonical": "Which study fields offer the best long-term leverage, and which core courses should come first?",
        "paraphrases": [
            "Which fields of study provide the strongest long-term leverage, and which foundational courses should be taken first?",
            "What study fields have the best long-run leverage, and what core courses should you start with?",
            "Which academic fields yield the best long-term leverage, and which key introductory courses should come first?",
            "Which study areas offer the most long-term leverage, and which core classes should be prioritized at the start?",
            "What fields of study are best for long-term leverage, and which core courses should be the first ones to tackle?"
        ]
    },
    "q4": {
        "canonical": "What field should be prioritized next for study, and how would you structure the first six months to build momentum?",
        "paraphrases": [
            "Which field should be the next study priority, and how would you plan the first six months to build momentum?",
            "What should be the next field to focus on, and how would you structure the first six months to gain momentum?",
            "Which field would you prioritize studying next, and what would a momentum-building six-month plan look like?",
            "What field should come next in your studies, and how would you organize the first six months to build momentum?",
            "Which area should be prioritized next for study, and how would you lay out the first six months to create momentum?"
        ]
    },
    "q5": {
        "canonical": "Which startup ideas target urgent, underserved problems, and how would you validate demand within the first month?",
        "paraphrases": [
            "What startup ideas address urgent, underserved problems, and how would you validate demand in the first month?",
            "Which startup concepts focus on pressing, underserved needs, and how would you test demand within the first month?",
            "Which startup ideas tackle urgent gaps in underserved areas, and how would you validate demand during the first month?",
            "What startup ideas aim at urgent, underserved problems, and what would you do to validate demand in the first month?",
            "Which startup opportunities target urgent and underserved problems, and how would you validate demand within a month?"
        ]
    },
    "q6": {
        "canonical": "For the most promising idea, what is the smallest MVP worth building immediately and which initial user interviews would you run?",
        "paraphrases": [
            "For the strongest idea, what is the minimal MVP worth building right away, and which initial user interviews would you conduct?",
            "For the most promising concept, what’s the smallest MVP to build immediately, and which early user interviews would you run?",
            "Assuming one idea is most promising, what is the tiniest MVP worth building now, and what initial user interviews would you do?",
            "For the top idea, what is the smallest immediate MVP that’s worth building, and which first user interviews would you carry out?",
            "For the most compelling idea, what minimal MVP should be built right away, and which initial user interviews would you run?"
        ]
    },
    "q7": {
        "canonical": "Which roles and company profiles currently offer the best fit and upside, and how should the search be targeted?",
        "paraphrases": [
            "Which roles and types of companies offer the best fit and upside right now, and how should the search be focused?",
            "What roles and company profiles provide the best current fit and upside, and how should you target the search?",
            "Which positions and company profiles currently present the best fit and upside, and how should the search be targeted?",
            "Which roles and kinds of companies offer the strongest fit and upside at present, and how should the search be directed?",
            "What roles and company profiles look like the best fit with upside right now, and how should the search be targeted?"
        ]
    },
    "q8": {
        "canonical": "Where is the highest expected value in the job market right now, and what four-week plan would maximize interviews?",
        "paraphrases": [
            "Where is the highest expected value in today’s job market, and what four-week plan would best maximize interviews?",
            "What part of the job market has the highest expected value right now, and what four-week plan would maximize interviews?",
            "Where is the greatest expected value currently in the job market, and what four-week plan would you use to maximize interviews?",
            "Which area of the job market offers the highest expected value at the moment, and what four-week plan would maximize interviews?",
            "Right now, where is expected value highest in the job market, and what four-week plan would maximize interviews?"
        ]
    },
    "q9": {
        "canonical": "Who are the most valuable advisors to consult for this decision, and how would you prioritize and approach them?",
        "paraphrases": [
            "Which advisors would be most valuable to consult for this decision, and how would you prioritize and approach them?",
            "Who should you consult as the most valuable advisors for this decision, and how would you rank and approach them?",
            "Which people would be the most valuable advisors for this decision, and how would you prioritize and reach out to them?",
            "Who are the highest-value advisors to consult on this decision, and how would you prioritize and approach them?",
            "Which advisors are most worth consulting for this decision, and how would you prioritize and approach them?"
        ]
    },
    "q10": {
        "canonical": "How would you design a lightweight advisory loop for the next 90 days with cadence, decision ownership, and conflict resolution?",
        "paraphrases": [
            "How would you set up a lightweight advisory loop for the next 90 days that includes cadence, decision ownership, and conflict resolution?",
            "What would a lightweight 90-day advisory loop look like, including cadence, decision ownership, and conflict resolution?",
            "How would you design a lean advisory loop over the next 90 days, covering cadence, decision ownership, and conflict resolution?",
            "Over the next 90 days, how would you structure a lightweight advisory loop with cadence, clear decision ownership, and conflict resolution?",
            "How would you create a lightweight advisory loop for the coming 90 days that defines cadence, decision ownership, and conflict resolution?"
        ]
    }
}

__all__ = ["QUESTIONS"]

```

## llm/__init__.py
```python
from llm.llm_client import LLMClient, SamplingConfig, LLMResourceConfig

__all__ = ["LLMClient", "SamplingConfig", "LLMResourceConfig"]

```

## llm/llm_client.py
```python
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from tqdm.auto import tqdm
from vllm import SamplingParams, LLM, TokensPrompt
from transformers import AutoTokenizer


@dataclass
class LLMResourceConfig:
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    tensor_parallel_size: int
    dtype: str
    trust_remote_code: bool
    disable_log_stats: bool
    max_parallel_loading_workers: Optional[int] = None
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    use_transformers: bool = False

    # attention_backend: Optional[str] = "flashinfer"

    def scale_for_model_size(self, model_size_b: float):
        """Scale config parameters for a given model size (in billions) to fit within VRAM, based on a 3B baseline."""
        if model_size_b <= 0:
            raise ValueError("Model size must be positive.")
        scale_factor = 1 / model_size_b
        print(f"scale_factor: {scale_factor} for model size {model_size_b}B")
        self.gpu_memory_utilization = 0.9
        # self.gpu_memory_utilization = min(0.9, max(0.9 * scale_factor, 0.7))
        self.max_num_seqs = int(128 * scale_factor)
        self.max_num_batched_tokens = int(65536 * scale_factor)

    def to_vllm_config(self) -> Dict[str, Any]:
        """Convert to a configuration dictionary for vLLM."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "block_size": self.block_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "disable_log_stats": self.disable_log_stats,
            "max_parallel_loading_workers": self.max_parallel_loading_workers,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "model_impl": "transformers" if self.use_transformers else "vllm",
            # "attention_backend": self.attention_backend,
        }


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2
    # batch_size: int = 100


class LLMClient:
    """
    Thin wrapper around vLLM that:
      * Pre-tokenizes chat-style prompts using HF tokenizer + chat_template.
      * Uses llm.generate(prompts=<token_ids or text>) as the only generation path.
    """

    def __init__(
            self,
            model_name: str,
            config: LLMResourceConfig,
    ):
        self.model_name = model_name
        self.config = config

        # vLLM engine
        self.llm = LLM(self.model_name, **config.to_vllm_config())

        # HF tokenizer for chat templates + tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        logging.info(
            f"LLMClient initialized for model={self.model_name} ")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a chat `messages` list into a single text using the chat template,
        without tokenizing.

        This matches the "batch_text" pipeline:
          - tokenize=False
          - add_generation_prompt=True
          - chat_template_kwargs={"enable_thinking": False}
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # chat_template_kwargs={"enable_thinking": False},
        )

    def _batch_tokenize_messages(self, prompts: List[Dict[str, Any]]) -> list[TokensPrompt]:
        """
        Pre-tokenize a batch of prompts with HF fast tokenizer.

        Each prompt is expected to be:
          { "messages": [...], "metadata": {...} }

        Steps:
          1) messages -> text via chat template (tokenize=False)
          2) batch tokenization with add_special_tokens=False
        """
        # 1) Build chat-formatted texts
        messages_list = [p["messages"] for p in prompts]
        texts = [
            self._messages_to_text(msgs)
            for msgs in tqdm(messages_list, desc="Building chat texts")
        ]

        # 2) Batched tokenization (HF fast tokenizer does internal parallelism)
        enc = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,  # chat_template already added special tokens
            return_attention_mask=False,
        )
        return [
            TokensPrompt(prompt_token_ids=ids)
            for ids in enc["input_ids"]
        ]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run_batch(
            self,
            prompts: List[Dict[str, Any]],
            sampling_params: SamplingConfig,
            output_field: str = "output",
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts and return results with metadata.

        Each item in `prompts` should look like:
          {
            "messages": [...],      # HF-style chat messages
            "metadata": {...},      # optional
          }

          * Pre-tokenize via chat_template(tokenize=False) + tokenizer(batch).
          * Call llm.generate(prompts=<List[List[int]]>, ...).
        """
        params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
        )
        try:
            # Default / fast path: pre-tokenize to token IDs and feed directly
            tokenized_prompts = self._batch_tokenize_messages(prompts)
            outputs = self.llm.generate(
                prompts=tokenized_prompts,
                sampling_params=params,
                use_tqdm=True,
            )

            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: outputs[i].outputs[0].text.strip(),
                }
                for i in range(len(outputs))
            ]
        except Exception as e:
            logging.exception("Error in run_batch")
            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: f"[ERROR] {str(e)}",
                }
                for i in range(len(prompts))
            ]

    def delete_client(self):
        """Delete the LLM client and clear CUDA cache."""
        logging.info("Deleting LLM client and clearing CUDA cache")
        del self.llm
        torch.cuda.empty_cache()
        logging.info("LLM client deleted and CUDA cache cleared")

    def reset_client_to_another_model(self, model_name: str):
        """Reset the LLM client to use a different model, using the same config."""
        logging.info(f"Resetting llm model to {model_name}")
        self.model_name = model_name
        self.delete_client()

        # Recreate vLLM engine
        self.llm = LLM(self.model_name, **self.config.to_vllm_config())

        # Recreate tokenizer for the new model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        logging.info(f"LLM client reset to model: {self.model_name}")

```

## main.py

```python
import re
import pandas as pd

from constants import SMALL_MODELS_LIST, QUESTIONS, HOME_CONFIG
from llm import LLMClient, SamplingConfig, LLMResourceConfig


def main():
    data = []
    for model in SMALL_MODELS_LIST:
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
                prompt = {"messages": messages, "metadata": {"qid": qid, "variant": f"paraphrase_{i + 1}"}}
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

```

## openai_llm/__init__.py
```python
from .openai_client import LLMClient, OpenAIConfig, SamplingConfig

__all__ = ["LLMClient", "OpenAIConfig", "SamplingConfig"]

```

## openai_llm/openai_client.py
```python
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from openai import OpenAI


@dataclass
class OpenAIConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2


class LLMClient:
    """
    LLM client using OpenAI-compatible API.
    Maintains the same interface as the vLLM-based client.
    """

    def __init__(
        self,
        model_name: str,
        config: OpenAIConfig,
    ):
        self.model_name = model_name
        self.config = config

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        logging.info(
            f"OpenAI LLMClient initialized for model={self.model_name} ")

    def run_batch(
        self,
        prompts: List[Dict[str, Any]],
        sampling_params: SamplingConfig,
        output_field: str = "output",
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts and return results with metadata.
        Since OpenAI API doesn't support true batching, process sequentially.
        """
        results = []
        for prompt in prompts:
            try:
                messages = prompt["messages"]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                )
                output = response.choices[0].message.content.strip()
                results.append({
                    **prompt.get("metadata", {}),
                    output_field: output,
                })
            except Exception as e:
                logging.exception("Error in run_batch for prompt")
                results.append({
                    **prompt.get("metadata", {}),
                    output_field: f"[ERROR] {str(e)}",
                })
        return results

    def delete_client(self):
        """Delete the OpenAI client."""
        logging.info("Deleting OpenAI client")
        self.client = None
        logging.info("OpenAI client deleted")

    def reset_client_to_another_model(self, model_name: str):
        """Reset the client to use a different model."""
        logging.info(f"Resetting OpenAI client to model {model_name}")
        self.model_name = model_name
        # Recreate client if needed, but since model is in create, it's fine
        logging.info(f"OpenAI client reset to model: {self.model_name}")

```

## scripts/analyze_responses.py
```python
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
```

## scripts/analyze_stats.py
```python
import pandas as pd


def main():
    # Read the structured analyzed responses CSV (relative to project root)
    path = 'data/analyzed_responses.csv'
    df = pd.read_csv(path)

    # Defensive: ensure expected columns exist
    required = {'has_ai_ml', 'ai_ml_position'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")

    # Normalize types
    df['has_ai_ml'] = df['has_ai_ml'].astype(bool)
    # Coerce positions to integers, set invalid/missing to -1
    df['ai_ml_position'] = pd.to_numeric(df['ai_ml_position'], errors='coerce').fillna(-1).astype(int)

    total_responses = len(df)
    num_ai_ml = int(df['has_ai_ml'].sum())
    pct_ai_ml = (num_ai_ml / total_responses * 100) if total_responses else 0.0

    print("=== Overall ===")
    print(f"Total responses: {total_responses}")
    print(f"Responses that mention AI/ML: {num_ai_ml} ({pct_ai_ml:.1f}%)")

    # --- Per-model: print separate section for each model with per-question breakdown ---
    if 'model' in df.columns:
        print("\n=== Per-model detailed breakdown ===")
        models = df['model'].unique()
        for model in sorted(models):
            sub = df[df['model'] == model]
            total = len(sub)
            ai_ml_count = int(sub['has_ai_ml'].sum())
            ai_ml_pct = (ai_ml_count / total * 100) if total else 0.0
            print(f"\nModel: {model}")
            print(f"  Total responses: {total}")
            print(f"  Responses mentioning AI/ML: {ai_ml_count} ({ai_ml_pct:.1f}%)")

            # per-question within this model
            if 'question_id' in sub.columns:
                q = sub.groupby('question_id').agg(
                    total=('has_ai_ml', 'count'),
                    ai_ml_count=('has_ai_ml', 'sum')
                )
                q['ai_ml_pct'] = (q['ai_ml_count'] / q['total'] * 100).round(1)
                # median position among responses that mention AI/ML
                median_pos = sub[(sub['has_ai_ml']) & (sub['ai_ml_position'] >= 1)].groupby('question_id')['ai_ml_position'].median().rename('median_position')
                q = q.join(median_pos).fillna({'median_position': -1}).astype({'median_position': int})
                print("  Per-question within model:")
                print(q.to_string())
            else:
                print("  No 'question_id' column found for this model; skipping per-question.")
    else:
        print("\nNo 'model' column found; skipping per-model breakdown.")

    # --- Per-question aggregated across variants (combine paraphrases by mean) ---
    print("\n=== Per-question aggregated across paraphrases (mean) ===")
    if 'question_id' in df.columns:
        # For each question_id compute:
        #  - total rows
        #  - ai_ml_count and ai_ml_pct (mean of has_ai_ml)
        #  - median position among positive positions
        q_all = df.groupby('question_id').agg(
            total_responses=('has_ai_ml', 'count'),
            ai_ml_count=('has_ai_ml', 'sum')
        )
        q_all['ai_ml_pct'] = (q_all['ai_ml_count'] / q_all['total_responses'] * 100).round(1)

        median_pos = df[(df['has_ai_ml']) & (df['ai_ml_position'] >= 1)].groupby('question_id')['ai_ml_position'].median().rename('median_position')
        q_all = q_all.join(median_pos).fillna({'median_position': -1}).astype({'median_position': int})

        # Also provide canonical vs paraphrase mean (combine paraphrases by mean)
        if 'variant' in df.columns:
            df['variant_type'] = df['variant'].apply(lambda x: 'canonical' if str(x) == 'canonical' else 'paraphrase')
            canonical_mean = df[df['variant_type'] == 'canonical'].groupby('question_id')['has_ai_ml'].mean().rename('canonical_mean')
            paraphrase_mean = df[df['variant_type'] == 'paraphrase'].groupby('question_id')['has_ai_ml'].mean().rename('paraphrase_mean')
            q_all = q_all.join(canonical_mean).join(paraphrase_mean)
            # fill NaN means with 0.0 where absent
            q_all['canonical_mean'] = q_all['canonical_mean'].fillna(0.0).round(3)
            q_all['paraphrase_mean'] = q_all['paraphrase_mean'].fillna(0.0).round(3)

        print(q_all.to_string())
    else:
        print("No 'question_id' column found; skipping per-question aggregation.")

    # Position distribution (including -1 for none)
    print("\n=== AI/ML Position Distribution ===")
    pos_counts = df['ai_ml_position'].value_counts().sort_index()
    # Present -1 as 'none'
    for pos, cnt in pos_counts.items():
        label = 'none' if pos == -1 else str(pos)
        print(f"Position {label}: {cnt}")

    # Examples (show more samples if present)
    examples = df[df['has_ai_ml']].head(300)
    if not examples.empty:
        print("\n=== Sample responses that mention AI/ML ===")
        for _, row in examples.iterrows():
            qid = row.get('question_id', '<no-question>')
            model = row.get('model', '<no-model>')
            variant = row.get('variant', '<no-variant>')
            pos = row['ai_ml_position']
            print(f"Question {qid} | Model {model} | Variant {variant} | Position {pos}")
    else:
        print("\nNo examples with AI/ML found.")


if __name__ == '__main__':
    main()

```

