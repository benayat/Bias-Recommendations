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

```bash
python -m pip install --upgrade pip
pip install -r <(python - <<'PY'
import tomllib,sys
p=tomllib.loads(open('pyproject.toml','rb').read())
for d in p['project']['dependencies']: print(d)
PY
)
```

Or install explicitly (recommended):

```bash
pip install openai pandas transformers vllm
```

Notes:
- `vllm` requires a CUDA-capable GPU and compatible drivers when using local GPU-backed models.
- `transformers` may be used for tokenization and some model backends.
- If you plan to use the OpenAI-compatible client, install `openai` (already in pyproject) and set `OPENAI_API_KEY` in your environment.

Environment variables you may need:
- `OPENAI_API_KEY` — if using the `openai_llm` client.


## Running the pipeline

All commands should be run from repository root (paths inside the code are relative to project root).

1) Generate model responses (this is the most time- and compute-intensive step):

```bash
python main.py
```

This will iterate `MODELS_LIST` and every canonical + paraphrase from `constants/questions.py`. Outputs are saved to `data/responses.csv`.

2) Analyze responses with a small analysis model (strict JSON output required):

```bash
python scripts/analyze_responses.py
```

On success this writes `data/analyzed_responses.csv` with the two structured columns appended: `has_ai_ml` and `ai_ml_position`.

3) Print aggregated statistics:

```bash
python scripts/analyze_stats.py
```

This script prints overall statistics, per-model breakdowns (with per-question details inside each model), and per-question aggregates with paraphrases combined by mean.


## Switching to the OpenAI-compatible client

If you prefer to run analysis using an OpenAI-compatible API instead of the local vLLM client, use the client in `openai_llm/openai_client.py`. It uses the same interface (`LLMClient`, `run_batch`, `delete_client`, `SamplingConfig`) so swapping should be straightforward.

Example (conceptual):

```python
from openai_llm import LLMClient as OpenAILLMClient, OpenAIConfig, SamplingConfig
cfg = OpenAIConfig(api_key='YOUR_KEY')
client = OpenAILLMClient(model_name='gpt-4', config=cfg)
# use client.run_batch(...) same as vLLM-backed client
```

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
