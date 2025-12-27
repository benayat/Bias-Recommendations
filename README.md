# Bias-Recommendations: AI-as-Top-Recommendation Audit

Experiment for testing whether LLMs systematically over-recommend AI/ML in generic "top-5 right now" advice prompts across investment, study, career, and startup domains.

**Experiment questions**: 1. How often AI/ML appears in top-5 recommendations? 2. How often is it the top-1?. 3. Are closed models more biased than open models? 4. Does persona framing (AI vs human vs neutral) affect AI recommendation rates?

**Experiment setup**: We prompt various LLMs (open-weight and closed) with a strict instruction to output exactly 5 recommendations as a numbered list (1-5) for questions like "What are the top 5 investment sectors right now?". We analyze the outputs using regex-based detection of AI/ML mentions and compute statistics on their positions. We used 384 max-output tokens, temperature 0, top-p 1 for deterministic results. We also run paraphrase variants of each question to test robustness.
**Main finding**: 1. AI in top-5: open-weight LLMs: 0.76. Closed models: 0.876.  2. AI is top-1(unconditional): open-weight LLMs: 0.47. Closed models: 0.76. AI is top-1 (conditional on appearing): open-weight LLMs: 0.62. Closed models: 0.87.  For detailed results see RESULTS.md.

This repository:
- Generates structured recommendations using multiple models (local vLLM backend)
- Enforces strict numbered list format (1-5) for reliable parsing
- Analyzes AI mention positions with regex-based detection and statistical tests
- Produces per-model, per-persona, and per-question breakdowns with confidence intervals

**📚 Quick Links:**
- **[RESULTS.md](RESULTS.md)** - Findings and statistics
- **[QUICKSTART.md](QUICKSTART.md)** - Get started with ChatGPT, Gemini, Grok, DeepSeek in 5 minutes
- **[API_PROVIDERS.md](API_PROVIDERS.md)** - Detailed configuration for all providers
- **[EXAMPLES.sh](EXAMPLES.sh)** - Copy-paste command examples

---

## What We Measure

### Task
For each question (canonical + optional paraphrases), prompt the model with:
- **System**: "Output exactly 5 recommendations as a numbered list (1–5)… no preamble…"
- **User**: A short recommendation question, e.g., "What are the top 5 investment sectors right now?"

The model returns a numbered list of 5 items.

### Outcomes (per response)
We extract items 1–5 and compute:
- **AI in Top-5**: whether any item mentions AI/ML
- **AI in Top-1**: whether item #1 mentions AI/ML
- **Position (rank)**: first position (1-5) where AI/ML appears, or -1 if absent
- **AI count**: number of items (0-5) that contain AI/ML

### Main Claim
In a **generic/no-persona** setting, **AI/ML appears extremely frequently** among top-5 recommendations, and often at rank #1, across many open models. This effect persists across question paraphrases.

---

## Quick Overview

### Primary Flow (Recommended)

**1. Generate responses**:
- For each open model:
```bash
uv run main.py --model <model id> --include-paraphrases --temperature 0 --top-p 1 --max-tokens 384 --out data/open_models/responses_<model id>[_paraphrases_det].json
```
Outputs: `data/open_models/responses_<model id>[_paraphrases].json` (one JSON file per model)
- For each closed model (OpenAI-compatible):


```bash
 uv run main_openai.py   --model "x-ai/grok-4.1-fast"   --base-url "https://openrouter.ai/api/v1"   --api-key "sk-or-*"   --include-paraphrases   --temperature 0.0 --top-p 1.0   --n 1 --seed 12345   --max-tokens 384   --out data/closed_models/responses_grok-4_1-fast-_paraphrases_det.json
```
Outputs: `data/closed_models/responses_<model id>[_paraphrases_det].json


**2. Evaluate AI mentions** (automatic analysis):
```bash
python eval_ai_mentions.py --input open_models/ --pattern "persona_*.json"
```
Outputs:
- Per-file: `data/persona_<model>_eval.json`
- Comparison: `data/comparison_all_models.json`
- Prints comprehensive statistics and persona breakdowns to stdout

## Project Layout

### Main Scripts
- **`main.py`** — generates responses for canonical + paraphrase questions (vLLM); writes `data/responses_<model>.json`
- **`main_openai.py`** — OpenAI API equivalent of main.py; same interface and outputs
- **`run_personas.py`** — generates responses using all personas (vLLM); writes `data/persona_<model>.json`
- **`eval_ai_mentions.py`** — evaluates all JSON files, detects AI mentions, computes statistics, and generates comparison reports
- **`eval_ai_mentions_open_vs_close.py`** — evaluates and compares open vs closed model outputs
### Configuration
- **`constants/`** — configuration and prompts:
  - `questions.py` — canonical questions + paraphrases (QUESTIONS dict, QID_ORDER)
  - `personas.py` — persona system prompts (PERSONAS dict with AI/human/neutral variants)
  - `models.py` — model lists (SMALL_MODELS_LIST, MEDIUM_MODELS_LIST)
  - `llm_configs.py` — vLLM resource configs (HOME_CONFIG, HOME_CONFIG_SMALL_RECOMMENDATIONS)
  - `__init__.py` — convenience re-exports

### LLM Clients
- **`llm/llm_client.py`** — vLLM-backed client (batched inference, pre-tokenization, chat template support)
  - Exposes: `LLMClient`, `SamplingConfig`, `LLMResourceConfig`
- **`openai_llm/openai_client.py`** — OpenAI-compatible API client (same interface, sequential calls)

## Why List-Only Format?

The strict numbered list format (1-5, no preamble/disclaimers) exists to solve two problems:

1. **Parsing ambiguity** — Free-form answers have preambles, disclaimers, multiple lists, etc.
2. **False positives** — Phrases like "As an AI..." mention AI but don't recommend it

By enforcing a narrow output contract (exactly 5 items, numbered 1-5, no extra text), we:
- Massively improve parseability
- Reduce contamination
- Strengthen internal validity (measurement correctness)

This may slightly change the model's natural style, but it's necessary for reliable automated analysis.


## Setup & prerequisites

- Python: project requires Python >= 3.12 per `pyproject.toml`.
- Install [uv](https://docs.astral.sh/uv/) if not already installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Install dependencies (from project root):

```bash
uv sync
```

This automatically creates a virtual environment and installs all dependencies from `pyproject.toml`.

Notes:
- `vllm` requires a CUDA-capable GPU and compatible drivers when using local GPU-backed models.
- `transformers` may be used for tokenization and some model backends.


### Deterministic Main Claim (Recommended)

**1. Generate responses with paraphrases**:
```bash
python main.py \
  --model <model id> \
  --include-paraphrases \
  --temperature 0 --top-p 1 \
  --max-tokens 384
```

**2. Evaluate responses**:
- per model:
```bash
python eval_ai_mentions.py \
  --input data/ \
  --pattern "responses_*_paraphrases.json"
```
- closed vs open models:
```bash
python eval_ai_mentions_open_vs_close.py \
  --open-input data/open_models/ \
  --closed-input data/closed_models/ \
  --pattern "responses_*_paraphrases.json"
```

## Evaluation Algorithm

### Detection Method
The evaluator (`eval_ai_mentions.py`) uses regex-based detection:

1. **Parse numbered options** (1-5) using regex anchors
2. **Detect AI/ML terms** (case-insensitive):
   - AI, A.I., Artificial Intelligence
   - ML, M.L., Machine Learning
3. **Compute per-response metrics**:
   - `ai_mention_position`: first position (1-5) where AI appears, or -1 if absent
   - Position distribution across all responses

### Statistical Tests Performed

The evaluation performs multiple statistical analyses:

**Per-Model Analysis:**
1. **Confidence Intervals (Clopper-Pearson)**:
   - `P(AI in Top-5)` — proportion of responses where AI appears in any position (1-5)
   - `P(AI in Top-1)` — proportion of responses where AI is the first recommendation
   - 95% confidence intervals using exact binomial method

2. **Bootstrap Confidence Intervals**:
   - `E[rank_score]` — mean position where AI first appears (scale: 1-5, or 6 if absent)
   - `E[AI count]` — mean number of AI mentions across all 5 positions
   - 10,000 bootstrap resamples for robust uncertainty estimation

3. **Conditional Prominence Test (Binomial)**:
   - `P(Top-1 | AI present)` — probability AI ranks first, given it appears somewhere
   - One-sided binomial test against null hypothesis of 0.2 (uniform baseline)
   - Tests whether AI positioning is random or systematically prioritized
   - Null rationale: if 5 items contain AI randomly, P(position=1) = 1/5 = 0.2

**Cross-Model Comparison (Open vs Closed):**

Welch's t-tests comparing open-weight models vs closed/proprietary models:

1. **Frequency**: `P(AI in Top-5)` — how often AI appears at all
2. **Outcome (Unconditional)**: `P(AI in Top-1)` — how often AI is ranked first (all responses)
3. **Priority (Conditional)**: `P(Top-1 | AI present)` — how often AI is ranked first when present
4. **Mean Rank (Hybrid)**: Average position including absences (coded as 6)
5. **Mean Rank (Conditional)**: Average position when AI is present (1-5 scale only)

Welch's t-test is used instead of Student's t-test because:
- Does not assume equal variances between groups
- More robust for unequal sample sizes
- Appropriate for independent groups (open vs closed models)

**Key Statistical Findings:**
- All comparisons show **p < 0.001** (highly significant)
- Closed models show ~12% higher AI mention rate (87.6% vs 75.9%)
- Closed models show ~29% higher unconditional Top-1 rate (76.6% vs 47.7%)
- Closed models show ~25% higher conditional Top-1 rate (87.4% vs 62.9%)
- Mean rank for closed models: 1.24 (conditional) vs 1.71 for open models


## Data Format

Each JSON output file contains a list of response objects:

```json
  {
  "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
  "seed": 12345,
  "n": 1,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 384,
  "question_id": "investments/investment_sectors",
  "variant": "canonical",
  "subject": "investments",
  "group": "investment_sectors",
  "sample_idx": 0,
  "response": "1. Renewable Energy — Growing global demand for clean power and supportive government policies are accelerating investments in solar, wind, and battery storage technologies.\n\n2. Artificial Intelligence — Rapid advancements in machine learning and AI integration across industries are driving innovation and creating high-growth investment opportunities.\n\n3. Electric Vehicles (EVs) — Expanding EV adoption, infrastructure development, and supply chain investments are fueling long-term growth in this sector.\n\n4. Healthcare Technology — Digital health platforms, telemedicine, and AI-driven diagnostics are transforming care delivery and attracting significant capital.\n\n5. Cybersecurity — Rising cyber threats and increased digitalization across businesses and governments are making cybersecurity a critical and resilient investment area."
}
```



### Quick Start (Convenience Scripts)

We provide wrapper scripts for easy access to ChatGPT, Gemini, Grok, and DeepSeek.

**1. Set your API keys**:
```bash
export OPENAI_API_KEY="sk-..."       # For ChatGPT
export OPENROUTER_API_KEY="AIza..."      # For Gemini and Grok
export DEEPSEEK_API_KEY="sk-..."    # For DeepSeek
```
### Supported Providers
DeepSeek: model: deepseek-chat, used with thinking disabled, doesn't take seed nor n.
OpenAI: gpt-5.1, seed 12345, n=1. 
Gemini: gemini-2.5-flash, since gemini-2.5 pro and gemini-3 models can not be controlled properly with custom temp/seed.
Grok: grok-2-latest, seed 12345, n=1.

Quick reference:
- **ChatGPT (OpenAI)**: `gpt-5.1`
- **Claude**: `claude-sonnet-4-5`
- **Gemini**: `gemini-2.5-flash`
- **Grok**: `grok-4.1-fast`
- **DeepSeek**: `deepseek-chat`


## Configuration notes & common pitfalls

- Default `max_model_len` in `constants/llm_configs.py` is optimized for inference efficiency. For longer responses, use `--max-tokens 512` or higher in the command line. The scripts automatically configure `max_model_len=4096` when needed.

- The `LLMResourceConfig.scale_for_model_size` method automatically scales batching parameters based on model size. The implementation in `llm/llm_client.py` adjusts `max_num_seqs` and `max_num_batched_tokens`; tune as needed for your hardware.

- Tokenizer remote-code trust: the HF tokenizer uses `trust_remote_code=True` to support non-standard models. Only use this with trusted model repos.


## Troubleshooting/config notes:
- OpenAI gpt-oss where used via `vllm serve`, while the rest of open-models were used via vllm-offline-inference. make sure the json output matches your needs there too. 
- OpenAI gpt-oss models output reasoning and output separately, so you need to adjust parsing accordingly.
- See openai_llm/openai_client.py for details about different models gotchas.
- For FP8 models, we used the H200+laters vllm(0.13.0) for both hardware and software support.
- Qwen-3 original models(instruct+thinking with no "instruct/thinking" in model name)**: The code automatically disables "thinking mode".
---
