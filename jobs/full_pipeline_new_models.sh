#!/bin/bash
#SBATCH --job-name=recommendations_det_only
#SBATCH --output=/home/fast/trabelb1/projects/Bias-Recommendations/slurm_det_%j.out
#SBATCH --error=/home/fast/trabelb1/projects/Bias-Recommendations/slurm_det_%j.err
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

set -euo pipefail

REPO_LOCATION="/home/fast/trabelb1/projects/Bias-Recommendations"
cd "$REPO_LOCATION"
export PYTHONPATH="$(pwd)"
# For bug with flash attention with gemma-2.
#export VLLM_FLASH_ATTN_VERSION=2
#export VLLM_DISABLE_COMPILE_CACHE=1

nvidia-smi

# -------------------------
# Config
# -------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
OUT_DIR="data/pipeline_${RUN_ID}"
mkdir -p "$OUT_DIR"

DET_TEMP="0.0"
DET_TOPP="1.0"
DET_N="1"
DET_SEED="12345"
DET_MAXTOK="384"
K="5"

# Model list
MODELS=(
#  "openai/gpt-oss-20b"
#  "openai/gpt-oss-120b"
#  "Qwen/Qwen3-32B"
#  "Qwen/Qwen3-Next-80B-A3B-Instruct"
#   "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#  "meta-llama/Llama-3.3-70B-Instruct"
#  "google/gemma-3-27b-it"
#  "01-ai/Yi-1.5-34B-Chat"
#  "dphn/dolphin-2.9.1-yi-1.5-34b"
#  "mistralai/Ministral-3-14B-Instruct-2512"
#  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
)

sanitize_model_id() {
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

# -------------------------
# Stage 1: Deterministic
# -------------------------
for model in "${MODELS[@]}"; do
  smodel="$(sanitize_model_id "$model")"
  out="${OUT_DIR}/responses_${smodel}_paraphrases_det.json"
  
  # Hardcoded list for TP=2
  TP_ARGS=""
  case "$model" in
    "openai/gpt-oss-120b" | \
    "Qwen/Qwen3-Next-80B-A3B-Instruct" | \
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" | \
    "meta-llama/Llama-3.3-70B-Instruct" | \
    "mistralai/Mixtral-8x22B-Instruct-v0.1")
      TP_ARGS="--tensor-parallel-size 2"
      ;;
  esac

  echo "--------------------------------------------------"
  echo "RUNNING: $model"
  [[ -n "$TP_ARGS" ]] && echo "Using $TP_ARGS"
  echo "--------------------------------------------------"

  uv run main.py \
    --model "$model" \
    --include-paraphrases \
    --temperature "$DET_TEMP" --top-p "$DET_TOPP" \
    --n "$DET_N" --seed "$DET_SEED" \
    --max-tokens "$DET_MAXTOK" \
    $TP_ARGS \
    --enforce-eager \
    --out "$out"
done

# -------------------------
# Evaluation
# -------------------------
echo "==================== EVALUATING RESULTS ========================"

uv run eval_ai_mentions.py \
  --input "$OUT_DIR" \
  --pattern "responses_*_paraphrases_det.json" \
  --k "$K" | tee "${OUT_DIR}/eval_det_rowlevel.txt"

uv run eval_ai_mentions.py \
  --input "$OUT_DIR" \
  --pattern "responses_*_paraphrases_det.json" \
  --k "$K" \
  --aggregate-by-qid | tee "${OUT_DIR}/eval_det_byqid.txt"

# Telegram summary
SUMMARY=$(
  {
    echo "Deterministic Pipeline Done (Job ${SLURM_JOB_ID})"
    echo "OUT_DIR=${OUT_DIR}"
    echo ""
    echo "=== Summary (by-qid) ==="
    sed -n '1,15p' "${OUT_DIR}/eval_det_byqid.txt" || true
  } | head -c 3500
)

send_telegram "$SUMMARY"
echo "Done."
