#!/bin/bash
#SBATCH --job-name=recommendations_answer_to_prompts_personas
#SBATCH --output=/home/fast/trabelb1/projects/Bias-Recommendations/answer_prompts_personas%j.out
#SBATCH --error=/home/fast/trabelb1/projects/Bias-Recommendations/answer_prompts_personas%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
source ~/.bash_profile
REPO_LOCATION="/home/fast/trabelb1/projects/Bias-Recommendations"
nvidia-smi
cd $REPO_LOCATION
export PYTHONPATH=$(pwd)

#uv run main.py --model "microsoft/phi-4"
#uv run main.py --model "Qwen/Qwen3-30B-A3B"
#uv run main.py --model "Qwen/Qwen3-32B"
#uv run main.py --model "mistralai/Ministral-3-14B-Instruct-2512"

uv run run_personas.py --model "microsoft/phi-4"
uv run run_personas.py --model "Qwen/Qwen3-30B-A3B"
uv run run_personas.py --model "Qwen/Qwen3-32B"

send_telegram "job ${SLURM_JOB_ID} is done"
echo "Done"
