#!/usr/bin/env bash
# Slurm array job: n-gram speculative decoding sweep (8 tasks = n={2,3} x gamma={2,3,4,5})
#   sbatch --array=0-7 --export=LANG_CODE=ber scripts/ngram_job.sh
#   sbatch --array=0-7 --export=LANG_CODE=ber,DECODING_MODE=greedy scripts/ngram_job.sh
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ngram_%A_%a.out
#SBATCH --error=logs/ngram_%A_%a.err

set -euo pipefail

LANG_CODE="${LANG_CODE:-ber}"
DECODING_MODE="${DECODING_MODE:-sample}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Map task index (0–7) → (n, gamma): n ∈ {2,3}, gamma ∈ {2,3,4,5}
N=$(( TASK_ID / 4 + 2 ))
G=$(( TASK_ID % 4 + 2 ))

module purge
module load uv

source .venv/bin/activate

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export HF_HOME="/scratch/alpine/${USER}/hf_cache"
export WANDB_DIR="/scratch/alpine/${USER}/wandb"
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

echo "========================================="
echo "  Array job: ${SLURM_ARRAY_JOB_ID} / task ${TASK_ID}"
echo "  Language:  ${LANG_CODE}  |  n=${N}  |  gamma=${G}  |  mode=${DECODING_MODE}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"
echo "  Start:     $(date)"
echo "========================================="

python run.py experiments/ngram.cfg \
    -o language_code="${LANG_CODE}" \
       ngram_n="${N}" \
       gamma="${G}" \
       decoding_mode="${DECODING_MODE}"

echo ""
echo "  Done: $(date)"
echo "=========================="
