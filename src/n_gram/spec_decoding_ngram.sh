#!/usr/bin/env bash
# SBATCH --job-name=n-gram
# SBATCH --partition=aa100
# SBATCH --qos=normal
# SBATCH --gres=gpu:1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=16
# SBATCH --mem=120G
# SBATCH --time=2:00:00
# SBATCH --output=n_gram.log
# SBATCH --output=n_gram.err
# SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO_ROOT="/projects/lude4390/spec-decoding"
cd "$REPO_ROOT"

module purge

source "$REPO_ROOT/.venv/bin/activate"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
    echo "Loaded tokens from .env"
else
    echo "ERROR: .env not found! Copy .env.example to .env and fill in your tokens."
    echo "  cp .env.example .env"
    exit 1
fi

export HF_HOME="/scratch/alpine/${USER}/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Repo root:     $REPO_ROOT"
echo "HF cache:      $HF_HOME"
echo "Python:        $(which python)"
echo "Started:       $(date)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

bash experiments/run_baseline_spec.sh

echo ""
echo "Finished: $(date)"
