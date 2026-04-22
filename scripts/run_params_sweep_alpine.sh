#!/usr/bin/env bash
#
# SLURM job-array sweep for distillation hyperparameter search.
#
# Each array task = one (lr, steps, grad_accum, warmup, weight_decay) combo.
# All tasks run in parallel (up to Alpine's 21-GPU per-user limit on aa100).
#
# Quick sweep  (8 combos):  each run ~30-45 min  → done in 1-2 hours total
# Full  sweep (324 combos): each run ~30-60 min  → done in ~3-5 hours total
#                           (21 jobs run simultaneously, so 324/21 ≈ 16 waves)
#
# ── Quick sweep (8 runs) ────
#   sbatch --array=0-7   scripts/run_params_sweep_alpine.sh
#
# ── Full sweep (324 runs) ────
#   sbatch --array=0-323 scripts/run_params_sweep_alpine.sh
#
# ── Override language / mode ─────
#   sbatch --array=0-323 --export=LANG_CODE=oci,MODE=general \
#          scripts/run_params_sweep_alpine.sh
#
# ── Multiple languages (each language = its own array batch) ───
#   for lang in ber npi haw smo; do
#     sbatch --array=0-323 --export=LANG_CODE=$lang,MODE=general \
#            --job-name=distill-${lang} scripts/run_params_sweep_alpine.sh
#   done
#

#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=ucb-general
#SBATCH --job-name=distill-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nirajan.paudel@colorado.edu

# %A = job array master ID, %a = this task's index
# Limit concurrent running tasks to 21 (Alpine aa100 per-user GPU cap).
# Set via --array=0-323%21 or override at submit time.

set -euo pipefail

# ── Defaults (override via --export) ───
LANG_CODE="${LANG_CODE:-chr}"
MODE="${MODE:-general}"
EXTRA="${EXTRA:-}"

# ── Hyperparameter grids ───
#
# The array index maps to one unique combo using integer arithmetic.
# Grid sizes must match --array=0-N (N = product of all grid sizes minus 1).
#
#   Quick  (--array=0-7):    2 x 2 x 2 x 1 x 1 =   8
#   Full   (--array=0-323):  4 x 3 x 3 x 3 x 3 = 324
#
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

if [[ "$TOTAL_TASKS" -le 8 ]]; then
    # Quick grid
    LR_GRID=(3e-5 5e-5)
    STEPS_GRID=(2000 3000)
    GA_GRID=(4 8)
    WARMUP_GRID=(0.06)
    WD_GRID=(0.01)
else
    # Full grid
    LR_GRID=(1e-5 3e-5 5e-5 8e-5)
    STEPS_GRID=(1500 2000 3000)
    GA_GRID=(4 8 16)
    WARMUP_GRID=(0.03 0.06 0.10)
    WD_GRID=(0.0 0.01 0.05)
fi

N_WD="${#WD_GRID[@]}"
N_WARMUP="${#WARMUP_GRID[@]}"
N_GA="${#GA_GRID[@]}"
N_STEPS="${#STEPS_GRID[@]}"

WD_IDX=$(( TASK_ID % N_WD ))
WARMUP_IDX=$(( (TASK_ID / N_WD) % N_WARMUP ))
GA_IDX=$(( (TASK_ID / (N_WD * N_WARMUP)) % N_GA ))
STEPS_IDX=$(( (TASK_ID / (N_WD * N_WARMUP * N_GA)) % N_STEPS ))
LR_IDX=$(( (TASK_ID / (N_WD * N_WARMUP * N_GA * N_STEPS)) ))

LR="${LR_GRID[$LR_IDX]}"
STEPS="${STEPS_GRID[$STEPS_IDX]}"
GA="${GA_GRID[$GA_IDX]}"
WARMUP="${WARMUP_GRID[$WARMUP_IDX]}"
WD="${WD_GRID[$WD_IDX]}"

LABEL="lr${LR}_s${STEPS}_ga${GA}_wu${WARMUP}_wd${WD}"

module purge
module load uv

source .venv/bin/activate

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export PYTHONPATH="$(pwd)"
export HF_HOME="/scratch/alpine/${USER}/hf_cache"
export WANDB_DIR="/scratch/alpine/${USER}/wandb"
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

export WANDB_DISTILL_SWEEP_TAG="grid_search"

if [[ "$MODE" == "general" ]]; then
    CONFIG="experiments/general_kd.cfg"
else
    CONFIG="experiments/seqkd.cfg"
fi

OUTPUT_DIR="/scratch/alpine/${USER}/distilled_models/sweep/${MODE}/${LANG_CODE}/${LABEL}"

echo "======================================================================"
echo "  Array job:  ${SLURM_ARRAY_JOB_ID}, task ${TASK_ID}/${TOTAL_TASKS}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Language:   ${LANG_CODE} | Mode: ${MODE}"
echo "  Combo:      lr=${LR} steps=${STEPS} ga=${GA} warmup=${WARMUP} wd=${WD}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Start:      $(date)"
echo "======================================================================"

OVERRIDES="language_code=${LANG_CODE} \
           output_dir=${OUTPUT_DIR} \
           hf_repo_id=lecslab \
           learning_rate=${LR} \
           max_steps=${STEPS} \
           grad_accum_steps=${GA} \
           warmup_ratio=${WARMUP} \
           weight_decay=${WD}"

if [[ -n "${EXTRA}" ]]; then
    OVERRIDES="${OVERRIDES} ${EXTRA}"
fi

PYTHONPATH="$(pwd)" python scripts/distill.py "${CONFIG}" -o ${OVERRIDES}

echo ""
echo "  Done: $(date)"
echo "=========================="

unset WANDB_DISTILL_SWEEP_TAG 2>/dev/null || true
