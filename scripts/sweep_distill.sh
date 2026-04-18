#!/usr/bin/env bash
#
# Two-stage hyperparameter sweep for distillation.
#
# Stage 1 (--stage 1): Sweep learning rate with default steps/grad_accum.
#   Pick the best LR from wandb, then run stage 2.
#
# Stage 2 (--stage 2): Fix LR (--lr), sweep max_steps x grad_accum_steps.
#
# Usage:
#   # Stage 1: sweep LR for general KD on Berber
#   bash scripts/sweep_distill.sh --mode general --lang ber --stage 1
#
#   # Stage 2: fix best LR, sweep steps x grad_accum
#   bash scripts/sweep_distill.sh --mode general --lang ber --stage 2 --lr 5e-5
#
#   # Task-specific (SeqKD) — needs seqkd_data_path override
#   bash scripts/sweep_distill.sh --mode task_specific --lang ber --stage 1 \
#       --extra "seqkd_data_path=lecslab/seqkd-Qwen2.5-7B-Instruct-ber-5000"
#
#   # Stage 1 for several languages at once:
#   for lang in ber npi haw; do
#     bash scripts/sweep_distill.sh --mode general --lang "$lang" --stage 1
#   done
#
# W&B: each stage sets WANDB_DISTILL_SWEEP_TAG so runs are tagged
#   learning_rate_sweep_runs (stage 1) or steps_grad_accum_sweep (stage 2).
#
set -euo pipefail

MODE="general"
LANG_CODE="ber"
STAGE=1
BEST_LR=""
EXTRA_OVERRIDES=""
HF_REPO_ID="${HF_REPO_ID:-lecslab}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  MODE="$2";       shift 2 ;;
        --lang)  LANG_CODE="$2";  shift 2 ;;
        --stage) STAGE="$2";      shift 2 ;;
        --lr)    BEST_LR="$2";    shift 2 ;;
        --extra) EXTRA_OVERRIDES="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "general" ]]; then
    CONFIG="experiments/general_kd.cfg"
else
    CONFIG="experiments/seqkd.cfg"
fi

SWEEP_BASE="../distilled_models/sweep/${MODE}/${LANG_CODE}"

run_one() {
    local label="$1"
    shift
    local overrides="$*"

    local output_dir="${SWEEP_BASE}/${label}"
    local full_overrides="language_code=${LANG_CODE} output_dir=${output_dir} hf_repo_id=${HF_REPO_ID} ${overrides}"

    if [[ -n "$EXTRA_OVERRIDES" ]]; then
        full_overrides="${full_overrides} ${EXTRA_OVERRIDES}"
    fi

    echo "============================================"
    echo "  Run: ${label}"
    echo "  Config: ${CONFIG}"
    echo "  Overrides: ${full_overrides}"
    echo "============================================"

    python scripts/distill.py "${CONFIG}" -o ${full_overrides}
}

if [[ "$STAGE" == "1" ]]; then
    export WANDB_DISTILL_SWEEP_TAG="learning_rate_sweep_runs"
    echo "=== STAGE 1: Learning Rate Sweep ==="
    echo "Mode: ${MODE} | Language: ${LANG_CODE}"
    echo ""

    for lr in 1e-5 2e-5 5e-5; do
        run_one "lr${lr}" "learning_rate=${lr}"
    done

    echo ""
    echo "Stage 1 complete. Check wandb to pick the best LR, then run:"
    echo "  bash scripts/sweep_distill.sh --mode ${MODE} --lang ${LANG_CODE} --stage 2 --lr <best_lr>"

elif [[ "$STAGE" == "2" ]]; then
    if [[ -z "$BEST_LR" ]]; then
        echo "ERROR: Stage 2 requires --lr <best_lr> from stage 1"
        exit 1
    fi

    export WANDB_DISTILL_SWEEP_TAG="steps_grad_accum_sweep"
    echo "=== STAGE 2: Steps x Grad Accum Sweep (LR=${BEST_LR}) ==="
    echo "Mode: ${MODE} | Language: ${LANG_CODE}"
    echo ""

    for steps in 2000 3000 5000; do
        for ga in 4 8 16; do
            run_one "lr${BEST_LR}_steps${steps}_ga${ga}" \
                "learning_rate=${BEST_LR} max_steps=${steps} grad_accum_steps=${ga}"
        done
    done

    echo ""
    echo "Stage 2 complete. Check wandb to pick the best combo."

else
    echo "ERROR: --stage must be 1 or 2"
    exit 1
fi

unset WANDB_DISTILL_SWEEP_TAG 2>/dev/null || true
