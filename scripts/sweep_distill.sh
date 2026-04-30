#!/usr/bin/env bash
#
# Full grid search for distillation hyperparameters.
#
# Sweeps over: learning_rate x max_steps x grad_accum_steps x warmup_ratio x weight_decay
# Each run trains from scratch and logs to wandb with the tag "grid_search".
#
# Usage:
#   # General KD, full grid search for Berber
#   bash scripts/sweep_distill.sh --mode general --lang ber
#
#   # Task-specific (SeqKD) — needs seqkd_data_path override
#   bash scripts/sweep_distill.sh --mode task_specific --lang ber \
#       --extra "seqkd_data_path=lecslab/seqkd-Qwen2.5-7B-Instruct-ber-5000"
#
#   # Quick sweep (fewer combos, good for initial exploration)
#   bash scripts/sweep_distill.sh --mode general --lang ber --quick
#
#   # Multiple languages:
#   for lang in ber npi haw; do
#     bash scripts/sweep_distill.sh --mode general --lang "$lang" &
#   done
#
set -euo pipefail

MODE="general"
LANG_CODE="ber"
EXTRA_OVERRIDES=""
HF_REPO_ID="${HF_REPO_ID:-lecslab}"
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  MODE="$2";       shift 2 ;;
        --lang)  LANG_CODE="$2";  shift 2 ;;
        --extra) EXTRA_OVERRIDES="$2"; shift 2 ;;
        --quick) QUICK=1;         shift   ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "general" ]]; then
    CONFIG="experiments/general_kd.cfg"
else
    CONFIG="experiments/seqkd.cfg"
fi

SWEEP_BASE="../distilled_models/sweep/${MODE}/${LANG_CODE}"
export WANDB_DISTILL_SWEEP_TAG="grid_search"

# ── Hyperparameter grids ─────────────────────────────────────────────────
if [[ "$QUICK" == "1" ]]; then
    LR_GRID=(3e-5 5e-5)
    STEPS_GRID=(2000 3000)
    GA_GRID=(4 8)
    WARMUP_GRID=(0.06)
    WD_GRID=(0.01)
else
    LR_GRID=(1e-5 3e-5 5e-5 8e-5)
    STEPS_GRID=(1500 2000 3000)
    GA_GRID=(4 8 16)
    WARMUP_GRID=(0.03 0.06 0.10)
    WD_GRID=(0.0 0.01 0.05)
fi

TOTAL=0
for lr in "${LR_GRID[@]}"; do
for steps in "${STEPS_GRID[@]}"; do
for ga in "${GA_GRID[@]}"; do
for warmup in "${WARMUP_GRID[@]}"; do
for wd in "${WD_GRID[@]}"; do
    TOTAL=$((TOTAL + 1))
done; done; done; done; done

echo "============================================================"
echo "  Grid search: ${MODE} | ${LANG_CODE}"
echo "  LR:       ${LR_GRID[*]}"
echo "  Steps:    ${STEPS_GRID[*]}"
echo "  GradAcc:  ${GA_GRID[*]}"
echo "  Warmup:   ${WARMUP_GRID[*]}"
echo "  WD:       ${WD_GRID[*]}"
echo "  Total runs: ${TOTAL}"
echo "============================================================"

run_one() {
    local label="$1"
    shift
    local overrides="$*"

    local output_dir="${SWEEP_BASE}/${label}"
    local full_overrides="language_code=${LANG_CODE} output_dir=${output_dir} hf_repo_id=${HF_REPO_ID} ${overrides}"

    if [[ -n "$EXTRA_OVERRIDES" ]]; then
        full_overrides="${full_overrides} ${EXTRA_OVERRIDES}"
    fi

    echo "------------------------------------------------------------"
    echo "  Run: ${label}"
    echo "  Overrides: ${full_overrides}"
    echo "------------------------------------------------------------"

    PYTHONPATH="$(pwd)" python scripts/distill.py "${CONFIG}" -o ${full_overrides}
}

RUN_NUM=0
for lr in "${LR_GRID[@]}"; do
for steps in "${STEPS_GRID[@]}"; do
for ga in "${GA_GRID[@]}"; do
for warmup in "${WARMUP_GRID[@]}"; do
for wd in "${WD_GRID[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    LABEL="lr${lr}_s${steps}_ga${ga}_wu${warmup}_wd${wd}"
    echo ""
    echo ">>> Run ${RUN_NUM}/${TOTAL}: ${LABEL}"
    run_one "$LABEL" \
        "learning_rate=${lr} max_steps=${steps} grad_accum_steps=${ga} warmup_ratio=${warmup} weight_decay=${wd}"
done; done; done; done; done

echo ""
echo "============================================================"
echo "  Grid search complete. ${TOTAL} runs finished."
echo "  Check wandb (tag: grid_search) to find the best combo."
echo "  Then evaluate with:"
echo "    python scripts/eval_distill_sweep.py --sweep-dir ${SWEEP_BASE} --language ${LANG_CODE}"
echo "============================================================"

unset WANDB_DISTILL_SWEEP_TAG 2>/dev/null || true
