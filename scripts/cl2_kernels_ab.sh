#!/bin/bash
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=kernels-ab
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

# *****************************************************************************
#  A/B check: fast kernels (fla + FlashAttention-2) vs plain torch, same code otherwise.
#  Submit once from each checkout:
#    torch arm: a checkout of ngram-optim (cache fix, no fla)  -> sbatch scripts/cl2_kernels_ab.sh torch
#    fla arm:   ngram-optim with fla-kernels merged in         -> sbatch scripts/cl2_kernels_ab.sh fla
#  The script refuses to run if the installed kernels don't match the arm.
#
#  Expected in W&B (tags kernels-ab-torch / kernels-ab-fla):
#    acceptance rates within noise (bf16 kernels change logits slightly, so not identical)
#    lower average_verifier_time (and neural average_draft_time) with fla
# *****************************************************************************

ARM=${1:?usage: sbatch scripts/cl2_kernels_ab.sh torch|fla}

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
mkdir -p $HF_HOME
export WANDB_DIR="/scratch/alpine/$USER/wandb"
mkdir -p $WANDB_DIR

module load uv
uv sync

echo "=== kernel check (arm: $ARM) ==="
ARM=$ARM uv run python - <<'PY' || exit 1
import os, torch
import transformers.models.qwen3_5.modeling_qwen3_5 as m
print("GPU:", torch.cuda.get_device_name(0))
has_fla = m.chunk_gated_delta_rule is not None
print("fla linear-attention kernels:", has_fla)
if has_fla != (os.environ["ARM"] == "fla"):
    raise SystemExit(f"arm is {os.environ['ARM']!r} but fla installed = {has_fla}; wrong checkout?")
PY

# 100 test sentences per language (max_samples=500 -> 20% test split): this checks
# alpha and timing, not final numbers. que = LRL, zh = HRL.
for lang in que zh
do
    uv run python run.py experiments/spec_sampled.cfg \
        -o language_code=$lang gamma=3 max_samples=500 wandb_tag=kernels-ab-$ARM
    uv run python run.py experiments/ngram.cfg \
        -o language_code=$lang gamma=3 max_samples=500 wandb_tag=kernels-ab-$ARM
done
