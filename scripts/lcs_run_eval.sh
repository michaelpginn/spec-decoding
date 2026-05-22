#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=specdec
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --mail-type=END,FAIL

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
mkdir -p $HF_HOME
export WANDB_DIR="/scratch/alpine/$USER/wandb"
mkdir -p $WANDB_DIR

module load uv
uv sync

echo "=== CUDA + PyTorch diagnostics ==="
uv run python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

LANGS="amh ber chr grn haw ibo npi oci que yor zgh zh"
GAMMAS="2 3 4"
# DRAFT="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"

# for draft in $DRAFT
# do
    for lang in $LANGS
    do
        for gamma in $GAMMAS
        do
            uv run python run.py "$1" \
                -o language_code=$lang \
                gamma=$gamma \
                wandb_tag=final
        done
    done
# done
