#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=specdec
#SBATCH --partition=blanca-clearlab1
#SBATCH --account=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --mail-type=END,FAIL

export HF_HOME="/projects/$USER/.cache/huggingface"
mkdir -p $HF_HOME

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

LANGS="ber chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn"
GAMMAS="1 2 3 4 5 6 7"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct"

for draft in $DRAFT
do
    for lang in $LANGS
    do
        for gamma in $GAMMAS
        do
            uv run python run.py "$1" \
                -o language_code=$lang \
                draft_model=$draft \
                gamma=$gamma \
                wandb_tag=final
        done
    done
done
