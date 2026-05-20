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

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_file> <general|translation>" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: config file '$1' does not exist" >&2
    exit 1
fi

if [ "$2" != "general" ] && [ "$2" != "translation" ]; then
    echo "Error: second argument must be 'general' or 'translation' (got '$2')" >&2
    exit 1
fi

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

LANGS="amh ber chr grn haw ibo npi oci que yor zgh zh"
# DRAFT="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"

# for draft in $DRAFT
# do
    for lang in $LANGS
    do
        uv run scripts/distill.py "$1" \
            -o language_code=$lang \
            output_dir="/scratch/alpine/$USER/spec-dec/" \
            dataset_path="logprobs/logprobs-Qwen3.5-9B-$lang-$2.parquet" \
            task=$2
            # draft_model=$draft \
    done
# done
