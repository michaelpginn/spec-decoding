#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=specdec
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

if [ -z "$1" ]; then
    echo "Error: No config file provided. Usage: sbatch run_multiple_ngram.sh path/to/config.yaml"
    exit 1
fi
CONFIG_FILE=$1

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

cd ..

langs=("ber" "chr" "haw" "ibo" "amh" "yor" "oci" "grn" "yua" "zgh")

for item in "${langs[@]}"; do
    echo "-----------------------------------"
    echo "running ${item}"
    uv run python testing_story.py "$CONFIG_FILE" --override language_code=$item
done
