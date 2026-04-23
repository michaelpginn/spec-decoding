#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=4:00:00
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
CONFIG_PATH=$1

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

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

cd ..

langs=("chr" "amh" "yor" "npi" "grn" "yua")

for item in "${langs[@]}"; do
    echo "running ${item}"
    if [[ $item == "yor" || $item == "amh" ]]; then
        uv run python run.py "$CONFIG_PATH" --override language_code=$item include_aya=True
    else
        uv run python run.py "$CONFIG_PATH" --override language_code=$item include_aya=False
    fi
done
