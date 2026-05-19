#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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

for lang in amh ber chr grn haw ibo npi oci que yor zgh zh
do
    uv run scripts/generate_teacher_logprobs.py "$1" \
        -o language_code=$lang
done
