#!/bin/bash
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=ngram-ab
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

# *****************************************************************************
#  A/B check: n-gram drafting, old dense path vs new sparse path.
#  Same GPU, same job, arms interleaved, so alpha and draft time are comparable.
#
#  Usage (from repo root):
#    sbatch scripts/cl2_ngram_ab.sh
#
#  Expected in W&B (tag ngram-ab, split by config.ngram_sparse_drafting):
#    greedy: identical acceptance rate and outputs in both arms
#    sample: acceptance rates within noise; sparse has lower average_draft_time
# *****************************************************************************

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
mkdir -p $HF_HOME
export WANDB_DIR="/scratch/alpine/$USER/wandb"
mkdir -p $WANDB_DIR

module load uv
uv sync

echo "=== q-equivalence unit test (CPU) ==="
uv run python tests/test_ngram_sparse.py || { echo "unit test FAILED, stopping"; exit 1; }

# que: where the Mac runs showed the largest sampled-alpha gap.
# sample: full 400-sentence test set (alpha SE ~0.007 per arm, enough to see a 0.02 gap).
# greedy: outputs must match exactly, so 100 sentences (max_samples=500 -> 20% test split) is plenty.
# Target is overridden to Qwen3.5-2B: this checks the draft code, so a small target is enough (and faster)
LANG=que
GAMMA=3

for sparse in False True
do
    uv run python run.py experiments/ngram.cfg \
        -o target_model=Qwen/Qwen3.5-2B language_code=$LANG decoding_mode=sample gamma=$GAMMA \
        ngram_sparse_drafting=$sparse wandb_tag=ngram-ab
    uv run python run.py experiments/ngram.cfg \
        -o target_model=Qwen/Qwen3.5-2B language_code=$LANG decoding_mode=greedy gamma=$GAMMA max_samples=500 \
        ngram_sparse_drafting=$sparse wandb_tag=ngram-ab
done
