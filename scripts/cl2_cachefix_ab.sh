#!/bin/bash
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=cachefix-ab
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

# *****************************************************************************
#  A/B check: how much the Qwen3.5 linear-attention cache fix changes the results.
#  Same code, same GPU, same job; only linear_cache_rewind differs:
#    False = old crop-only rollback (linear-attention states left stale after a rejection)
#    True  = fix (states restored from a snapshot)
#  Both arms use the same per-sentence seeds, so each sentence starts from the same RNG state.
#
#  Usage (from repo root):
#    sbatch scripts/cl2_cachefix_ab.sh            # amh (best-performing language in the paper)
#    sbatch scripts/cl2_cachefix_ab.sh zh         # any other language
#
#  Then compare the two arms (tag cachefix-ab):
#    uv run python scripts/cachefix_ab_report.py
#
#  Both tasks: translation (400 test sentences, chrF/BLEU too) and story generation (200 prompts).
#  Story outputs are long (128 tokens) with many rejections, so if stale states matter it should
#  show most there; the octile acceptance rates show whether the error compounds.
# *****************************************************************************

LANG_CODE=${1:-amh}
GAMMA=3
SEED=1234

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
mkdir -p $HF_HOME
export WANDB_DIR="/scratch/alpine/$USER/wandb"
mkdir -p $WANDB_DIR

module load uv
uv sync

for cfg in experiments/spec_sampled.cfg experiments/spec_sampled_story.cfg
do
    for rewind in False True
    do
        uv run python run.py $cfg \
            -o language_code=$LANG_CODE gamma=$GAMMA seed=$SEED \
            linear_cache_rewind=$rewind wandb_tag=cachefix-ab
    done
done
