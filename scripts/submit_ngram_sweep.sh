#!/usr/bin/env bash
# Submit all ngram sweep jobs for a given language (8 jobs: n=2,3 × gamma=2,3,4,5)
# Uses defaults from experiments/ngram.cfg — max_samples_mono=20000, etc.
#
# Usage:  bash scripts/submit_ngram_sweep.sh ber
#         bash scripts/submit_ngram_sweep.sh zh

set -euo pipefail

LANG_CODE="$1"
if [ -z "$LANG_CODE" ]; then
    echo "Usage: bash scripts/submit_ngram_sweep.sh <language_code>"
    echo "Example: bash scripts/submit_ngram_sweep.sh ber"
    exit 1
fi

echo "Submitting ngram sweep for ${LANG_CODE}..."

sbatch --array=0-7 \
       --export=LANG_CODE="${LANG_CODE}" \
       --job-name="ngram-${LANG_CODE}" \
       scripts/ngram_job.sh

echo "Done. Check with: squeue -u ${USER}"
