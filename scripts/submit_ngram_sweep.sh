#!/usr/bin/env bash
# Submit 8-job ngram sweep for one language. Defaults to sample decoding.
#   bash scripts/submit_ngram_sweep.sh ber          # sample (default)
#   bash scripts/submit_ngram_sweep.sh ber greedy   # greedy override

set -euo pipefail

LANG_CODE="$1"
DECODING_MODE="${2:-sample}"
if [ -z "$LANG_CODE" ]; then
    echo "Usage: bash scripts/submit_ngram_sweep.sh <language_code> [mode]"
    echo "  mode: sample (default) | greedy"
    echo "Example: bash scripts/submit_ngram_sweep.sh ber"
    echo "         bash scripts/submit_ngram_sweep.sh ber greedy"
    exit 1
fi

echo "Submitting ngram sweep for ${LANG_CODE} (mode=${DECODING_MODE})..."

sbatch --array=0-7 \
       --export=LANG_CODE="${LANG_CODE}",DECODING_MODE="${DECODING_MODE}" \
       --job-name="ngram-${LANG_CODE}" \
       scripts/ngram_job.sh

echo "Done. Check with: squeue -u ${USER}"
