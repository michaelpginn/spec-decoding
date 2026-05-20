#!/usr/bin/env bash

# *****************************************************************************
#  Run full baseline + speculative decoding experiments (Qwen, Llama, Aya).
#
#  Usage (from repo root):
#    bash experiments/run_baseline_spec.sh
#
#  Prerequisites:
#    - wandb and huggingface-cli must be logged in
#    - Access to gated models (Llama, etc.) must be granted
# *****************************************************************************

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ─────────────────────────────────────────────
LANGS="ber chr haw ibo lkt mus npi oci oji que yua zgh"
BASELINE_CFG="experiments/baseline.cfg"
SPEC_CFG="experiments/spec_greedy.cfg"
MAX_SAMPLES=100
GAMMAS="3 5 7"

FAILED=0
TOTAL=0

run_one() {
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] $*"
    echo "────────────────────────────────────────"
    "$@" || { echo "WARN: FAILED"; FAILED=$((FAILED + 1)); }
}

# ───────────────────────────────────────────────────────────────
#  EXPERIMENT 1 — Sanity check: Qwen 0.5B + 0.5B (same model)
# ───────────────────────────────────────────────────────────────

echo "###  Quick test — Qwen 0.5B + 0.5B, gamma=5, zgh/chr/npi  ###"
for lang in ber chr npi; do
    run_one python run.py $SPEC_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen3.5-0.8B \
           draft_model=Qwen/Qwen3.5-0.8B \
           gamma=5 \
           max_samples=$MAX_SAMPLES
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — Qwen 7B target + 0.5B draft
# ══════════════════════════════════════════════════════════════

echo "###  Baseline — Qwen 7B  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen3.5-9B \
           max_samples=$MAX_SAMPLES
done

echo "###  Spec — Qwen 7B + 0.5B, gamma=$GAMMAS  ###"
for gamma in $GAMMAS; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen3.5-9B \
               draft_model=Qwen/Qwen3.5-0.8B \
               gamma=$gamma \
               max_samples=$MAX_SAMPLES
    done
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — Qwen 32B target + 0.5B and 1.5B drafts
# ══════════════════════════════════════════════════════════════

echo "###  Baseline — Qwen 32B  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen3.5-27B \
           max_samples=$MAX_SAMPLES
done

# --- 32B + 0.5B draft ---
echo "###  Spec — Qwen 32B + 0.5B, gamma=$GAMMAS  ###"
for gamma in $GAMMAS; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen3.5-27B \
               draft_model=Qwen/Qwen3.5-0.8B \
               gamma=$gamma \
               max_samples=$MAX_SAMPLES
    done
done

# --- 32B + 1.5B draft ---
echo "###  Spec — Qwen 32B + 1.5B, gamma=$GAMMAS  ###"
for gamma in $GAMMAS; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen3.5-27B \
               draft_model=Qwen/Qwen3.5-2B \
               gamma=$gamma \
               max_samples=$MAX_SAMPLES
    done
done

# --- 32B + 3B draft ---
echo "###  Spec — Qwen 32B + 3B, gamma=$GAMMAS  ###"
for gamma in $GAMMAS; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen3.5-27B \
               draft_model=Qwen/Qwen3.5-4B \
               gamma=$gamma \
               max_samples=$MAX_SAMPLES
    done
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — Llama 3.1 8B target + Llama 3.2 1B draft
# ══════════════════════════════════════════════════════════════

echo "###  Baseline — Llama 3.1 8B  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=meta-llama/Llama-3.1-8B-Instruct \
           max_samples=$MAX_SAMPLES
done

echo "###  Spec — Llama 8B + 1B, gamma=$GAMMAS  ###"
for gamma in $GAMMAS; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=meta-llama/Llama-3.1-8B-Instruct \
               draft_model=meta-llama/Llama-3.2-1B-Instruct \
               gamma=$gamma \
               max_samples=$MAX_SAMPLES
    done
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 5 — Aya Expanse 32B target + 8B draft
# ══════════════════════════════════════════════════════════════

AYA_LANGS="zgh npi haw que"

echo "###  Baseline — Aya Expanse 32B  ###"
for lang in $AYA_LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=CohereLabs/aya-expanse-32b \
           max_samples=$MAX_SAMPLES
done

echo "###  Spec — Aya 32B + 8B, gamma=5  ###"
for lang in $AYA_LANGS; do
    run_one python run.py $SPEC_CFG \
        -o language_code=$lang \
           target_model=CohereLabs/aya-expanse-32b \
           draft_model=CohereLabs/aya-expanse-8b \
           gamma=5 \
           max_samples=$MAX_SAMPLES
done

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " DONE: $((TOTAL - FAILED))/$TOTAL succeeded"
[[ $FAILED -gt 0 ]] && echo " FAILED: $FAILED"
echo "============================================"
