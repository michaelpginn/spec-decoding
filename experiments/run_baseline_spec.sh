#!/usr/bin/env bash


# *********************
#  Run full baseline + spec (Qwen, Llama, Aya). Use from REPO root:
#    bash experiments/run_baseline_spec.sh

# set up the wandb and the huggingface before running the script
# also, ensure that access to gated models has been granted
# *********************

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LANGS="ber chr haw ibo lkt mus npi oci oji que yua zgh"
BASELINE_CFG="experiments/baseline.cfg"
SPEC_CFG="experiments/spec_greedy.cfg"

FAILED=0
TOTAL=0

run_one() {
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$TOTAL] $*"
    echo "────────────────────────────────────────"
    "$@" || { echo "WARN: FAILED"; FAILED=$((FAILED + 1)); }
}

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 1: 7B target + 0.5B draft, gamma = 3, 5, 7
# ══════════════════════════════════════════════════════════════

# Baseline for 7B (all languages, runs once)
echo "###  Baseline — Qwen 7B  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen2.5-7B-Instruct \
           max_samples=50
done

# Spec: 7B + 0.5B, gamma = 3, 5, 7 (all languages)
echo "###  Spec — 7B + 0.5B, gamma=3,5,7  ###"
for gamma in 3 5 7; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen2.5-7B-Instruct \
               draft_model=Qwen/Qwen2.5-0.5B-Instruct \
               gamma=$gamma \
               max_samples=50
    done
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2: 32B target + 0.5B draft, gamma = 3, 5, 7
# ══════════════════════════════════════════════════════════════

# Baseline for 32B (all languages, runs once)
echo "###  Baseline — Qwen 32B ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen2.5-32B-Instruct \
           max_samples=50
done

# Spec: 32B + 0.5B, gamma = 3, 5, 7 (all languages)
echo "###  Spec — 32B + 0.5B, gamma=3,5,7  ###"
for gamma in 3 5 7; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen2.5-32B-Instruct \
               draft_model=Qwen/Qwen2.5-0.5B-Instruct \
               gamma=$gamma \
               max_samples=50
    done
done

# Spec: 32B + 1.5B, gamma = 3, 5, 7 
echo "###  Spec — 32B + 1.5B, gamma=3,5,7 ###"
for gamma in 3 5 7; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=Qwen/Qwen2.5-32B-Instruct \
               draft_model=Qwen/Qwen2.5-1.5B-Instruct \
               gamma=$gamma \
               max_samples=50
    done
done


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Quick test — 0.5B + 0.5B, gamma=5, few languages
# ══════════════════════════════════════════════════════════════

# Only 3 languages: ber, chr, npi.
echo "###  Quick test — 0.5B + 0.5B, gamma=5, ber/chr/npi  ###"
for lang in ber chr npi; do
    run_one python run.py $SPEC_CFG \
        -o language_code=$lang \
           target_model=Qwen/Qwen2.5-0.5B-Instruct \
           draft_model=Qwen/Qwen2.5-0.5B-Instruct \
           gamma=5 \
           max_samples=50
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Llama 3.1 8B target + Llama 3.2 1B draft
# ══════════════════════════════════════════════════════════════

# Baseline for Llama 8B (all languages, runs once)
echo "###  Baseline — Llama 3.1 8B  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=meta-llama/Llama-3.1-8B-Instruct \
           max_samples=50
done

# Spec: Llama 8B + 1B, gamma = 3, 5, 7 (all languages)
echo "###  Spec — Llama 8B + 1B, gamma=3,5,7  ###"
for gamma in 3 5 7; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=meta-llama/Llama-3.1-8B-Instruct \
               draft_model=meta-llama/Llama-3.2-1B-Instruct \
               gamma=$gamma \
               max_samples=50
    done
done

# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 5: CohereLabs/aya-expanse-32b + CohereLabs/aya-expanse-8b draft
# ══════════════════════════════════════════════════════════════

# # Baseline for Aya 32b (all languages, runs once)
echo "###  Baseline — CohereLabs/aya-expanse-32b  ###"
for lang in $LANGS; do
    run_one python run.py $BASELINE_CFG \
        -o language_code=$lang \
           target_model=CohereLabs/aya-expanse-32b \
           max_samples=50
done


echo "###  Spec — CohereLabs/aya-expanse-32b + CohereLabs/aya-expanse-8b, gamma=3,5,7  ###"
for gamma in 3 5 7; do
    for lang in $LANGS; do
        run_one python run.py $SPEC_CFG \
            -o language_code=$lang \
               target_model=CohereLabs/aya-expanse-32b \
               draft_model=CohereLabs/aya-expanse-8b \
               gamma=$gamma \
               max_samples=50
    done
done

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " DONE: $((TOTAL - FAILED))/$TOTAL succeeded"
[[ $FAILED -gt 0 ]] && echo " FAILED: $FAILED"
echo "============================================"
