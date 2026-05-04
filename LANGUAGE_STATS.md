# Language Data Stats for N-gram Speculative Decoding

Model: `Qwen/Qwen2.5-0.5B-Instruct` | Tokenizer vocab: 151,936
Generated: 2026-05-03 | Script: `scripts/stats_langs.py`

## Table (2,000 mono test-sample, 10 languages)

| Lang | Name | Bi pairs (ref) | Bi pairs (loaded) | Mono sent (ref) | Mono sent (loaded) | 2-grams | 3-grams | Ratio (3/2) | Avg tok/sent |
|---|---|---|---|---|---|---|---|---|---|
| ber | Berber | 353,517 | 400/100 | 52,834 | 1,600/400 | 8,415 | 13,967 | 1.7x | 16 |
| que | Quechua | 2,874,369 | 400/100 | 175,843 | 1,600/400 | 8,108 | 14,386 | 1.8x | 20 |
| amh | Amharic | 195,608 | 400/100 | 0† | 1,600/400 | 41,943 | 66,635 | 1.6x | 114 |
| grn | Guarani | 986 | 400/100 | 10,000 | 1,600/400 | 20,995 | 37,534 | 1.8x | 41 |
| ibo | Igbo | 1,749 | 400/100 | 56,097 | 1,600/400 | 18,361 | 38,378 | 2.1x | 57 |
| yor | Yoruba | 20,100 | 400/100 | 0† | 1,600/400 | 24,335 | 41,010 | 1.7x | 53 |
| npi | Nepali | 3,917 | 400/100 | 6,395,621 | 1,600/400 | 5,750 | 35,038 | **6.1x** | 428 |
| chr | Cherokee | 11,468 | 400/100 | 31† | 1,600/400 | 1,865 | 7,009 | **3.8x** | 100 |
| zgh | Tamazight | 15,760 | 400/100 | 20,720 | 1,600/400 | 8,997 | 29,215 | **3.2x** | 1,072 |
| zh | Chinese | 1,000,000 | 400/100 | 958,000,000 | 1,600/400 | 724,879 | 1,251,654 | 1.7x | 926 |

† No monolingual data in reference table — entirely dependent on CohereLabs/aya_dataset fallback.

### Dropped

| Lang | Reason |
|---|---|
| haw (Hawaiian) | 0 usable n-grams — all mono sentences < 3 tokens |
| oci (Occitan) | Avg 2.7 tok/sent — sentences too short for meaningful n-gram |
| lkt (Lakota) | No monolingual data available |
| mus (Muskogee) | No bilingual or monolingual data available |
| oji (Ojibwe) | No monolingual data available |
| yua (Maya) | Bug in language name lookup (`assemble_dataset` uses bilingual table for name, but yua only has mono entries) |

## Column Definitions

| Header | What it measures |
|---|---|
| **Bi pairs (ref)** | Total translation pairs across all sources in `reference_table_bilingual.csv` |
| **Bi pairs (loaded)** | `train/test` sampled at 80/20 split (capped at `max_samples=500`) |
| **Mono sent (ref)** | Total monolingual sentences in `reference_table_monolingual.csv` († = Aya fallback only) |
| **Mono sent (loaded)** | `train/test` sampled at 80/20 split (capped at `max_samples=2000`) |
| **N-grams** | Number of unique n-gram contexts found in loaded mono training data |
| **Ratio (3/2)** | How much bigger trigram vocabulary is vs bigram — high ratio = long/varied sentences |
| **Avg tok/sent** | Mean tokens per monolingual sentence after tokenization. Ideal: 10–80 |

## Analysis: n=2 vs n=3

The n-gram model stores its dict in **CPU RAM** (not GPU VRAM). A 1M-entry trigram dict is ~200MB. Cluster nodes have 64GB+ CPU RAM, so even 10M+ entries are safe.

## Cluster Sweep Plan

### Fixed parameters
| Parameter | Value | Notes |
|---|---|---|
| `target_model` | `Qwen/Qwen2.5-7B-Instruct` | |
| `max_samples` (bi) | `6000` | 1200 test examples per run (80/20 split) |
| `max_samples_mono` | `20000` | Default from config — same for all languages |
| `max_new_tokens` | `512` | |
| `decoding_mode` | `greedy` | |
| `top_k` / `top_p` | `0` / `0.0` | No sampling filters |
| `wandb_tag` | `ngram-sweep` | Groups all runs in the sweep |

### Swept parameters
| Languages | n | gamma |
|---|---|---|
| amh, ber, chr, grn, ibo, npi, que, yor, zgh, zh | 2, 3 | 2, 3, 4, 5 |

**Total runs: 10 × 2 × 4 = 80 jobs**

### Gamma sweep rationale
`gamma` controls how many draft tokens are proposed per iteration. Higher gamma → more speculative tokens → lower per-token acceptance (draft has less context ahead). The sweet spot depends on the draft model quality — n-gram drafts degrade quickly with distance, so the optimal gamma is likely lower (2–3) vs. neural drafts (5+). Sweeping 2–5 covers the full range.

### How to submit (Alpine / CURC)
Use the submission wrapper — each language = 1 sbatch array of 8 jobs:

```bash
# Pre-reqs (do once):
#   git clone this repo to /projects/$USER/spec-decoding
#   module load uv && uv sync
#   huggingface-cli login && wandb login

# Submit language by language:
bash scripts/submit_ngram_sweep.sh ber
bash scripts/submit_ngram_sweep.sh amh
bash scripts/submit_ngram_sweep.sh chr
# ... etc ...

bash scripts/submit_ngram_sweep.sh zh

# Monitor: squeue -u $USER
```

### What each job runs (for reference)
```bash
uv run python run.py experiments/ngram.cfg \
    -o language_code=$LANG ngram_n=$N gamma=$G
```
