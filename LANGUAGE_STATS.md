# Language Data Stats for N-gram Speculative Decoding

Model: `Qwen/Qwen2.5-0.5B-Instruct` | Tokenizer vocab: 151,936
Generated: 2026-05-04 | Script: `scripts/stats_langs.py`

## Table (2,000 mono samples, 12 languages)

| Lang | Name | Bi pairs (ref) | Bi pairs (loaded) | Mono sent (ref) | Mono sent (loaded) | 2-grams | 3-grams | Ratio (3/2) | Avg tok/sent |
|---|---|---|---|---|---|---|---|---|---|
| amh | Amharic | 195,608 | 400/100 | 0† | 1,600/400 | 41,943 | 66,635 | 1.6x | 114 |
| ber | Berber | 353,517 | 400/100 | 52,834 | 1,600/400 | 8,415 | 13,967 | 1.7x | 16 |
| chr | Cherokee | 11,468 | 400/100 | 31† | 1,600/400 | 1,865 | 7,009 | 3.8x | 100 |
| grn | Guarani | 986 | 400/100 | 10,000 | 1,600/400 | 20,995 | 37,534 | 1.8x | 41 |
| haw | Hawaiian | 121 | 96/25 | 206,456 | 1,600/400 | 1,405 | 2,232 | 1.6x | 4.5 |
| ibo | Igbo | 1,749 | 400/100 | 56,097 | 1,600/400 | 18,361 | 38,378 | 2.1x | 57 |
| npi | Nepali | 3,917 | 400/100 | 6,395,621 | 1,600/400 | 5,750 | 35,038 | 6.1x | 428 |
| oci | Occitan | 4,540 | 400/100 | 581,690 | 1,600/400 | 1,985 | 1,069 | **0.5x** | 2.7 |
| que | Quechua | 2,874,369 | 400/100 | 175,843 | 1,600/400 | 8,108 | 14,386 | 1.8x | 20 |
| yor | Yoruba | 20,100 | 400/100 | 0† | 1,600/400 | 24,335 | 41,010 | 1.7x | 53 |
| zgh | Tamazight | 15,760 | 400/100 | 20,720 | 1,600/400 | 8,997 | 29,215 | 3.2x | 1,072 |
| zh | Chinese | 1,000,000 | 400/100 | 958,000,000 | 1,600/400 | 724,879 | 1,251,654 | 1.7x | 926 |

† No monolingual data in reference table — entirely dependent on CohereLabs/aya_dataset fallback.

### Marginal cases

| Lang | Issue |
|---|---|
| haw (Hawaiian) | Avg 4.5 tok/sent, only 2,232 trigrams. N-gram model has very limited context — drafts may be weak. |
| oci (Occitan) | Avg 2.7 tok/sent, trigram vocab *smaller* than bigram (0.5x ratio). Sentences too short for n=3 to help. Likely n=2 only, or skip. |

### Dropped

| Lang | Reason |
|---|---|
| lkt (Lakota) | No monolingual data available |
| mus (Muskogee) | No bilingual or monolingual data available |
| oji (Ojibwe) | No monolingual data available |
| yua (Maya) |

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

Most languages benefit from trigrams (ratio ≥ 1.6x). Exceptions:
- **npi** (6.1x): very long sentences, trigrams add substantial context diversity
- **oci** (0.5x): trigrams *shrink* — sentences shorter than 3 tokens destroy n=3

### Fixed parameters
| Parameter | Value |
|---|---|
| `target_model` | `Qwen/Qwen2.5-7B-Instruct` |
| `max_samples` (bi) | `6000` (1200 test per run) |
| `max_samples_mono` | `20000` |
| `max_new_tokens` | `512` |
| `wandb_tag` | `ngram-sweep` |

### Swept parameters
| Parameter | Values |
|---|---|
| languages | amh, ber, chr, grn, haw, ibo, npi, oci, que, yor, zgh, zh |
| ngram_n | 2, 3 |
| gamma | 2, 3, 4, 5 |
| decoding_mode | greedy, sample |

**Total: 12 × 2 × 4 × 2 = 192 jobs** (96 per decoding mode)