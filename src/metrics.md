# Metrics Reference

N = number of sentences (max_samples). All times exclude prompt prefill.

---

## Summary Metrics (derived from all N sentences)

### Speed
| Metric | Formula | Unit |
| --- | --- | --- |
| avg_time_per_sentence | sum(sentence times) / N | seconds |
| median_time_per_sentence | median(sentence times) | seconds |
| avg_time_per_token | mean of (time_i / tokens_i) for each sentence | seconds |
| tokens_per_second | sum(all generated tokens) / sum(all sentence times) | tok/s |
| sentence_avg_tokens_per_second | mean of per-sentence tok/s — each sentence weighted equally | tok/s |
| sentence_std_tokens_per_second | sample stdev of per-sentence tok/s (N ≥ 2) | tok/s |
| average_draft_time | mean time for single forward pass | s |
| average_verifier_time | mean time for single forward pass | s |
| draft_time_std | standard deviation of draft forward pass times | s |
| verifier_time_std | standard deviation of verifier forward pass times | s |
| draft_time_count | total number of draft forward passes | count |
| verifier_time_count | total number of verifier forward passes | count |


### Quality
| Metric | Formula | Range |
| --- | --- | --- |
| bleu | SacreBLEU corpus score over all N translations vs references | 0–100 |
| chrf2 | chrF2 corpus score (character n-gram F-score, beta=2) | 0–100 |

### Speculative Decoding Efficiency (spec runs only)
| Metric | Formula | Range |
| --- | --- | --- |
| token_weighted_acceptance_rate | sum(matched tokens) / sum(draft tokens) across all N sentences — longer sentences have more influence | 0.0–1.0 |
| sentence_avg_acceptance_rate | mean of per-sentence acceptance rates — each sentence weighted equally | 0.0–1.0 |
| mean_accepted_tokens | sum(matched tokens) / sum(iterations) across all N sentences | 0–gamma |
| block_efficiency | mean_accepted_tokens / gamma | 0.0–1.0 |
| draft_to_output_ratio | sum(draft tokens) / sum(generated tokens) across all N sentences | >= 1.0 |
| speedup_factor | Expected time without spec decoding / with | >= 0.0 |
| speedup_factor_std | Std of speedup via delta method on α and c=T_d/T_v (N ≥ 2, CUDA only) | ≥ 0.0 |
| sentence_std_acceptance_rate | sample stdev of per-sentence acceptance rates (N ≥ 2) | 0.0–1.0 |
| sentence_std_mean_accepted_tokens | sample stdev of per-sentence mean accepted tokens (N ≥ 2) | ≥ 0.0 |
---

## Per-Sentence Charts (sentence/*, one value per sentence, x-axis = sentence_idx)

| Metric | What it measures | Unit |
| --- | --- | --- |
| sentence/time | Decode wall-clock time for this sentence | seconds |
| sentence/time_per_token | sentence/time / sentence/generated_tokens | seconds |
| sentence/generated_tokens | Number of new tokens produced for this sentence | count |
| sentence/draft_tokens | (spec only) Draft tokens proposed for this sentence | count |
| sentence/matched_tokens | (spec only) Draft tokens accepted by target for this sentence | count |
| sentence/acceptance_rate | (spec only) matched_tokens / draft_tokens for this sentence | 0.0–1.0 |

---

## Timing Notes
All time metrics measure **decode time only** — new token generation, excluding prompt prefill.

- **Baseline**: Prefill measured via a separate forward pass, subtracted from generate() wall time  
- **Spec decode**: Timer starts after prefill + first token, covers only the speculative loop
