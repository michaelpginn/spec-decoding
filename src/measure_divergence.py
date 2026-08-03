"""Computes the KL divergence (P||Q) and total variation distance between a
target model P and a draft model Q, on monolingual text.

Reports each divergence two ways:

  * per token   - averaged over scored token positions. This is the unit the
    speculative-decoding acceptance rate lives in, but it is NOT comparable
    across languages: a script missing from the vocabulary falls back to UTF-8
    bytes, and the resulting continuation-byte positions are near-deterministic
    for both models, so they contribute ~0 and dilute the average.
  * per character - the same divergence mass divided by the characters it
    covers. This is the bits-per-byte convention used for cross-tokenizer
    comparison, and is what to use when comparing across languages.

Also reports each model's bits per character, a tokenizer-invariant measure of
how well the model models the language.

This is a bit redundant with the eval_kl script, we should probably unify later
"""

import argparse
import csv
import math
from logging import getLogger

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data.dataset import assemble_dataset, get_language_name, LANGUAGES
from src.utils import load_model

logger = getLogger(__name__)

MAX_MONO = 20000

FIELDNAMES = [
    "language_code", "language",
    "kl_per_token", "tvd_per_token",
    "kl_per_char", "tvd_per_char",
    "bits_per_char_target", "bits_per_char_draft",
    "n_examples", "n_chars",
]

parser = argparse.ArgumentParser()
parser.add_argument("--p", help="HF model key to use for P distribution (target)")
parser.add_argument("--q", help="HF model key to use for Q distribution (draft)")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-length", type=int, default=128)
parser.add_argument("--output", default=None, help="defaults to viz/divergences_<P>_<Q>.csv")
args = parser.parse_args()

p_model, p_tokenizer = load_model(args.p)
q_model, q_tokenizer = load_model(args.q)
if not p_tokenizer.pad_token:
    p_tokenizer.pad_token = p_tokenizer.eos_token

# Verify matching tokenizers
if p_tokenizer.get_vocab() != q_tokenizer.get_vocab():
    raise ValueError(
        f"{args.p} and {args.q} do not share a tokenizer; this script feeds P's "
        f"token ids to both models."
    )
device = next(p_model.parameters()).device

divergences = []

for language_code in LANGUAGES:
    language = get_language_name(language_code)
    logger.info(f"Running on {language}")
    dataset = assemble_dataset(language_code, 'mono', p_tokenizer, MAX_MONO)['test']
    dataloader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    p_model.eval()
    q_model.eval()

    total_kl = 0.
    total_lk = 0.
    n_scored = 0
    n_skipped = 0
    # Per-character quantities are pooled over the corpus (total mass / total
    # characters), the standard bits-per-character convention, rather than
    # averaged over per-example ratios, which short examples would dominate.
    sum_kl = 0.
    sum_tvd = 0.
    sum_bits_p = 0.
    sum_bits_q = 0.
    sum_chars = 0
    for batch in tqdm(dataloader):
        inputs = p_tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True).to(device)
        with torch.no_grad():
            p_out = p_model(**inputs)
            q_out = q_model(**inputs)
            # float32 before the softmax: bf16 loses ~1% of the probability mass
            # over a 150k+ vocabulary.
            p_logprobs = torch.nn.functional.log_softmax(p_out.logits[..., :-1, :].float(), dim=-1)
            q_logprobs = torch.nn.functional.log_softmax(q_out.logits[..., :-1, :].float(), dim=-1)
            kl = (torch.exp(p_logprobs) * (p_logprobs - q_logprobs)).sum(-1)
            lk = (1/2 * torch.abs(p_logprobs.exp() - q_logprobs.exp())).sum(-1)
            label_mask = inputs['attention_mask'][:, 1:]
            kl = kl * label_mask
            lk = lk * label_mask

            # Negative log-likelihood of the token actually observed, for BPC.
            labels = inputs['input_ids'][:, 1:]
            p_token_lp = p_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * label_mask
            q_token_lp = q_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * label_mask

            # Characters covered by the scored positions: decode tokens 1..n-1,
            # i.e. exactly the tokens being predicted, after truncation.
            n_valid = inputs['attention_mask'].sum(dim=-1)
            chars = torch.tensor(
                [
                    len(p_tokenizer.decode(inputs['input_ids'][i, 1:int(n)], skip_special_tokens=True))
                    for i, n in enumerate(n_valid)
                ],
                device=device,
            )

            # Drop single-token examples
            scored = label_mask.sum(dim=-1)
            keep = scored > 0
            total_kl += (kl.sum(dim=-1)[keep] / scored[keep]).sum().item()
            total_lk += (lk.sum(dim=-1)[keep] / scored[keep]).sum().item()
            n_scored += int(keep.sum().item())
            n_skipped += int((~keep).sum().item())

            # Pooled per-character totals, over examples that have both a scored
            # position and at least one character.
            keep_c = keep & (chars > 0)
            sum_kl += kl.sum(dim=-1)[keep_c].sum().item()
            sum_tvd += lk.sum(dim=-1)[keep_c].sum().item()
            sum_bits_p += -p_token_lp.sum(dim=-1)[keep_c].sum().item() / math.log(2)
            sum_bits_q += -q_token_lp.sum(dim=-1)[keep_c].sum().item() / math.log(2)
            sum_chars += int(chars[keep_c].sum().item())

    if n_skipped:
        logger.warning(f"{language}: skipped {n_skipped} example(s) with no scored positions")
    if n_scored == 0 or sum_chars == 0:
        logger.warning(f"{language}: no scorable examples, omitting from output")
        continue

    row = {
        "language_code": language_code,
        "language": language,
        "kl_per_token": total_kl / n_scored,
        "tvd_per_token": total_lk / n_scored,
        "kl_per_char": sum_kl / sum_chars,
        "tvd_per_char": sum_tvd / sum_chars,
        "bits_per_char_target": sum_bits_p / sum_chars,
        "bits_per_char_draft": sum_bits_q / sum_chars,
        "n_examples": n_scored,
        "n_chars": sum_chars,
    }
    logger.info(
        f"  KL/token={row['kl_per_token']:.4f}  TVD/token={row['tvd_per_token']:.4f}  "
        f"KL/char={row['kl_per_char']:.4f}  TVD/char={row['tvd_per_char']:.4f}  "
        f"BPC(target)={row['bits_per_char_target']:.4f}  BPC(draft)={row['bits_per_char_draft']:.4f}"
    )
    divergences.append(row)

p_model_name = args.p.split("/")[-1]
q_model_name = args.q.split("/")[-1]
output = args.output or f"viz/divergences_{p_model_name}_{q_model_name}.csv"
with open(output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(divergences)
logger.info(f"Wrote {len(divergences)} rows to {output}")
