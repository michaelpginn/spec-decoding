"""Measures tokenizer fertility (tokens per character) for each language.

Fertility is an alternative resourcedness measure to raw pretraining token
counts: it captures how well a model's tokenizer actually covers a language.
A language written in a script absent from the vocabulary falls back to raw
UTF-8 bytes and scores above 1.0, while a well-covered Latin-script language
sits near 0.4.

Unlike the FineWeb counts it is defined for every language (no -1 holes), but
it is tokenizer-specific, so it is computed once per model family over the same
monolingual split that src/measure_divergence.py scores.

    uv run -m src.compute_fertility
"""

import argparse
import csv
import logging

from transformers import AutoTokenizer

from src.data.dataset import assemble_dataset, get_language_name, LANGUAGES

logger = logging.getLogger(__name__)

MAX_MONO = 20000
MAX_EXAMPLES = 2000  # per language; the estimate is stable well before this

# The tokenizer each family's models share, matching the pairs measured by
# src/measure_divergence.py.
FAMILY_TOKENIZERS = {
    "qwen": "Qwen/Qwen3.5-9B",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="viz/fertility.csv")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    rows = []
    for family, tokenizer_name in FAMILY_TOKENIZERS.items():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"[{family}] tokenizer {tokenizer_name} (vocab={len(tokenizer)})")
        for lang in args.languages:
            try:
                # Same split measure_divergence.py scores, assembled with the
                # same tokenizer, so the two files describe identical text.
                dataset = assemble_dataset(lang, "mono", tokenizer, MAX_MONO)["test"]
            except Exception as e:
                logger.error(f"[{family}] {lang}: failed to load ({type(e).__name__}: {e})")
                continue
            texts = dataset["text"][:MAX_EXAMPLES]
            if not texts:
                logger.warning(f"[{family}] {lang}: no examples, skipping")
                continue
            n_tokens = sum(len(tokenizer.tokenize(t)) for t in texts)
            n_chars = sum(len(t) for t in texts)
            if n_chars == 0:
                logger.warning(f"[{family}] {lang}: no characters, skipping")
                continue
            row = {
                "language_code": lang,
                "language": get_language_name(lang),
                "family": family,
                "fertility": n_tokens / n_chars,
                "mean_tokens": n_tokens / len(texts),
                "mean_chars": n_chars / len(texts),
                "n_examples": len(texts),
            }
            rows.append(row)
            logger.info(
                f"[{family}] {lang}: fertility={row['fertility']:.4f} "
                f"mean_tokens={row['mean_tokens']:.1f} n={len(texts)}"
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language_code", "language", "family", "fertility",
                        "mean_tokens", "mean_chars", "n_examples"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
