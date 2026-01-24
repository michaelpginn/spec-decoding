"""
experimenting translation tasks using speculative decoding
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from translation.data_loader import load_tatoeba_data
from translation.models import load_target_model, translate_target
# from translation.spec_decode import speculative_decode
from translation.evaluate import compute_bleu

def main():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--target-lang", type=str, required=True, 
                       help="Target language code (ber, chr, haw...)")
    parser.add_argument("--data-source", type=str, default="tatoeba",
                       choices=["tatoeba", "flores..."])

    parser.add_argument("--max-samples", type=int, default=None,
                    help="Max number of examples to load (default: all)")

    # model args
    parser.add_argument("--target-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="base translation model")   
    parser.add_argument("--draft-model-type", type=str,
                       choices=["ngram", "distill", "neural"],
                       default="ngram")
    parser.add_argument("--draft-model-path", type=str, default=None,
                       help="path to  draft model")

    # spec decoding args
    parser.add_argument("--spec-method", type=str, default="greedy",
                        choices=["greedy", "eagle", "medusa"])  # although mentioned, we are only using greedy approach now
    parser.add_argument("--draft-k", type=int, default=4,
                       help="Number of draft tokens")

    # training args
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cuda", "cpu", "auto"])
    parser.add_argument("--output-dir", type=str, default="./outputs/translation")

    parser.add_argument("--wandb_project", type=str, default=None)

    args = parser.parse_args()

    # load data
    print(f"Loading data for {args.target_lang}...")
    pairs = load_tatoeba_data(args.target_lang, max_samples=args.max_samples)
    print(f"Loaded {len(pairs)} source-target pairs")

    # load model
    print(f"Loading target model: {args.target_model}...")
    model, tokenizer = load_target_model(args.target_model, device=args.device)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # translate sentences
    translations = []
    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]

    lang_name = args.target_lang.upper()  # or map to full name
    
    for source in tqdm(sources, desc="translating..."):
        translation = translate_target(
            model, tokenizer, source, lang_name,
            max_length=args.max_length, device=device
        )
        translations.append(translation)
    
    # save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"translations_{args.target_lang}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for src, ref, trans in zip(sources, references, translations):
            f.write(f"SOURCE: {src}\n")
            f.write(f"REFERENCE: {ref}\n")
            f.write(f"TRANSLATION: {trans}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Saved translations to {output_file}")
    print(f"Translated {len(translations)} sentences")

    metrics = compute_bleu(references, translations, verbose=True)
    metrics_file = output_dir / f"metrics_{args.target_lang}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"BLEU: {metrics['bleu']:.2f}\n")
        f.write(f"chrF2: {metrics['chrf2']:.2f}\n")

if __name__ == "__main__":
    main()