"""
Translation with speculative decoding experiments.
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import time
from pathlib import Path
from tqdm import tqdm
from translation.data_loader import load_tatoeba_data, get_language_name
from translation.models import load_target_model, translate_target
from translation.spec_decode import assisted_decode_hf, speculative_decode_translate
from translation.evaluate import compute_bleu, compute_spec_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Translation with speculative decoding experiments"
    )

    parser.add_argument("--target-lang", type=str, required=True, 
                       help="Target language code (ber, chr, haw, npi, etc.)")
    parser.add_argument("--data-source", type=str, default="tatoeba",
                       choices=["tatoeba", "flores"],
                       help="Data source for evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max number of examples to load (default: all)")

    parser.add_argument("--target-model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Target model (HuggingFace model name)")
    parser.add_argument("--draft-model", type=str, default=None,
                       help="Draft model (HuggingFace model name)")
    parser.add_argument("--draft-model-path", type=str, default=None,
                       help="Local path to draft model")

    parser.add_argument("--max-length", type=int, default=256,
                       help="Max new tokens")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cuda", "cpu", "auto"],
                       help="Device to run on")
    parser.add_argument("--output-dir", type=str, default="./outputs/translation",
                       help="Output directory for results")

    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline translation")
    
    # Speculative decoding options (custom implementation is default)
    parser.add_argument("--gamma", type=int, default=5,
                       help="Number of draft tokens per iteration (default: 5)")
    
    # Use HuggingFace's assisted generation instead of custom implementation
    parser.add_argument("--use-hf-assisted", action="store_true",
                       help="Use HuggingFace's assisted generation instead of custom implementation")
    parser.add_argument("--num-assistant-tokens", type=int, default=5,
                       help="[HF only] Number of tokens draft model generates before verification (default: 5)")
    parser.add_argument("--assistant-tokens-schedule", type=str, default="heuristic",
                       choices=["heuristic", "constant"],
                       help="[HF only] Token schedule: 'heuristic' (dynamic) or 'constant' (fixed)")

    args = parser.parse_args()

    print(f"Loading data for {args.target_lang}...")
    pairs = load_tatoeba_data(args.target_lang, max_samples=args.max_samples)
    print(f"Loaded {len(pairs)} source-target pairs")

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    lang_name = get_language_name(args.target_lang)

    print(f"\nLoading target model: {args.target_model}...")
    target_model, target_tokenizer = load_target_model(args.target_model, device=args.device)
    device = next(target_model.parameters()).device
    print(f"Target model loaded on {device}")

    baseline_times = []
    baseline_translations = []

    if not args.skip_baseline:
        print("\nRunning Baseline")
        for i, source in enumerate(tqdm(sources, desc="Baseline")):
            start = time.time()
            translation = translate_target(
                target_model, target_tokenizer, source, lang_name,
                max_length=args.max_length, device=device,
                debug=(i == 0)  # Print prompt only for first sample
            )
            baseline_times.append(time.time() - start)
            baseline_translations.append(translation)

        baseline_bleu = compute_bleu(references, baseline_translations, verbose=True)
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        print(f"Baseline avg time: {avg_baseline_time:.3f}s")
    else:
        print("\nSkipping Baseline")
        baseline_bleu = {"bleu": 0, "chrf2": 0}
        avg_baseline_time = 0

    print("\nLoading Draft Model")
    if args.draft_model_path:
        print(f"Loading from: {args.draft_model_path}")
        draft_model, draft_tokenizer = load_target_model(args.draft_model_path, device=args.device)
    elif args.draft_model:
        print(f"Loading from HuggingFace: {args.draft_model}")
        draft_model, draft_tokenizer = load_target_model(args.draft_model, device=args.device)
    else:
        print("No draft model specified. Using target model as draft.")
        draft_model = target_model
        draft_tokenizer = target_tokenizer

    print(f"Draft model ready on {device}")

    spec_translations = []
    spec_results = []  # Full metrics from each sample
    spec_metrics = None  # Will be set for custom implementation

    if args.use_hf_assisted:
        # HF ASSISTED GENERATION
        print("\nRunning Speculative Decoding (HuggingFace assisted generation)")
        print(f"Using {args.num_assistant_tokens} draft tokens with '{args.assistant_tokens_schedule}' schedule")
        
        for source in tqdm(sources, desc="HF assisted"):
            translation, metrics = assisted_decode_hf(
                target_model, target_tokenizer,
                draft_model, draft_tokenizer,
                source, lang_name,
                max_length=args.max_length,
                device=device,
                return_metrics=True,
                num_assistant_tokens=args.num_assistant_tokens,
                num_assistant_tokens_schedule=args.assistant_tokens_schedule,
            )
            spec_translations.append(translation)
            spec_results.append(metrics)
    else:
        #  CUSTOM SPECULATIVE DECODING (DEFAULT)
        # Requires same tokenizer for target and draft models
        if target_tokenizer is not draft_tokenizer:
            print("\nChecking tokenizer compatibility...")
        
        print(f"\nRunning Custom Speculative Decoding (greedy, gamma={args.gamma})")
        
        for i, source in enumerate(tqdm(sources, desc="Spec decode")):
            translation, metrics = speculative_decode_translate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tokenizer,  # Shared tokenizer
                source=source,
                target_lang=lang_name,
                max_length=args.max_length,
                gamma=args.gamma,
                device=device,
                debug=(i == 0),  # Print first prompt only
            )
            spec_translations.append(translation)
            spec_results.append(metrics)
        
        # Compute and print spec decoding metrics
        spec_metrics = compute_spec_metrics(spec_results, gamma=args.gamma, verbose=True)

    # Translation quality metrics
    print("\n=== Translation Quality ===")
    spec_bleu = compute_bleu(references, spec_translations, verbose=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"translations_{args.target_lang}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        if baseline_translations:
            f.write("=" * 80 + "\n")
            f.write("BASELINE (target model only, greedy)\n")
            f.write("=" * 80 + "\n\n")
            for src, ref, trans in zip(sources, references, baseline_translations):
                f.write(f"SOURCE: {src}\n")
                f.write(f"REFERENCE: {ref}\n")
                f.write(f"TRANSLATION: {trans}\n")
                f.write("-" * 80 + "\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("SPECULATIVE DECODING (target + draft)\n")
        f.write("=" * 80 + "\n\n")
        for src, ref, trans in zip(sources, references, spec_translations):
            f.write(f"SOURCE: {src}\n")
            f.write(f"REFERENCE: {ref}\n")
            f.write(f"TRANSLATION: {trans}\n")
            f.write("-" * 80 + "\n")

    metrics_file = output_dir / f"metrics_{args.target_lang}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        if not args.skip_baseline:
            f.write("=== Baseline ===\n")
            f.write(f"BLEU: {baseline_bleu['bleu']:.2f}\n")
            f.write(f"chrF2: {baseline_bleu['chrf2']:.2f}\n\n")
        
        f.write("=== Speculative Decoding ===\n")
        if args.use_hf_assisted:
            f.write("Implementation: HuggingFace assisted generation\n")
            f.write(f"BLEU: {spec_bleu['bleu']:.2f}\n")
            f.write(f"chrF2: {spec_bleu['chrf2']:.2f}\n")
        else:
            f.write(f"Implementation: Custom (greedy, gamma={args.gamma})\n")
            f.write(f"BLEU: {spec_bleu['bleu']:.2f}\n")
            f.write(f"chrF2: {spec_bleu['chrf2']:.2f}\n")
            # Write spec decode metrics
            f.write(f"Acceptance Rate: {spec_metrics['acceptance_rate']:.2%}\n")
            f.write(f"Mean Accepted Tokens: {spec_metrics['mean_accepted_tokens']:.2f}\n")
            f.write(f"Block Efficiency: {spec_metrics['block_efficiency']:.2%}\n")

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
