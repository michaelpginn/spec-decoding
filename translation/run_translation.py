"""
Experimenting translation tasks using speculative decoding.
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import time
from pathlib import Path
from tqdm import tqdm
from translation.data_loader import load_tatoeba_data
from translation.models import load_target_model, translate_target
from translation.spec_decode import speculative_decode
from translation.evaluate import compute_bleu, compute_spec_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Translation with speculative decoding experiments"
    )

    # Data args
    parser.add_argument("--target-lang", type=str, required=True, 
                       help="Target language code (ber, chr, haw, npi, etc.)")
    parser.add_argument("--data-source", type=str, default="tatoeba",
                       choices=["tatoeba", "flores"],
                       help="Data source for evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max number of examples to load (default: all)")

    # Model args
    parser.add_argument("--target-model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Target model (HuggingFace model name)")
    parser.add_argument("--draft-model", type=str, default=None,
                       help="Draft model (HuggingFace model name, e.g., Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--draft-model-type", type=str,
                       choices=["ngram", "distill", "neural"],
                       default="neural",
                       help="Type of draft model")
    parser.add_argument("--draft-model-path", type=str, default=None,
                       help="Local path to draft model (overrides --draft-model)")

    # Spec decoding args
    parser.add_argument("--spec-method", type=str, default="greedy",
                       choices=["greedy", "eagle", "medusa"],
                       help="Speculative decoding method")
    parser.add_argument("--draft-k", type=int, default=4,
                       help="Number of draft tokens per step")

    # Runtime args
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (currently unused)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Max sequence length")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cuda", "cpu", "auto"],
                       help="Device to run on")
    parser.add_argument("--output-dir", type=str, default="./outputs/translation",
                       help="Output directory for results")

    # Optional
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="Wandb project name (not implemented yet)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline translation (faster, but no speedup metric)")

    args = parser.parse_args()


    print(f"Loading data for {args.target_lang}...")
    pairs = load_tatoeba_data(args.target_lang, max_samples=args.max_samples)
    print(f"Loaded {len(pairs)} source-target pairs")

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    lang_name = args.target_lang.upper()

    print(f"\nLoading target model: {args.target_model}...")
    target_model, target_tokenizer = load_target_model(args.target_model, device=args.device)
    device = next(target_model.parameters()).device
    print(f"Target model loaded on {device}")

    # baseline translation
    baseline_times = []
    baseline_translations = []

    if not args.skip_baseline:
        print("\nRunning Baseline (for comparison)")
        for source in tqdm(sources, desc="Baseline translation"):
            start = time.time()
            translation = translate_target(
                target_model, target_tokenizer, source, lang_name,
                max_length=args.max_length, device=device
            )
            baseline_times.append(time.time() - start)
            baseline_translations.append(translation)

        baseline_bleu = compute_bleu(references, baseline_translations, verbose=True)
        print(f"Baseline avg time: {sum(baseline_times)/len(baseline_times):.3f}s")
    else:
        print("\nSkipping Baseline")

    print("\nLoading Draft Model")

    if args.draft_model_path:
        print(f"Loading draft model from local path: {args.draft_model_path}")
        draft_model, draft_tokenizer = load_target_model(args.draft_model_path, device=args.device)
    elif args.draft_model:
        print(f"Loading draft model from HuggingFace: {args.draft_model}")
        draft_model, draft_tokenizer = load_target_model(args.draft_model, device=args.device)
    else:
        print("No draft model specified. Using target model as draft (placeholder).")
        draft_model = target_model
        draft_tokenizer = target_tokenizer

    try:
        if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
            print("Warning: Draft and target tokenizers have different vocabularies!")
            print("Using target tokenizer for both.")
            draft_tokenizer = target_tokenizer
    except AttributeError:
        pass

    print(f"Draft model ready on {device}")

    print(f"\nRunning Speculative Decoding (method: {args.spec_method})")

    spec_translations = []
    spec_metrics_list = []

    for source in tqdm(sources, desc="Spec decoding"):
        translation, metrics = speculative_decode(
            target_model, target_tokenizer,
            draft_model, draft_tokenizer,
            source, lang_name,
            draft_k=args.draft_k,
            max_length=args.max_length,
            device=device,
            return_metrics=True
        )
        spec_translations.append(translation)
        spec_metrics_list.append(metrics)

    print("\nTranslation Quality")
    spec_bleu = compute_bleu(references, spec_translations, verbose=True)

    # Compute spec decoding metrics
    if baseline_times:
        spec_metrics = compute_spec_metrics(
            baseline_times, spec_metrics_list, draft_k=args.draft_k, verbose=True
        )
    else:
        print("\nComputing metrics without baseline comparison")
        spec_metrics = {}
        if spec_metrics_list:
            total_accepted = sum(r['total_accepted_tokens'] for r in spec_metrics_list)
            total_proposed = sum(r['total_draft_tokens'] for r in spec_metrics_list)
            all_accepted_per_step = []
            for r in spec_metrics_list:
                all_accepted_per_step.extend(r['accepted_tokens_per_step'])

            mat = sum(all_accepted_per_step) / len(all_accepted_per_step) if all_accepted_per_step else 0
            spec_metrics = {
                "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0,
                "mean_accepted_tokens": mat,
                "block_efficiency": mat / args.draft_k if args.draft_k > 0 else 0,
            }
            print(f"Acceptance Rate: {spec_metrics['acceptance_rate']:.3f}")
            print(f"Mean Accepted Tokens: {spec_metrics['mean_accepted_tokens']:.2f}")
            print(f"Block Efficiency: {spec_metrics['block_efficiency']:.2%}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save translations
    output_file = output_dir / f"translations_{args.target_lang}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for src, ref, trans in zip(sources, references, spec_translations):
            f.write(f"SOURCE: {src}\n")
            f.write(f"REFERENCE: {ref}\n")
            f.write(f"TRANSLATION: {trans}\n")
            f.write("-" * 80 + "\n")

    # Save all metrics
    metrics_file = output_dir / f"metrics_{args.target_lang}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("-------Translation Quality -- baseline--------\n")
        f.write(f"BLEU: {baseline_bleu['bleu']:.2f}\n")
        f.write(f"chrF2: {baseline_bleu['chrf2']:.2f}\n")
        f.write("-------Translation Quality -- speculative--------\n")
        f.write(f"BLEU: {spec_bleu['bleu']:.2f}\n")
        f.write(f"chrF2: {spec_bleu['chrf2']:.2f}\n")
        f.write("\nSpeculative Decoding Performance\n")
        if 'speedup' in spec_metrics:
            f.write(f"Speedup: {spec_metrics['speedup']:.2f}x\n")
            f.write(f"Decode-Only TPS: {spec_metrics['decode_tps']:.2f}\n")
        f.write(f"Acceptance Rate: {spec_metrics.get('acceptance_rate', 0):.3f}\n")
        f.write(f"Mean Accepted Tokens: {spec_metrics.get('mean_accepted_tokens', 0):.2f}\n")
        f.write(f"Block Efficiency: {spec_metrics.get('block_efficiency', 0):.2%}\n")
        if 'avg_kl_divergence' in spec_metrics:
            f.write(f"Avg KL Divergence: {spec_metrics['avg_kl_divergence']:.4f}\n")

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()