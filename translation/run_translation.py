"""
experimenting translation tasks using speculative decoding
"""

import argparse
from translation.data_loader import load_tatoeba_data
# from translation.models import load_target_model, load_draft_model
# from translation.spec_decode import speculative_decode
# from translation.evaluate import compute_bleu

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

# code here

if __name__ == "__main__":
    main()