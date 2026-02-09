# src/tasks/translation/run.py

import logging
import time
from pathlib import Path
from tqdm import tqdm
import wandb

from src.config.config import ExperimentConfig
from src.decoding.models import load_model
from src.evaluation import compute_spec_metrics
from src.tasks.translation.data_loader import load_tatoeba_data, get_language_name
from src.tasks.translation.translate import (
    translate_target,
    speculative_decode_translate,
    assisted_decode_hf,
)
from src.tasks.translation.evaluate import compute_bleu

logger = logging.getLogger(__name__)

def run_translation(config: ExperimentConfig):
    """Run translation experiment driven by ExperimentConfig."""

    # 1. Load data
    logger.info(f"Loading data for {config.language_code}...")
    pairs = load_tatoeba_data(config.language_code, max_samples=config.max_samples)
    logger.info(f"Loaded {len(pairs)} source-target pairs")

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    lang_name = get_language_name(config.language_code)

    # 2. Load target model
    logger.info(f"Loading target model: {config.target_model}...")
    target_model, target_tokenizer = load_model(config.target_model, device=config.device)
    device = next(target_model.parameters()).device

    # 3. Baseline (always run for comparison)
    baseline_times = []
    baseline_translations = []
    logger.info("Running baseline...")
    for i, source in enumerate(tqdm(sources, desc="Baseline")):
        start = time.time()
        translation = translate_target(
            target_model, target_tokenizer, source, lang_name,
            max_new_tokens=config.max_new_tokens, device=device,
        )
        baseline_times.append(time.time() - start)
        baseline_translations.append(translation)

    baseline_bleu = compute_bleu(references, baseline_translations, verbose=False)
    wandb.log({"baseline/bleu": baseline_bleu["bleu"], "baseline/chrf2": baseline_bleu["chrf2"]})
    logger.info(f"Baseline BLEU: {baseline_bleu['bleu']:.2f}  chrF2: {baseline_bleu['chrf2']:.2f}")

    # 4. Load draft model
    if config.draft_model:
        logger.info(f"Loading draft model: {config.draft_model}...")
        draft_model, draft_tokenizer = load_model(config.draft_model, device=config.device)
    else:
        logger.info("No draft model specified, using target as draft.")
        draft_model = target_model
        draft_tokenizer = target_tokenizer

    # 5. Speculative decoding
    spec_translations = []
    spec_results = []

    if config.use_hf_assisted:
        logger.info(f"Running HF assisted generation (gamma={config.gamma})...")
        for source in tqdm(sources, desc="HF assisted"):
            translation, metrics = assisted_decode_hf(
                target_model, target_tokenizer,
                draft_model, draft_tokenizer,
                source, lang_name,
                max_new_tokens=config.max_new_tokens,
                device=device,
                num_assistant_tokens=config.gamma,
            )
            spec_translations.append(translation)
            spec_results.append(metrics)
    else:
        logger.info(f"Running custom spec decode (greedy, gamma={config.gamma})...")
        for i, source in enumerate(tqdm(sources, desc="Spec decode")):
            translation, metrics = speculative_decode_translate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tokenizer,
                source=source,
                target_lang=lang_name,
                max_new_tokens=config.max_new_tokens,
                gamma=config.gamma,
                device=device,
                track_iterations=True,
            )
            spec_translations.append(translation)
            spec_results.append(metrics)

        spec_metrics = compute_spec_metrics(
            spec_results, gamma=config.gamma,
            baseline_times=baseline_times, verbose=False
        )
        wandb.log(spec_metrics)

    # 6. Translation quality
    spec_bleu = compute_bleu(references, spec_translations, verbose=False)
    wandb.log({"spec/bleu": spec_bleu["bleu"], "spec/chrf2": spec_bleu["chrf2"]})
    logger.info(f"Spec BLEU: {spec_bleu['bleu']:.2f}  chrF2: {spec_bleu['chrf2']:.2f}")

    # 7. Save token flow trace
    output_dir = Path(f"./outputs/{config.language_code}")
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_file = output_dir / f"token_flow_{config.language_code}.txt"
    with open(iteration_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(spec_results):
            if "iteration_history" in result:
                f.write(f"\n{'='*60}\n")
                f.write(f"Sample {i+1}: \"{sources[i]}\"\n")
                f.write(f"{'='*60}\n")
                for item in result["iteration_history"]:
                    drafted_str = " ".join(f"[{t}]" for t in item["drafted"])
                    f.write(f"  Iter {item['iter']}: {drafted_str}\n")
                    f.write(f"           -> {item['result']}\n")
    logger.info(f"Token flow trace saved to {iteration_file}")