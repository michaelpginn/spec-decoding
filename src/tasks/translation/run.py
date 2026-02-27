# src/tasks/translation/run.py

import logging
from pathlib import Path
from tqdm import tqdm
import wandb

from src.config.config import ExperimentConfig
from src.decoding.models import load_model
from src.evaluation import compute_baseline_metrics, compute_spec_metrics
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
    if not pairs:
        raise ValueError("No source-target pairs loaded; cannot run metrics.")

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    lang_name = get_language_name(config.language_code)

    # 2. Load target model
    logger.info(f"Loading target model: {config.target_model}...")
    target_model, target_tokenizer = load_model(config.target_model, device=config.device)
    device = next(target_model.parameters()).device

    # 3. Baseline run
    if config.draft_model_type == "none":
        baseline_times = []
        baseline_token_counts = []
        baseline_translations = []
        logger.info("Running baseline (no draft model)...")
        for i, source in enumerate(tqdm(sources, desc="Baseline")):
            translation, num_tokens, decode_time = translate_target(
                target_model, target_tokenizer, source, lang_name,
                max_new_tokens=config.max_new_tokens, device=device,
            )
            baseline_times.append(decode_time)
            baseline_token_counts.append(num_tokens)
            baseline_translations.append(translation)

        baseline_bleu = compute_bleu(references, baseline_translations, verbose=False)
        per_sentence, summary = compute_baseline_metrics(baseline_times, baseline_token_counts)

        for entry in per_sentence:
            wandb.log(entry)

        summary["bleu"] = baseline_bleu["bleu"]
        summary["chrf2"] = baseline_bleu["chrf2"]
        wandb.summary.update(summary)

        # Remove per-sentence keys that wandb.log in summary
        for key in list(wandb.summary.keys()):
            if key.startswith("sentence/") or key == "sentence_idx":
                del wandb.summary[key]

        logger.info(
            f"Baseline BLEU: {baseline_bleu['bleu']:.2f}  chrF2: {baseline_bleu['chrf2']:.2f}  "
            f"Avg: {summary['avg_time_per_sentence']:.2f}s/sentence  "
            f"Avg time/token: {summary['avg_time_per_token']:.4f}s  "
            f"Tokens/sec: {summary['tokens_per_second']:.2f}"
        )
        return

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
        hf_times = []
        hf_token_counts = []
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
            hf_times.append(metrics["time"])
            hf_token_counts.append(metrics["generated_tokens"])

        # HF assisted is a black box — no acceptance rate data available,
        # so we use baseline-style metrics (time + token counts only).
        per_sentence, summary = compute_baseline_metrics(hf_times, hf_token_counts)
        summary["method"] = "hf_assisted"

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

        per_sentence, summary = compute_spec_metrics(
            spec_results, gamma=config.gamma, verbose=False
        )

    # 6. Log per-sentence metrics
    for entry in per_sentence:
        wandb.log(entry)

    spec_bleu = compute_bleu(references, spec_translations, verbose=False)
    summary["bleu"] = spec_bleu["bleu"]
    summary["chrf2"] = spec_bleu["chrf2"]
    wandb.summary.update(summary)

    # Remove per-sentence keys that wandb.log leaked into the summary
    for key in list(wandb.summary.keys()):
        if key.startswith("sentence/") or key == "sentence_idx":
            del wandb.summary[key]

    logger.info(
        f"Spec BLEU: {spec_bleu['bleu']:.2f}  chrF2: {spec_bleu['chrf2']:.2f}  "
        f"Avg: {summary['avg_time_per_sentence']:.2f}s/sentence  "
        f"Avg time/token: {summary['avg_time_per_token']:.4f}s  "
        f"Tokens/sec: {summary['tokens_per_second']:.2f}"
    )

    # 7. Log token flow trace as wandb Table
    if any("iteration_history" in r for r in spec_results):
        flow_table = wandb.Table(columns=[
            "sample_idx", "source_text", "iteration", "drafted_tokens",
            "num_drafted", "result",
        ])
        for i, result in enumerate(spec_results):
            for item in result.get("iteration_history", []):
                flow_table.add_data(
                    i, sources[i], item["iter"],
                    " ".join(f"[{t}]" for t in item["drafted"]),
                    len(item["drafted"]), item["result"],
                )
        wandb.log({"token_flow": flow_table})

    # 8. Save token flow trace locally
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
