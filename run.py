import argparse
import logging
import pprint
import os
from dataclasses import asdict

import wandb

from src.config.config import ExperimentConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.tasks.translation.run import run_translation

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


def run(config: ExperimentConfig):
    """Run experiment: load config, init wandb, dispatch to task (e.g. translation)."""
    target_short = config.target_model.split("/")[-1]
    is_spec = config.draft_model_type != "none"
    draft_short = (
        config.draft_model.split("/")[-1]
        if config.draft_model and config.draft_model != "None"
        else None
    )
    job_type = "spec" if is_spec else "baseline"

    group = f"{target_short}__{config.language_code}"

    if is_spec:
        name = f"{config.language_code}_{draft_short}_g{config.gamma}"
    else:
        name = f"{config.language_code}_baseline"

    tags = [config.language_code, target_short, config.decoding_mode, config.task]
    if is_spec:
        tags += [draft_short, f"gamma={config.gamma}", config.draft_model_type]
    else:
        tags.append("baseline")

    wandb_config = asdict(config)
    wandb_config["target_model_short"] = target_short
    wandb_config["draft_model_short"] = draft_short
    wandb_config["model_pair"] = f"{target_short}+{draft_short}" if is_spec else target_short
    wandb_config["run_type"] = job_type

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "spec-decoding"),
        entity=os.environ.get("WANDB_ENTITY", "lecs-general"),
        config=wandb_config,
        group=group,
        job_type=job_type,
        name=name,
        tags=tags,
    )

    wandb.define_metric("sentence_idx")
    wandb.define_metric("sentence/*", step_metric="sentence_idx", summary="mean")

    wandb.run.notes = (
        "## Summary Metrics\n"
        "- **total_time**: Total decode time across all sentences, excluding prefill (seconds)\n"
        "- **avg_time_per_sentence**: Average decode time per sentence, excluding prefill (seconds)\n"
        "- **median_time_per_sentence**: Median decode time per sentence — less sensitive to outliers (seconds)\n"
        "- **avg_time_per_token**: Average time to generate a single token, averaged across sentences (seconds)\n"
        "- **tokens_per_second**: Decoding throughput — total tokens generated / total decode time\n"
        "- **bleu**: Corpus-level BLEU score measuring n-gram overlap with reference translations (0–100)\n"
        "- **chrf2**: Character-level F-score (chrF2) — more robust than BLEU for morphologically rich languages (0–100)\n"
        "\n"
        "### Speculative Decoding Only\n"
        "- **total_generated_tokens**: Total number of output tokens produced across all sentences\n"
        "- **total_draft_tokens**: Total number of draft tokens proposed across all sentences\n"
        "- **total_matched_tokens**: Total number of draft tokens accepted by the target model\n"
        "- **draft_to_output_ratio**: total_draft_tokens / total_generated_tokens — how many draft tokens were needed per output token\n"
        "- **acceptance_rate**: Fraction of all proposed draft tokens that matched the target model (weighted by token count)\n"
        "- **mean_acceptance_rate**: Simple average of per-sentence acceptance rates (each sentence weighted equally)\n"
        "- **mean_accepted_tokens**: Average number of tokens accepted per speculative iteration (out of gamma proposed)\n"
        "- **block_efficiency**: mean_accepted_tokens / gamma — how much of each draft block was useful (0.0–1.0)\n"
        "\n"
        "## Timing Notes\n"
        "All time metrics measure **decode time only** (new token generation), excluding prompt prefill.\n"
        "- **Baseline**: Prefill is measured via a separate forward pass and subtracted from total generate() time\n"
        "- **Spec decode**: Timer starts after prefill + first token, measuring only the speculative loop\n"
        "\n"
        "## Per-Sentence Charts (sentence/*)\n"
        "- **sentence/time**: Decode time for this sentence, excluding prefill (seconds)\n"
        "- **sentence/time_per_token**: Time per generated token for this sentence (seconds)\n"
        "- **sentence/generated_tokens**: Number of new tokens the model produced for this sentence\n"
        "- **sentence/draft_tokens**: (spec only) Total draft tokens proposed by the draft model\n"
        "- **sentence/matched_tokens**: (spec only) How many draft tokens the target model accepted\n"
        "- **sentence/acceptance_rate**: (spec only) Fraction of draft tokens accepted for this sentence\n"
    )

    try:
        if config.task == "translation":
            run_translation(config)
        else:
            raise ValueError(f"Unknown task: {config.task}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "--overrides",
        "-o",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
        nargs="+",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=ExperimentConfig,
    )
    logger.info(f"Experiment config:\n{pprint.pformat(config)}")
    run(config)
