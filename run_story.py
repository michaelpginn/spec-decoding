"""
Story generation experiment runner.

Mirrors run.py but sources prompts from story_gen_prompt instead of
bilingual translation data. Supports speculative decoding, repetition penalty,
and all standard config overrides.

Usage:
    uv run python run_story.py experiments/spec_greedy.cfg
    uv run python run_story.py experiments/spec_greedy.cfg -o language_code=npi num_prompts=20
    uv run python run_story.py experiments/spec_greedy.cfg -o repetition_penalty=1.1 repetition_penalty_window=16
    WANDB_MODE=disabled uv run python run_story.py experiments/spec_greedy.cfg -o num_prompts=5
"""

import argparse
import logging
import os
import pprint
from dataclasses import asdict
from pathlib import Path

import wandb
from tqdm import tqdm

from src.config.config import ExperimentConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.data.dataset import get_language_name
from src.generation import generate_output
from src.n_gram import NGramModel
from src.spec_dec_metrics import log_token_flow, summarize_metrics
from src.story_gen_prompt import create_inputs_story, create_prompt
from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_story(config: ExperimentConfig, num_prompts: int = 10, adj_n: bool = False):
    """Load models, generate story prompts, run decoding, log metrics."""
    language = get_language_name(config.language_code)
    logger.info(f"Story generation | language={language} | num_prompts={num_prompts}")

    # 1. Load target model
    logger.info(f"Loading target model: {config.target_model}...")
    target_model, target_tokenizer = load_model(config.target_model, device=config.device)
    device = next(target_model.parameters()).device

    # 2. Load draft model
    if config.draft_model_type == "none":
        logger.info("No draft model — running standard (non-speculative) decoding.")
        draft_model = None
        draft_tokenizer = None
    elif config.draft_model_type == "neural":
        if config.draft_model is None:
            raise ValueError("draft_model must be set when draft_model_type='neural'")
        logger.info(f"Loading draft model: {config.draft_model}...")
        if config.draft_model != config.target_model:
            draft_model, draft_tokenizer = load_model(config.draft_model, device=config.device)
        else:
            draft_model = target_model
            draft_tokenizer = target_tokenizer
    elif config.draft_model_type == "ngram":
        raise ValueError(
            "ngram draft model requires a monolingual training corpus — not supported for story gen. "
            "Use neural or none."
        )
    else:
        raise ValueError(f"Unknown draft_model_type: {config.draft_model_type!r}")

    # 3. Build story prompts
    prompts: dict = create_prompt(language=language, adj_n=adj_n, num_prompts=num_prompts)
    logger.info(f"Generated {len(prompts)} story prompts for {language}.")
    if config.repetition_penalty > 1.0:
        logger.info(
            f"Repetition penalty ENABLED: penalty={config.repetition_penalty}, "
            f"window={config.repetition_penalty_window}"
        )
    else:
        logger.info("Repetition penalty DISABLED (penalty=1.0).")

    # 4. Decoding loop
    all_metrics: list[dict] = []
    for idx, prompt_text in tqdm(prompts.items(), desc="Generating stories"):
        inputs = create_inputs_story(prompt_text, target_tokenizer, device=device)
        story, metrics = generate_output(
            inputs,
            target_model,
            target_tokenizer,
            draft_model,
            draft_tokenizer,
            config,
        )
        all_metrics.append(metrics)
        logger.info(f"\n{'─' * 60}\nPrompt [{idx}]: {prompt_text}\n\nStory:\n{story}\n{'─' * 60}")

    # 5. Aggregate and log speculative decoding metrics
    is_spec = config.draft_model_type != "none" and not config.use_hf_assisted
    per_sentence_metrics, summary_metrics = summarize_metrics(all_metrics, config.gamma, is_spec)
    wandb.summary.update(summary_metrics)
    for entry in per_sentence_metrics:
        wandb.log(entry)
    for key in list(wandb.summary.keys()):
        if key.startswith("sentence/") or key == "sentence_idx":
            del wandb.summary[key]
    log_token_flow(list(prompts.values()), all_metrics, config)

    # Story gen has no reference translations — skip BLEU/chrF2.
    logger.info(f"Done. Summary metrics:\n{pprint.pformat(summary_metrics)}")


def setup_wandb(config: ExperimentConfig, num_prompts: int):
    target_short = config.target_model.split("/")[-1]
    is_spec = config.draft_model_type != "none"
    draft_short = (
        config.draft_model.split("/")[-1]
        if (config.draft_model and config.draft_model_type == "neural")
        else config.draft_model_type
    )
    job_type = "story-spec" if is_spec else "story-baseline"
    group = f"{target_short}__{config.language_code}__story"
    name = (
        f"{config.language_code}_story_{draft_short}_g{config.gamma}"
        if is_spec
        else f"{config.language_code}_story_baseline"
    )
    tags = [config.language_code, target_short, config.decoding_mode, "story"]
    if is_spec:
        tags += [draft_short, f"gamma={config.gamma}"]
    if config.wandb_tag:
        tags.append(config.wandb_tag)

    wandb_config = asdict(config)
    wandb_config["num_prompts"] = num_prompts
    wandb_config["task"] = "story_generation"
    wandb_config["target_model_short"] = target_short
    wandb_config["draft_model_short"] = draft_short

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

    metrics_md = Path(__file__).parent / "src" / "metrics.md"
    wandb.run.notes = metrics_md.read_text(encoding="utf-8")  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run story generation experiment")
    parser.add_argument("config", help="Config file (cfg/ini)")
    parser.add_argument(
        "--overrides", "-o",
        help="Override config values: key1=value1 key2=value2",
        nargs="+",
    )
    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=10,
        help="Number of story prompts to generate (default: 10)",
    )
    parser.add_argument(
        "--adj-n",
        action="store_true",
        help="Use adjective+noun story seeds instead of noun-only",
    )
    args = parser.parse_args()

    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=ExperimentConfig,
    )
    logger.info(f"Story experiment config:\n{pprint.pformat(config)}")

    setup_wandb(config, num_prompts=args.num_prompts)
    try:
        run_story(config, num_prompts=args.num_prompts, adj_n=args.adj_n)
    finally:
        wandb.finish()
