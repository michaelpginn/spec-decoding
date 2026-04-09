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
from src.data.create_inputs import create_inputs, create_prompt
from src.data.dataset import assemble_dataset
from src.generation import generate_output
from src.n_gram import NGramModel
from src.spec_dec_metrics import log_token_flow, summarize_metrics
from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run(config: ExperimentConfig):
    """Run experiment: load config, init wandb, dispatch to task (e.g. translation)."""
    if config.task == "translation":
        from src.tasks.translation import compute_eval_metrics, load_data
    else:
        raise NotImplementedError()

    # 1. Load data
    logger.info(f"Loading data for {config.language_code}...")
    data, language = load_data(config)
    logger.info(f"Loaded {len(data)} examples")
    assert data and len(data) > 0

    # 2. Load target model
    logger.info(f"Loading target model: {config.target_model}...")
    target_model, target_tokenizer = load_model(
        config.target_model, device=config.device
    )
    device = next(target_model.parameters()).device

    # 3. Load draft model
    if config.draft_model_type == "none":
        logger.info("Specified no draft model, running without spec dec")
        draft_model = None
        draft_tokenizer = None
    elif config.draft_model_type == "neural":
        if config.draft_model is None:
            raise ValueError(
                "draft_model must be set when draft_model_type='neural'"
            )
        logger.info(f"Loading draft model: {config.draft_model}...")
        if config.draft_model != config.target_model:
            draft_model, draft_tokenizer = load_model(
                config.draft_model, device=config.device
            )
        else:
            draft_model = target_model
            draft_tokenizer = target_tokenizer
    elif config.draft_model_type == "ngram":
        draft_tokenizer = target_tokenizer
        draft_model = NGramModel(n=config.ngram_n, tokenizer=draft_tokenizer)
        draft_model.train(assemble_dataset(language, 'mono')['train'])
    else:
        raise ValueError()

    # 4. Decoding loop
    predictions = []
    all_metrics: list[dict] = []
    for input, _ in tqdm(data, desc="Decoding"):
        prompt = create_prompt(config.task, language, input)
        inputs = create_inputs(prompt, target_tokenizer, device)
        predicted, metrics = generate_output(
            inputs,
            target_model,
            target_tokenizer,
            draft_model,
            draft_tokenizer,
            config,
        )
        predictions.append(predicted)
        all_metrics.append(metrics)

    # 5. Aggregate and log speculative decoding metrics
    per_sentence_metrics, summary_metrics = summarize_metrics(
        all_metrics,
        config.gamma,
        config.draft_model_type != "none" and not config.use_hf_assisted,
    )
    wandb.summary.update(summary_metrics)
    for entry in per_sentence_metrics:
        wandb.log(entry)
    # Remove per-sentence keys that wandb.log in summary
    for key in list(wandb.summary.keys()):
        if key.startswith("sentence/") or key == "sentence_idx":
            del wandb.summary[key]
    log_token_flow([inp for inp, _ in data], all_metrics, config)

    # 6. Log evaluation metrics
    eval_metrics = compute_eval_metrics([ref for _, ref in data], predictions)
    wandb.summary.update(eval_metrics)


def setup_wandb(config: ExperimentConfig):
    target_short = config.target_model.split("/")[-1]
    is_spec = config.draft_model_type != "none"
    if config.draft_model_type == 'ngram':
        draft_short = "ngram"
    elif config.draft_model_type == 'neural':
        if config.draft_model:
            draft_short = config.draft_model.split("/")[-1] # type:ignore
        else:
            draft_short = target_short
    else:
        draft_short = None
        
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
    tags = [t for t in tags if t is not None]

    wandb_config = asdict(config)
    wandb_config["target_model_short"] = target_short
    wandb_config["draft_model_short"] = draft_short
    wandb_config["model_pair"] = (
        f"{target_short}+{draft_short}" if is_spec else target_short
    )
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

    metrics_md = Path(__file__).parent / "src" / "metrics.md"
    wandb.run.notes = metrics_md.read_text(encoding="utf-8")  # type:ignore


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

    setup_wandb(config)
    try:
        run(config)
    finally:
        wandb.finish()
