import argparse
import logging
import pprint
import os
from pathlib import Path
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

    metrics_md = Path(__file__).parent / "src" / "metrics.md"
    wandb.run.notes = metrics_md.read_text(encoding="utf-8")

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
