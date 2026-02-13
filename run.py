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
    # TODO: Main script for running spec decoding eval
    # Should take command-line args such as:
    # - source (ntrex | totoeba | other)
    # - language code
    # - spec decoding setting (greedy, eagle, etc)
    # - draft model setting (n-gram, distill, etc)
    # - draft model name (if already trained)
    # Then do the following:
    # 1. Load the appropriate evaluation dataset
    # 2. If no draft model is provided, train the draft model
    # 3. Run according to the setting and log metrics to wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "spec-decoding"),
        entity=os.environ.get("WANDB_ENTITY", "lecs-general"),
        config=asdict(config),
    )
    if config.task == "translation":
        run_translation(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")


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
