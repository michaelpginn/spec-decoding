"""
Entry point for knowledge distillation.
"""
import argparse
import logging
import pprint

from src.config.config import DistillConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.tasks.distillation.train import run_distillation

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument(
        "config", help="A config file (cfg, ini) with distillation parameters"
    )
    parser.add_argument(
        "--overrides", "-o",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
        nargs="+",
    )
    args = parser.parse_args()

    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=DistillConfig,
    )
    logger.info(f"Distillation config:\n{pprint.pformat(config)}")
    run_distillation(config)
