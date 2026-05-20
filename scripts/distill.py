"""
Entry point for knowledge distillation.
"""
import argparse
import logging
from pathlib import Path
import pprint

import wandb

from src.config.config import WANDB_ENTITY, DistillConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.tasks.distillation.train import build_repo_name, run_distillation, setup_wandb
from src.utils import load_model

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

    sweep_config = {
        "name": f"{config.language_code}-{config.student_model}-{config.task}",
        "method": "grid",
        "metric": {"goal": "minimize", "name": "eval/loss"},
        "parameters": {
            "lr": {"values": [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]}
        }
    }
    entity = WANDB_ENTITY
    project = config.wandb_project
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=entity,
        project=project
    )
    hf_repo_id = config.hf_repo_id
    config.hf_repo_id = None
    def one_run():
        run = setup_wandb(config)
        config.learning_rate = run.config["lr"]
        run.config.update({"learning_rate": config.learning_rate}, allow_val_change=True)
        run_distillation(config)
    wandb.agent(sweep_id, function=one_run, count=5)
    sweep = wandb.Api().sweep(f"{entity}/{project}/sweeps/{sweep_id}")
    best_run = sweep.best_run()

    # Push winner to hub
    winner_path = Path(config.output_dir) / f"{best_run.name}-final.ckpt"
    student, tokenizer = load_model(str(winner_path), device=config.device)
    config.hf_repo_id = hf_repo_id
    repo_name = build_repo_name(config)
    logger.info(f"Pushing to HF Hub: {repo_name}")
    student.push_to_hub(repo_name, commit_message="Distilled model") # type:ignore
    tokenizer.push_to_hub(repo_name, commit_message="Distilled tokenizer")
