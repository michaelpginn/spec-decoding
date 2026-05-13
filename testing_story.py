import argparse
import logging
import pprint

import src.story_gen_prompt
from src.config.config import ExperimentConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.generation import generate_output
from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main(config:ExperimentConfig):
    target_model, target_tokenizer = load_model(
        config.target_model, device=config.device
    )

    draft_model = None
    if config.draft_model_type != "none":
            draft_model, draft_tokenizer = load_model(
                config.draft_model,
                device=config.device
            )

    prompts = src.story_gen_prompt.create_prompt(
        language_code=config.language_code,
        adj_n=True,
        num_prompts=1
    )

    tokenizer = target_tokenizer
    for prompt in prompts.values():
        messages = [
            {"role": "system", "content": f"You are a helpful assistant who only writes in {config.language_code}."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(config.device)

        decoded_story, metrics = generate_output(
            inputs=inputs,
            model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_tokenizer=draft_model if draft_model else None,
            config=config
        )

        print(f"Generated Story:\n{decoded_story}")
        logger.info(f"Performance Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("--overrides", "-o", nargs="+", help="Config overrides")

    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=ExperimentConfig,
    )

    main(config)
