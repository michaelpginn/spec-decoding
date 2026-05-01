"""
Generate teacher logits for Sequence-Level Knowledge Distillation (SeqKD).

Outputs are saved as a parquet file, where each row contains the following:
    - token_ids: A tensor of token IDs, including the prompt and generated output
    - prompt_length: The length of just the prompt part of the token IDs
    - logprobs: (seq_len, top_k) tensor for each token (excluding the prompt), with the top K logprob values
    - logprobs_vocab_idx: (seq_len, top_k) the corresponding vocabulary indices for each logprob

Usage:
    python scripts/generate_teacher_logprobs.py experiments/distillation/logprobs_translation.cfg
    python scripts/generate_teacher_logprobs.py experiments/logprobs_translation.cfg -o language_code=ber max_samples=5000
"""

import argparse
import logging
import os
import pprint
from typing import Mapping

import torch
from datasets import Dataset
from tqdm import tqdm

from src.config.config import DistillConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.data.create_inputs import create_inputs, create_prompt
from src.data.dataset import load_monolingual_dataset
from src.utils import load_model
from src.tasks.translation import _get_language_name

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_logprobs(
    model, inputs: dict, top_k: int,
):
    with torch.no_grad():
        out = model(**inputs)
    logprobs = torch.nn.functional.log_softmax(out.logits[0, :-1, :], dim=-1)
    topk, topk_indices = torch.topk(logprobs, k=top_k, dim=-1)
    return {
        "token_ids": inputs['input_ids'][0],
        "prompt_length": 1, # for the BOS token
        "logprobs": topk,
        "logprobs_vocab_idx": topk_indices,
    }

def generate_with_logprobs(
    model, tokenizer, inputs: dict, max_new_tokens: int, top_k: int,
) -> dict:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_logits=True,
            return_dict_in_generate=True
        )
    logits = torch.stack(out.logits).squeeze(1) # (num_new_tokens, vocab_size)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    topk, topk_indices = torch.topk(logprobs, k=top_k, dim=-1)
    return {
        "token_ids": out.sequences[0],
        "prompt_length": inputs["input_ids"].shape[1],
        "logprobs": topk,
        "logprobs_vocab_idx": topk_indices,
    }


def generate_teacher_logprobs(config: DistillConfig) -> Dataset:
    """Compute logprobs (generating if necessary) with the teacher and return a HF Dataset."""
    # 1. Load data
    logger.info(f"Loading train split for {config.language_code}...")
    if config.task == 'general':
        language = _get_language_name(config.language_code)
        dataset = load_monolingual_dataset(language, 'mono', True)['train']
        if config.max_samples and config.max_samples <= len(dataset):
            dataset = dataset.select(range(config.max_samples))
    elif config.task == "translation":
        from src.tasks.translation import load_data
        dataset, language = load_data(config)
        dataset = dataset["train"]
    else:
        raise NotImplementedError()
    logger.info(f"Loaded dataset: {pprint.pformat(dataset)}")

    # 2. Load teacher model
    logger.info(f"Loading teacher model: {config.teacher_model}")
    model, tokenizer = load_model(config.teacher_model, device=config.device)
    device = next(model.parameters()).device

    # 3. Generate translations
    data: list[dict] = []
    for row in tqdm(dataset, desc="Generating logprobs"):
        assert isinstance(row, Mapping)
        if config.task == 'general':
            inputs = tokenizer(row['text'], return_tensors="pt").to(device)
            outputs = compute_logprobs(model, inputs, top_k=config.top_k)
        else:
            prompt = create_prompt(config.task, language, row["source"])
            inputs = create_inputs(prompt, tokenizer, device)
            outputs = generate_with_logprobs(
                model,
                tokenizer,
                inputs,
                max_new_tokens=config.max_length,
                top_k=config.top_k,
            )
        data.append(outputs)
    ds = Dataset.from_list(data)
    logger.info(f"Generated dataset: {len(ds)} examples")
    return ds


def _teacher_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher logprobs")
    parser.add_argument("config", help="Config file (cfg/ini)")
    parser.add_argument(
        "--overrides",
        "-o",
        help="Override config values: key1=value1 key2=value2",
        nargs="+",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push generated dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=DistillConfig,
    )
    logger.info(f"SeqKD generation config:\n{pprint.pformat(config)}")

    dataset = generate_teacher_logprobs(config)

    teacher_short = _teacher_short_name(config.teacher_model)
    dataset_name = f"logprobs-{teacher_short}-{config.language_code}-{config.task}.parquet"
    save_dir = os.path.join(config.output_dir, dataset_name)
    dataset.to_parquet(save_dir)
    logger.info(f"Saved dataset locally: {save_dir}")

    if args.push_to_hub and config.hf_repo_id:
        repo_id = f"{config.hf_repo_id}/{dataset_name}"
        logger.info(f"Pushing dataset to HF Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        logger.info(f"Pushed: https://huggingface.co/datasets/{repo_id}")
    elif args.push_to_hub:
        logger.warning("--push-to-hub requested but hf_repo_id is not set in config")

    logger.info("Done!")
