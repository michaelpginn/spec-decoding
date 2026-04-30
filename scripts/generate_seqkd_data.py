"""
Generate teacher pseudo-translations for Sequence-Level Knowledge Distillation (SeqKD).

Loads English sentences from bilingual datasets, translates them with the teacher model,
and saves the (source, teacher_translation) pairs as a HuggingFace dataset.

Usage:
    python scripts/generate_seqkd_data.py experiments/seqkd.cfg
    python scripts/generate_seqkd_data.py experiments/seqkd.cfg -o language_code=ber max_samples=5000
"""
import argparse
import logging
import os
import pprint

import torch
from datasets import Dataset
from tqdm import tqdm

from src.config.config import DistillConfig
from src.config.config_to_dataclass import config_to_dataclass
from src.data.create_inputs import create_inputs, create_prompt
from src.data.dataset import load_bilingual_dataset
from src.tasks.translation import _get_language_name
from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_english_sources(
    language_code: str,
    max_samples: int | None,
) -> list[str]:
    """Load English sentences from the bilingual train split."""
    splits = load_bilingual_dataset(language_code, max_samples)
    ds = splits["train"]
    columns = ds.column_names

    if "English" in columns:
        src_col = "English"
    elif len(columns) == 2:
        src_col = columns[0]
    else:
        raise ValueError(f"Cannot determine English column from {columns}")

    sources = []
    seen: set[str] = set()
    for row in ds:
        text = str(row[src_col]).strip()
        if text and text not in seen:
            seen.add(text)
            sources.append(text)

    return sources


def _translate_with_teacher(model, tokenizer, source: str, lang_name: str,
                            max_new_tokens: int, device) -> str:
    """Translate a single sentence using the teacher model."""
    prompt = create_prompt("translation", lang_name, source)
    inputs = create_inputs(prompt, tokenizer, device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def generate_seqkd_dataset(config: DistillConfig) -> Dataset:
    """Translate English sentences with the teacher and return a HF Dataset."""
    lang_name = _get_language_name(config.language_code)
    logger.info(f"Language: {lang_name} ({config.language_code})")

    max_samples = config.max_samples if config.max_samples > 0 else None
    sources = _load_english_sources(config.language_code, max_samples)
    logger.info(f"Loaded {len(sources)} unique English sentences")

    if not sources:
        raise ValueError(f"No bilingual data found for {config.language_code}")

    logger.info(f"Loading teacher model: {config.teacher_model}")
    model, tokenizer = load_model(config.teacher_model, device=config.device)
    device = next(model.parameters()).device

    translations: list[str] = []
    logger.info(f"Generating {len(sources)} teacher translations...")
    for source in tqdm(sources, desc="Teacher translating"):
        translation = _translate_with_teacher(
            model, tokenizer, source, lang_name,
            max_new_tokens=config.max_length, device=device,
        )
        translations.append(translation)

    ds = Dataset.from_dict({
        "source": sources,
        "teacher_translation": translations,
    })
    logger.info(f"Generated dataset: {len(ds)} examples")
    return ds


def _teacher_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SeqKD teacher translations")
    parser.add_argument("config", help="Config file (cfg/ini)")
    parser.add_argument(
        "--overrides", "-o",
        help="Override config values: key1=value1 key2=value2",
        nargs="+",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push generated dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=DistillConfig,
    )
    logger.info(f"SeqKD generation config:\n{pprint.pformat(config)}")

    dataset = generate_seqkd_dataset(config)

    teacher_short = _teacher_short_name(config.teacher_model)
    dataset_name = f"seqkd-{teacher_short}-{config.language_code}-{len(dataset)}"

    save_dir = os.path.join(config.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)
    logger.info(f"Saved dataset locally: {save_dir}")

    if args.push_to_hub and config.hf_repo_id:
        repo_id = f"{config.hf_repo_id}/{dataset_name}"
        logger.info(f"Pushing dataset to HF Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        logger.info(f"Pushed: https://huggingface.co/datasets/{repo_id}")
    elif args.push_to_hub:
        logger.warning("--push-to-hub requested but hf_repo_id is not set in config")

    logger.info("Done!")
