"""
Story generation task — WordNet-based short-phrase seeds.
"""

import itertools

import nltk
from datasets import Dataset, DatasetDict
from nltk.corpus import wordnet as wn

from src.config.config import ExperimentConfig
from src.data.dataset import get_language_name

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def _synset_names(pos: str, n: int) -> list[str]:
    return [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets(pos), n)]


def load_data(config: ExperimentConfig, tokenizer) -> tuple[DatasetDict, str]:
    """Generate adjective+noun seeds (e.g. "able entity") as story prompts."""
    lang_name = get_language_name(config.language_code)
    n = config.max_samples

    adjs = _synset_names('a', n)
    nouns = _synset_names('n', n)
    seeds = [f"{a} {noun}" for a, noun in zip(adjs, nouns)]

    dataset = Dataset.from_dict({"source": seeds, "target": [""] * len(seeds)})
    return DatasetDict({"test": dataset}), lang_name


def compute_eval_metrics(
    references: list[str], hypotheses: list[str], verbose: bool = False
) -> dict:
    return {}
