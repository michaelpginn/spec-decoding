"""
Story generation task — concrete noun + descriptive adjective seeds.
"""

import csv
import functools
import random
import re
from typing import Any

import nltk
import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from nltk.corpus import brown

from src.config.config import ExperimentConfig
from src.data.dataset import get_language_name

nltk.download("brown", quiet=True)

CONCRETENESS_MIN = 4.0
ADJ_POS_MIN = 0.7
ADJ_MIN_OCCURRENCES = 2
EMBEDDING_MODEL = "glove-wiki-gigaword-100"
TOP_K_ADJ = 20

_BROWN_STORY_CATEGORIES = (
    "fiction",
    "adventure",
    "mystery",
    "romance",
    "science_fiction",
)

CONCRETENESS_REPO_ID = "lecslab/brysbaert_concreteness"
CONCRETENESS_FILENAME = "concreteness_ratings_brysbaert.csv"

_WORD_RE = re.compile(r"^[a-z]+$")


@functools.lru_cache(maxsize=1)
def _load_vectors() -> Any:
    """Load GloVe vectors lazily"""
    import gensim.downloader as api  # type: ignore[import-untyped]

    return api.load(EMBEDDING_MODEL)


@functools.lru_cache(maxsize=1)
def _concreteness_csv_path() -> str:
    """Fetch the Brysbaert CSV from HF Hub; cached locally after first call."""
    return hf_hub_download(
        repo_id=CONCRETENESS_REPO_ID,
        filename=CONCRETENESS_FILENAME,
        repo_type="dataset",
    )


@functools.lru_cache(maxsize=1)
def _load_concrete_nouns() -> list[str]:
    """Concrete single-word nouns from Brysbaert ratings on HF Hub."""
    nouns: list[str] = []
    seen: set[str] = set()
    with open(_concreteness_csv_path(), newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            word, score_str = row[0].strip().lower(), row[1].strip()
            if word in seen or not _WORD_RE.match(word):
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            if score < CONCRETENESS_MIN:
                continue
            seen.add(word)
            nouns.append(word)
    return nouns


@functools.lru_cache(maxsize=1)
def _load_descriptive_adjs() -> list[str]:
    """Adjectives that are tagged JJ in most Brown fiction uses."""
    total: dict[str, int] = {}
    jj: dict[str, int] = {}
    for word, tag in brown.tagged_words(categories=_BROWN_STORY_CATEGORIES):
        w = str(word).lower()
        if not _WORD_RE.match(w):
            continue
        total[w] = total.get(w, 0) + 1
        if str(tag).startswith("JJ"):
            jj[w] = jj.get(w, 0) + 1

    adjs: list[str] = []
    for w, n in total.items():
        if n < ADJ_MIN_OCCURRENCES:
            continue
        if jj.get(w, 0) / n >= ADJ_POS_MIN:
            adjs.append(w)
    return adjs


def _topk_adjs(
    noun: str, adj_pool: list[str], adj_matrix: np.ndarray, vecs: Any, k: int
) -> list[str]:
    """Return the k adjectives most cosine-similar to noun."""
    sims = vecs.cosine_similarities(vecs[noun], adj_matrix)
    k = min(k, len(adj_pool))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    return [adj_pool[i] for i in top_idx]


def load_data(config: ExperimentConfig, tokenizer) -> tuple[DatasetDict, str]:
    """Generate semantically-related adj+noun seeds as story prompts."""
    lang_name = get_language_name(config.language_code)
    n = config.max_samples
    rng = random.Random(config.story_seed)

    vecs = _load_vectors()
    adjs = [w for w in _load_descriptive_adjs() if w in vecs]
    nouns = [w for w in _load_concrete_nouns() if w in vecs]
    if not adjs or not nouns:
        raise RuntimeError(
            f"Adjective pool ({len(adjs)}) or noun pool ({len(nouns)}) is empty "
            f"after filtering to words present in '{EMBEDDING_MODEL}' vocabulary."
        )
    rng.shuffle(nouns)

    adj_matrix = np.stack([vecs[a] for a in adjs])
    seeds = [
        f"{rng.choice(_topk_adjs(noun, adjs, adj_matrix, vecs, TOP_K_ADJ))} {noun}"
        for noun in nouns[:n]
    ]

    dataset = Dataset.from_dict({"source": seeds, "target": [""] * len(seeds)})
    return DatasetDict({"test": dataset}), lang_name


def compute_eval_metrics(
    references: list[str], hypotheses: list[str], verbose: bool = False
) -> dict:
    return {}
