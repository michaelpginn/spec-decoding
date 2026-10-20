"""
Story generation task — concrete noun + descriptive adjective seeds.
"""

import csv
import functools
import random
import re
from collections import Counter
from typing import Any

import nltk
import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from nltk.corpus import brown, wordnet as wn

from src.config.config import ExperimentConfig
from src.data.dataset import get_language_name

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


nltk.download("brown", quiet=True)
nltk.download("wordnet", quiet=True)

CONCRETENESS_MIN = 4.0
ADJ_POS_MIN = 0.9
ADJ_MIN_OCCURRENCES = 10
EMBEDDING_MODEL = "glove-wiki-gigaword-100"
TOP_K_ADJ = 3
MAX_VOCAB_RANK = 40000  # GloVe vocab is frequency-ordered
NOUN_LEXNAMES = ("noun.person", "noun.animal")
# WordNet ancestors that make a noun a story setting (e.g. palace, village, river)
PLACE_HYPERNYMS = ("building.n.01", "geological_formation.n.01", "body_of_water.n.01",
                   "urban_area.n.01", "tract.n.01")
FICTION_RATIO_MIN = 1.5
ADJ_MAX_USES = 6
# Manually vetted: offensive, mature, or otherwise unsuitable story seeds
BLOCKLIST = frozenset("""
    whore pimp prostitute mistress slave assassin hostage rapist kidnapper gunman
    sniper negro gypsy bourbon egg feather wing tail beef hood mount bum
    naked nude intimate dead alive stupid ugly violent insane painful sick
    white yellow brown black pink fat dumb
    aware occasional usual long wrong ready dry bare faint limp thick
    hot cool fresh sudden heavy sharp delicate bitter alien neat dim
    rump feeder booby napoleon algae seaweed vertebrate primates mammal livestock
    poultry fowl tackle horseback female male blonde escort claw paw fly
    brothel casino morgue cavity pit trough site tee infield outfield fairway grill
    terrain graveyard battleground
    crazy mad comfortable cheap dull decent dirty
    accountant advisor analyst anthropologist applicant archaeologist attendant
    attorney broker buyer chairman chemist columnist commentator congressman
    contestant contractor controller coroner counselor custodian cyclist dean
    demonstrator editor educator engraver entertainer executive geographer geologist
    golfer gymnast headmaster instructor interviewer investor jockey juror lecturer
    librarian manager midshipman pathologist performer petitioner pharmacist physicist
    proprietor prosecutor psychiatrist psychologist publicist receptionist retailer
    senator shopper songwriter spectator spokesman sportsman sprinter steward
    stockbroker stylist superintendent supervisor taxpayer technician tenant therapist
    undertaker valet viewer vocalist watcher employee executive politician governor
    president cop
    athlete batter catcher fullback halfback infielder linebacker lineman outfielder
    pitcher quarterback referee shortstop teammate runner player wrestler
    mane snout beak carrier charger conveyor pop planter primate cardinal mate chick
    cricket turkey cocoon larva carcass reptile bacteria parasite human adult guy yank
    mutant zombie fisher grower marine seaman picket bat
    attacker hijacker smuggler mercenary executioner intruder gunner guerrilla convict
    pope archbishop cleric reverend minister
    python anaconda rhinoceros salamander
    bank steakhouse motel cafeteria clubhouse garage dormitory campus infirmary resort
    synagogue cathedral churchyard waterway inlet estuary strait gulf gully knoll
    ledge slope lawn backyard coop
    conductor speaker sage collector greenhouse cattle terrier glacier
""".split())

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


def _is_story_noun(word: str) -> bool:
    """Singular noun whose most frequent WordNet sense is a person, animal, or place."""
    synsets = wn.synsets(word, pos="n")
    if not synsets or wn.morphy(word, "n") != word:
        return False
    first = synsets[0]
    sense_count = sum(lem.count() for lem in first.lemmas() if lem.name().lower() == word)
    if sense_count == 0 or first.instance_hypernyms():  # rare sense or proper noun
        return False
    ancestors = {h.name() for h in first.closure(lambda s: s.hypernyms())}
    return first.lexname() in NOUN_LEXNAMES or not ancestors.isdisjoint(PLACE_HYPERNYMS)


def _is_relational(word: str) -> bool:
    """Adjective whose most frequent WordNet sense is relational (e.g. 'agricultural')."""
    synsets = wn.synsets(word, pos="a")
    return not synsets or synsets[0].lexname() == "adj.pert"


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
            if score < CONCRETENESS_MIN or word in BLOCKLIST or not _is_story_noun(word):
                continue
            seen.add(word)
            nouns.append(word)
    return nouns


@functools.lru_cache(maxsize=1)
def _load_descriptive_adjs() -> list[str]:
    """Adjectives that are tagged JJ in most Brown fiction uses and are
    over-represented in fiction relative to the rest of Brown."""
    other_cats = [c for c in brown.categories() if c not in _BROWN_STORY_CATEGORIES]
    other = Counter(str(w).lower() for w in brown.words(categories=other_cats))
    n_other = sum(other.values())
    total: dict[str, int] = {}
    jj: dict[str, int] = {}
    for word, tag in brown.tagged_words(categories=_BROWN_STORY_CATEGORIES):
        w = str(word).lower()
        if not _WORD_RE.match(w):
            continue
        total[w] = total.get(w, 0) + 1
        if str(tag) == "JJ":
            jj[w] = jj.get(w, 0) + 1

    n_fiction = sum(total.values())
    adjs: list[str] = []
    for w, n in total.items():
        if n < ADJ_MIN_OCCURRENCES or w in BLOCKLIST:
            continue
        fiction_ratio = ((n + 1) / n_fiction) / ((other[w] + 1) / n_other)
        if fiction_ratio < FICTION_RATIO_MIN:
            continue
        if jj.get(w, 0) / n >= ADJ_POS_MIN and not _is_relational(w):
            adjs.append(w)
    return adjs


def _topk_adjs(
    noun: str, adj_pool: list[str], adj_matrix: np.ndarray, vecs: Any, k: int
) -> list[str]:
    """Return the k adjectives most cosine-similar to noun, most similar first."""
    sims = vecs.cosine_similarities(vecs[noun], adj_matrix)
    top_idx = np.argsort(-sims)[:k]
    return [adj_pool[i] for i in top_idx]


def load_data(config: ExperimentConfig, tokenizer) -> tuple[DatasetDict, str]:
    """Generate semantically-related adj+noun seeds as story prompts."""
    lang_name = get_language_name(config.language_code)
    n = config.max_samples
    rng = random.Random(config.story_seed)

    vecs = _load_vectors()
    in_vocab = lambda w: vecs.key_to_index.get(w, MAX_VOCAB_RANK) < MAX_VOCAB_RANK
    adjs = [w for w in _load_descriptive_adjs() if in_vocab(w)]
    nouns = [w for w in _load_concrete_nouns() if in_vocab(w)]
    if not adjs or not nouns:
        raise RuntimeError(
            f"Adjective pool ({len(adjs)}) or noun pool ({len(nouns)}) is empty "
            f"after filtering to words present in '{EMBEDDING_MODEL}' vocabulary."
        )
    rng.shuffle(nouns)

    adj_matrix = np.stack([vecs[a] for a in adjs])
    adj_uses: Counter[str] = Counter()
    seeds = []
    for noun in nouns[:n]:
        candidates = [a for a in _topk_adjs(noun, adjs, adj_matrix, vecs, len(adjs))
                      if adj_uses[a] < ADJ_MAX_USES][:TOP_K_ADJ]
        adj = rng.choice(candidates)
        adj_uses[adj] += 1
        seeds.append(f"{adj} {noun}")

    dataset = Dataset.from_dict({"source": seeds, "target": [""] * len(seeds)})
    return DatasetDict({"test": dataset}), lang_name


def compute_eval_metrics(
    references: list[str], hypotheses: list[str], verbose: bool = False
) -> dict:
    return {}
