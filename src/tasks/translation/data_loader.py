"""
Load translation data from various sources.
"""
from pathlib import Path
import csv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REFERENCE_TABLE = REPO_ROOT / "reference_table_bilingual.csv"
DATA_DIR = REPO_ROOT / "data"

def get_language_name(lang_code: str) -> str:
    """
    Get full language name from language code using reference_table.csv.
    e.g. 'npi' -> 'Nepali', 'chr' -> 'Cherokee'
    """
    lang_code = lang_code.strip().lower()
    # Use utf-8-sig
    with open(REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Code"].strip().lower() == lang_code:
                return row["Language"].strip()
    # Fallback: return the code itself if not found
    return lang_code


def _get_tatoeba_path(target_lang: str) -> Path:
    """Resolve path to Tatoeba TSV for a given target language."""
    target_lang = target_lang.strip().lower()
    with open(REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Code"].strip().lower() == target_lang and row["source"].strip().lower() == "tatoeba":
                return REPO_ROOT / row["path"].strip()
    raise FileNotFoundError(f"No tatoeba data for language '{target_lang}' in {REFERENCE_TABLE}")


def load_tatoeba_data(target_lang: str, max_samples: int | None = None) -> list[tuple[str, str]]:
    """
    Load (source, target) pairs from Tatoeba TSV.
    
    Args:
        target_lang: Language code, e.g. 'ber', 'chr', 'haw', 'npi'
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of (source_text, target_text) tuples
    """
    path = _get_tatoeba_path(target_lang)
    pairs = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            _id, source, _user, target = row[0], row[1], row[2], row[3]
            source = source.strip()
            target = target.strip()
            if not source or not target:
                continue
            pairs.append((source, target))
            if max_samples is not None and max_samples > 0 and len(pairs) >= max_samples:
                break
        return pairs