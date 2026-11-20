"""chrF of the verifier, the base draft model and the translation-distilled draft
model, per language, for both model families, sampled with the same parameters
as the speculative decoding runs. Writes viz/translation_scores.tex.

Scores are cached in viz/translation_scores.json after every (family, language),
so a rerun skips what is already done. Pass --force to recompute everything, or
--table-only to rebuild the table from the cache without loading any model.
"""
import argparse
import gc
import json
from pathlib import Path
from typing import Mapping

import torch
from tqdm import tqdm

from src.config.config import ExperimentConfig
from src.data.create_inputs import create_inputs, create_prompt
from src.data.dataset import HIGH_RESOURCE_LANGUAGES, LANGUAGES
from src.tasks.translation import compute_eval_metrics, load_data
from src.utils import load_model

FAMILIES = {
    "qwen": {
        "label": "Qwen",
        "verifier": "Qwen/Qwen3.5-9B",
        "draft": "Qwen/Qwen3.5-0.8B",
        "distilled": "lecslab/{lang}-translation-Qwen3.5-9B-Qwen3.5-0.8B",
        "columns": ["9b", "0.8b", "0.8b dist"],
    },
    "llama": {
        "label": "Llama",
        "verifier": "meta-llama/Llama-3.2-3B-Instruct",
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "distilled": "lecslab/{lang}-translation-Llama-3.2-3B-Instruct-Llama-3.2-1B-Instruct",
        "columns": ["3b", "1b", "1b dist"],
    },
}
ROLES = ["verifier", "draft", "distilled"]

# English is only ever the source language, so there is nothing to translate into.
HIGH_RESOURCE = [lang for lang in HIGH_RESOURCE_LANGUAGES if lang != "eng"]
# Drawn as commented-out rows, matching the paper's table.
COMMENTED_OUT = {"chr"}

CACHE = Path("viz/translation_scores.json")
TABLE = Path("viz/translation_scores.tex")


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save_cache(scores: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(scores, indent=2))


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(model, tokenizer, inputs, prompt_len: int, max_new_tokens: int) -> str:
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def eval_family(family: str, langs: list[str], scores: dict, force: bool):
    cfg = FAMILIES[family]
    fam_scores = scores.setdefault(family, {})
    todo = [lang for lang in langs if force or lang not in fam_scores]
    if not todo:
        print(f"[{family}] all languages cached")
        return

    verifier, tokenizer = load_model(cfg["verifier"])
    draft, _ = load_model(cfg["draft"])
    device = next(draft.parameters()).device

    for lang in todo:
        try:
            distilled, _ = load_model(cfg["distilled"].format(lang=lang))
        except Exception as e:  # no distilled model for this language
            print(f"[{family}] {lang}: could not load distilled model ({e}); leaving it blank")
            distilled = None
        models = {"verifier": verifier, "draft": draft, "distilled": distilled}

        config = ExperimentConfig(
            task="translation",
            language_code=lang,
            draft_model=None,
            target_model="",
            draft_model_type="none",
            decoding_mode="sample",
        )
        dataset, language = load_data(config, tokenizer)
        dataset = dataset["test"]

        preds: dict[str, list[str]] = {role: [] for role in ROLES if models[role] is not None}
        for row in tqdm(dataset, desc=f"{family} {lang}"):
            assert isinstance(row, Mapping)
            prompt = create_prompt(config.task, language, row["source"])
            inputs = create_inputs(prompt, tokenizer, device)
            prompt_len = inputs["input_ids"].shape[1]
            for role in preds:
                preds[role].append(
                    _generate(models[role], tokenizer, inputs, prompt_len, config.max_new_tokens)
                )

        references = dataset["target"]
        fam_scores[lang] = {role: compute_eval_metrics(references, p) for role, p in preds.items()}
        print(f"[{family}] {lang}: " + ", ".join(
            f"{role}={m['chrf2']:.1f}" for role, m in fam_scores[lang].items()
        ))
        _save_cache(scores)
        del distilled, models
        _empty_cache()

    del verifier, draft
    _empty_cache()


def _row(lang: str, scores: dict) -> str:
    cells = [lang]
    for family in FAMILIES:
        lang_scores = scores.get(family, {}).get(lang, {})
        for role in ROLES:
            m = lang_scores.get(role)
            cells.append(f"{m['chrf2']:.1f}" if m else "")
    line = " & ".join(cells) + r" \\"
    return f"        % {line}" if lang in COMMENTED_OUT else f"        {line}"


def write_table(scores: dict):
    header_fams = " & ".join(
        rf"\multicolumn{{3}}{{c{'|' if i < len(FAMILIES) - 1 else ''}}}{{{cfg['label']}}}"
        for i, cfg in enumerate(FAMILIES.values())
    )
    header_cols = " & ".join(" & ".join(cfg["columns"]) for cfg in FAMILIES.values())
    lines = [
        r"\begin{table}[h!]",
        r"    \small",
        r"    \centering",
        r"    \begin{tabular}{c | c c c | c c c}",
        r"        \toprule",
        rf"        Lang & {header_fams} \\",
        rf"        & {header_cols} \\",
        r"        \midrule",
        *[_row(lang, scores) for lang in LANGUAGES],
        r"        \hline",
        *[_row(lang, scores) for lang in HIGH_RESOURCE],
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{chrF++ scores for translations generated by the Qwen/Llama verifier models and draft models before and after distillation on translation data. Outputs are generated using sampling with the same parameters as the speculative decoding experiments---resulting in some cases where the distilled draft model looks stronger than the verifier.}",
        r"    \label{tab:translation_scores}",
        r"\end{table}",
    ]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--langs", nargs="+", default=LANGUAGES + HIGH_RESOURCE)
    parser.add_argument("--force", action="store_true", help="recompute cached scores")
    parser.add_argument("--table-only", action="store_true", help="only rebuild the table from the cache")
    args = parser.parse_args()

    scores = _load_cache()
    if not args.table_only:
        for family in args.families:
            eval_family(family, args.langs, scores, args.force)
    write_table(scores)
