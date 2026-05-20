"""Stats about the dataset

uv run -m src.data.describe_data
"""

import logging
from collections import Counter
from pathlib import Path
from pprint import pprint

from transformers import AutoTokenizer

#
from src.data.dataset import assemble_dataset, get_language_name

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_MONO = 20000
MAX_BI=6000

languages = [
    "amh", "ber", "chr", "grn", "haw", "ibo", "npi", "oci","que", "yor", "zgh", "zh"
]

# Cut: lkt,mus,oji

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

def add_token_counts(row):
    return {'num_tokens': len(tokenizer.tokenize(row['text']))}

lang_data = {}
for language in languages:
    data: dict = {'code': language}
    data['name'] = get_language_name(language)
    print(data['name'])
    mono_data = assemble_dataset(language, 'mono', tokenizer, MAX_MONO)
    mono_data = mono_data.map(add_token_counts)
    data['mono'] = {
        'train': {
            'num_examples': len(mono_data['train']),
            'num_tokens': sum(mono_data['train']['num_tokens']),
            'sources': Counter(mono_data['train']['origin']),
            'first_example': mono_data['train'][0]
        },
        'test': {
            'num_examples': len(mono_data['test']),
            'num_tokens': sum(mono_data['test']['num_tokens']),
            'sources': Counter(mono_data['test']['origin'])
        }
    }
    bi_data = assemble_dataset(language, 'bi', tokenizer, MAX_BI)
    data['bi'] = {
        'train': {
            'num_examples': len(bi_data['train']),
            'sources': Counter(bi_data['train']['origin']),
            'first_example': bi_data['train'][0]
        },
        'test': {
            'num_examples': len(bi_data['test']),
            'sources': Counter(bi_data['test']['origin'])
        }
    }
    lang_data[language] = data

pprint(lang_data)

# Make latex tables
Path("./viz").mkdir(exist_ok=True)

mono_table = """\\begin{table}[h!]
    \\small
    \\centering
        \\begin{tabular}{l c c c}
            \\toprule
            \\textbf{Language} & \\textbf{Train toks} & \\textbf{Test toks} \\\\
            \\midrule
"""

def short(n):
    for div, suf in [(1e9,'B'), (1e6,'M'), (1e3,'k')]:
        if abs(n) >= div: return f"{n/div:.1f}{suf}"
    return str(n)

for lang in sorted(languages):
    d = lang_data[lang]
    num_tokens_train = short(d['mono']['train']['num_tokens'])
    num_tokens_test = short(d['mono']['test']['num_tokens'])
    mono_table += f"            {d['name']} [{lang}] & {num_tokens_train} & {num_tokens_test} \\\\ \n"

mono_table += """            \\bottomrule
        \\end{tabular}
    \\caption{Monolingual corpora for each language, with token counts under the Qwen tokenizer. Sources are described in \\autoref{tab:mono_source_counts}.}
    \\label{tab:monolingual}
\\end{table}"""
with open("./viz/monolingual.tex", 'w') as f:
    f.write(mono_table)

parallel_table = """\\begin{table}[h!]
    \\small
    \\centering
        \\begin{tabular}{l c c c}
            \\toprule
            \\textbf{Language} & \\textbf{\\# Train} & \\textbf{\\# Test} \\\\
            \\midrule
"""

for lang in sorted(languages):
    d = lang_data[lang]
    parallel_table += f"            {d['name']} [{lang}] & {d['bi']['train']['num_examples']} & {d['bi']['test']['num_examples']} \\\\ \n"

parallel_table += """            \\bottomrule
        \\end{tabular}
    \\caption{Number of parallel sentences for each language. Sources are described in \\autoref{tab:par_source_counts}. Our main evaluation uses the test split.}
    \\label{tab:bilingual}
\\end{table}"""
with open("./viz/bilingual.tex", 'w') as f:
    f.write(parallel_table)

# Source Count
mono_source_table = """\\begin{table}[h!]
    \\small
    \\centering
        \\begin{tabular}{l c l}
            \\toprule
            \\textbf{Language} & \\textbf{Total Tokens} & \\textbf{Sources} \\\\
            \\midrule
"""
for lang in sorted(languages):
    d = lang_data[lang]
    total_tokens = short(d['mono']['train']['num_tokens'] + d['mono']['test']['num_tokens'])

    sources_list = ", ".join(d['mono']['train']['sources'].keys())

    mono_source_table += f"            {d['name']} [{lang}] & {total_tokens} & {sources_list} \\\\ \n"

mono_source_table += """            \\bottomrule
        \\end{tabular}
    \\caption{monolingual source counts for each language.}
    \\label{tab:mono_source_counts}
\\end{table}"""

with open("./viz/monolingual_source.tex", 'w') as f:
    f.write(mono_source_table)


bi_source_table = """\\begin{table}[h!]
    \\small
    \\centering
        \\begin{tabular}{l c l}
            \\toprule
            \\textbf{Language} & \\textbf{Total Tokens} & \\textbf{Sources} \\\\
            \\midrule
"""

for lang in sorted(languages):
    d = lang_data[lang]
    total_tokens = short(d['bi']['train']['num_examples'] + d['bi']['test']['num_examples'])
    temp = ""
    for item in d['bi']['train']['sources'].keys():
        if item.lower() == "tateoba":
            temp += ", Tat"
        elif item.lower() == "https://cherokeedictionary.net":
            temp += ", ChEn"
        elif item.lower() == "opus":
            temp += ", Opus"
        elif item == "Durbin Feeling Cherokee English Dictionary 1975":
            temp += ", Che"
        elif "menyo" in item:
            temp += ", Men"
        else:
            temp += f", {item}"
    sources_list = temp

    bi_source_table += f"            {d['name']} [{lang}] & {total_tokens} & {sources_list} \\\\ \n"

bi_source_table += """            \\bottomrule
        \\end{tabular}
    \\caption{Bilingual source counts for each language.}
    \\label{tab:bi_source_counts}
\\end{table}"""

with open("./viz/bilingual_source.tex", 'w') as f:
    f.write(bi_source_table)
