"""Stats about the dataset

uv run -m src.data.describe_data
"""

from collections import Counter
import logging
from pprint import pprint

from transformers import AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lang_data = []
for language in languages:
    data: dict = {'code': language}
    data['name'] = get_language_name(language)
    print(data['name'])
    mono_data = assemble_dataset(language, 'mono', tokenizer, MAX_MONO)
    data['mono'] = {
        'train': {
            'num_examples': len(mono_data['train']),
            'sources': Counter(mono_data['train']['origin']),
            'first_example': mono_data['train'][0]
        },
        'test': {
            'num_examples': len(mono_data['test']),
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
    lang_data.append(data)

pprint(lang_data)
