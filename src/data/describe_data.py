"""Stats about the dataset

uv run -m src.data.describe_data
"""

from collections import Counter
from pprint import pprint
from src.data.dataset import assemble_dataset, get_language_name

MAX_SAMPLES=6000

languages = [
    "amh", "ber", "chr", "grn", "haw", "ibo", "npi", "oci","que", "yor", "zgh", "zh"
]

# Cut: lkt,mus,oji

lang_data = []
for language in languages:
    data: dict = {'code': language}
    data['name'] = get_language_name(language)
    print(data['name'])
    try:
        mono_data = assemble_dataset(language, 'mono', MAX_SAMPLES)
        data['mono'] = {
            'train': {
                'num_examples': len(mono_data['train']),
                'sources': Counter(mono_data['train']['source'])
            },
            'test': {
                'num_examples': len(mono_data['test']),
                'sources': Counter(mono_data['test']['source'])
            }
        }
    except Exception as e:
        print(language, "MONO", e)
    try:
        bi_data = assemble_dataset(language, 'bi', MAX_SAMPLES)
        data['bi'] = {
            'train': {
                'num_examples': len(bi_data['train']),
                'sources': Counter(bi_data['train']['source'])
            },
            'test': {
                'num_examples': len(bi_data['test']),
                'sources': Counter(bi_data['test']['source'])
            }
        }
    except Exception as e:
        print(language, "BI", e)
    lang_data.append(data)

pprint(lang_data)
