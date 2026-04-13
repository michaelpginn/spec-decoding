from pathlib import Path

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/CohereLabs/aya_dataset/" + splits["train"])

filter_lang = [
    'English',
    'Spanish',
    'French',
    'German',
    'Japanese',
    'Korean',
    'Italian',
    'Simplified Chinese',
    'Standard Arabic',
    'Russian',
    'Polish',
    'Turkish',
    'Traditional Chinese'
]

lang_code: set[str] = {lang for lang in set(df['language']) if lang not in filter_lang}

df = df[df['language'].isin(list(lang_code))]

base_dir = Path.home() / "Desktop" / "datas"

for lang in lang_code:
    print(f"{lang}")
    lang_folder = base_dir / lang
    lang_folder.mkdir(parents=True, exist_ok=True)
    csv_path = lang_folder / f"{lang}.csv"

    csv_lang = df[df["language"] == lang]
    csv_lang = pd.DataFrame(csv_lang)
    new_series = pd.concat(
        [csv_lang['inputs'], csv_lang['targets']],
        ignore_index=True
    )
    csv_output = new_series.to_frame(name=f'{lang}')
    csv_output.to_csv(csv_path, index=False)
