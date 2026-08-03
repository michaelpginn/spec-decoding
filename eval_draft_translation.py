from collections import defaultdict
from pprint import pprint
from typing import Mapping

from tqdm import tqdm

from src.config.config import ExperimentConfig
from src.data.create_inputs import create_inputs, create_prompt
from src.tasks.translation import compute_eval_metrics, load_data
from src.utils import load_model
langs = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh"]

base_verifier_model, tokenizer = load_model("meta-llama/Llama-3.2-3B-Instruct")
base_draft_model, _ = load_model("meta-llama/Llama-3.2-1B-Instruct")
device = next(base_draft_model.parameters()).device
metrics = dict()
for lang in langs:
    distilled_model, _ = load_model(f"lecslab/{lang}-translation-Llama-3.2-3B-Instruct-Llama-3.2-1B-Instruct")
    config = ExperimentConfig(
        task="translation",
        language_code=lang,
        draft_model=None,
        target_model="",
        draft_model_type="none",
        decoding_mode='sample'
    )
    dataset, _ = load_data(config, tokenizer)
    dataset = dataset['test']

    verifier_preds = []
    draft_preds = []
    distilled_preds = []
    for row in tqdm(dataset, desc="Decoding"):
        assert isinstance(row, Mapping)
        prompt = create_prompt(config.task, lang, row['source'])
        inputs = create_inputs(prompt, tokenizer, device)
        prompt_len = inputs["input_ids"].shape[1]

        # Draft
        out = base_verifier_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50,
        )
        generated_token_count = out.shape[1] - prompt_len
        decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        verifier_preds.append(decoded)

        # Draft
        out = base_draft_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50,
        )
        generated_token_count = out.shape[1] - prompt_len
        decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        draft_preds.append(decoded)

        # Distilled
        out = distilled_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50,
        )
        generated_token_count = out.shape[1] - prompt_len
        decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        distilled_preds.append(decoded)
    # Metrics
    references = dataset['target']
    metrics[lang] = {
        "verifier": compute_eval_metrics(references, verifier_preds),
        "draft": compute_eval_metrics(references, draft_preds),
        "draft_distilled": compute_eval_metrics(references, distilled_preds)
    }
    print(lang)
    pprint(metrics[lang])
