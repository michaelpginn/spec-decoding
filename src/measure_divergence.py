"""Computes the kl-divergence (P||Q) and LK divergence (total variation distance)

This is a bit redundant with the eval_kl script, we should probably unify later"""

import argparse
from logging import getLogger

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data.dataset import assemble_dataset, get_language_name
from src.data.describe_data import languages
from src.utils import load_model

logger = getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--p", help="HF model key to use for P distribution")
parser.add_argument("--q", help="HF model key to use for Q distribution")
args = parser.parse_args()

p_model, p_tokenizer = load_model(args.p)
q_model, q_tokenizer = load_model(args.q)
device = next(p_model.parameters()).device

divergences = []

for language_code in languages:
    language = get_language_name(args.language_code)
    logger.info("Running on {}")
    dataset = assemble_dataset(args.language_code, 'mono', p_tokenizer, None)['test']
    dataloader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=16,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    p_model.eval()
    q_model.eval()

    mean_kl = 0.
    mean_lk = 0.
    for batch in tqdm(dataloader):
        inputs = p_tokenizer(batch['text'], return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            p_out = p_model(**inputs)
            q_out = q_model(**inputs)
            p_logprobs = torch.nn.functional.log_softmax(p_out.logits[..., :-1, :].contiguous(), dim=-1)
            q_logprobs = torch.nn.functional.log_softmax(q_out.logits[..., :-1, :].contiguous(), dim=-1)
            kl = (torch.exp(p_logprobs) * (p_logprobs - q_logprobs)).sum(-1)
            lk = (1/2 * torch.abs(p_logprobs.exp() - q_logprobs.exp())).sum(-1)
            label_mask = inputs['attention_mask'][:, 1:]
            kl = kl * label_mask
            lk = lk * label_mask
            mean_kl += kl.mean().item() / len(dataset)
            mean_lk += lk.mean().item()/ len(dataset)

    print(f"KL: {mean_kl}")
    print(f"LK: {mean_lk}")
    divergences.append([language, mean_kl, mean_lk])

p_model_name = args.p.split("/")[-1]
q_model_name = args.q.split("/")[-1]
with open(f"viz/divergences_{p_model_name}_{q_model_name}.csv", 'w') as f:
    for d in divergences:
        f.write(",".join(d) + "\n")
