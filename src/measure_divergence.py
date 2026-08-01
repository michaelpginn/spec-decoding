"""Computes the kl-divergence (P||Q) and LK divergence (total variation distance)

This is a bit redundant with the eval_kl script, we should probably unify later"""

import argparse
from typing import Mapping

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data.dataset import assemble_dataset, get_language_name
from src.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument("language_code")
parser.add_argument("--p", help="HF model key to use for P distribution")
parser.add_argument("--q", help="HF model key to use for Q distribution")
args = parser.parse_args()

p_model, p_tokenizer = load_model(args.p)
q_model, q_tokenizer = load_model(args.q)
device = next(p_model.parameters()).device

language = get_language_name(args.language_code)
dataset = assemble_dataset(args.language_code, 'mono', p_tokenizer, None)['test']
dataloader = DataLoader(
    dataset,  # type: ignore[arg-type]
    batch_size=4,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)

p_model.eval()
q_model.eval()

mean_kl = 0.
mean_lk = 0.
for batch in tqdm(dataloader):
    breakpoint()
    inputs = p_tokenizer(row['text'], return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        p_out = p_model(**inputs)
        q_out = q_model(**inputs)
        p_logprobs = torch.nn.functional.log_softmax(p_out.logits[..., :-1, :].contiguous(), dim=-1)
        q_logprobs = torch.nn.functional.log_softmax(q_out.logits[..., :-1, :].contiguous(), dim=-1)
        kl = (torch.exp(p_logprobs) * (p_logprobs - q_logprobs)).sum(-1).mean()
        lk = (1/2 * torch.abs(p_logprobs - q_logprobs)).sum(-1).mean()
        mean_kl += kl.item() / len(dataset)
        mean_lk += lk.item()/ len(dataset)

print(f"KL: {mean_kl}")
print(f"LK: {mean_lk}")
