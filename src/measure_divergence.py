"""Computes the kl-divergence (P||Q) and LK divergence (total variation distance)

This is a bit redundant with the eval_kl script, we should probably unify later"""

import argparse
from typing import Mapping

import torch

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

p_model.eval()
q_model.eval()

mean_kl = 0.
mean_lk = 0.
for row in dataset:
    assert isinstance(row, Mapping)
    inputs = p_tokenizer(row['text'], return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out_p = p_model(**inputs)
        out_q = q_model(**inputs)
        breakpoint()
        kl = (torch.exp(topk_logprobs) * (topk_logprobs - student_logprobs)).sum(-1)
