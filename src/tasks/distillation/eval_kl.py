"""
Compute average KL(teacher || student) on the translation dev split for each language,
using the distilled student at lecslab/{lang}-general-Qwen3.5-9B-Qwen3.5-0.8B.

The dev split mirrors `train.py`: load the teacher-logprobs parquet for the language,
filter empty rows, then take the test fold of a 95/5 train_test_split (seed=42).

Writes a CSV with: language_code, kl_divergence, min_acceptance_rate (= 1 - sqrt(KL/2)).

Usage:
    uv run python -m src.tasks.distillation.eval_kl \\
        --output kl_results.csv \\
        --logprobs-dir logprobs/
"""
import argparse
import csv
import logging
import math
import os

import datasets
import torch
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)

LANGUAGES = ["amh", "ber", "chr", "grn", "haw", "ibo", "npi", "oci", "que", "yor", "zgh", "zh"]


def compute_kl(student, batch, device) -> torch.Tensor:
    """Modified compute_loss: returns KL(teacher || student) averaged over label positions."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    topk_logprobs = batch["topk_logprobs"].to(device, non_blocking=True)
    topk_logprobs_idx = batch["topk_logprobs_indices"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device, non_blocking=True)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :].contiguous(), dim=-1)
        student_logprobs = logprobs.gather(dim=-1, index=topk_logprobs_idx)
        kl = (torch.exp(topk_logprobs) * (topk_logprobs - student_logprobs)).sum(-1)
        kl = (kl * label_mask).sum() / label_mask.sum().clamp(min=1)
    return kl


def build_collate_fn(tokenizer, student_dtype):
    def collate_fn(batch):
        bs = len(batch)
        seq_len = max(len(r["token_ids"]) for r in batch)
        topk = len(batch[0]["logprobs"][0])

        input_ids = torch.full((bs, seq_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        topk_logprobs = torch.zeros((bs, seq_len - 1, topk), dtype=student_dtype)
        topk_logprobs_indices = torch.zeros((bs, seq_len - 1, topk), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len - 1), dtype=student_dtype)

        for idx in range(bs):
            item_seq_len = len(batch[idx]["token_ids"])
            item_prompt_len = batch[idx]["prompt_length"]
            input_ids[idx][0:item_seq_len] = torch.as_tensor(batch[idx]["token_ids"])
            attention_mask[idx][0:item_seq_len] = 1
            topk_logprobs[idx][item_prompt_len - 1:item_seq_len - 1] = torch.as_tensor(batch[idx]["logprobs"])
            topk_logprobs_indices[idx][item_prompt_len - 1:item_seq_len - 1] = torch.as_tensor(batch[idx]["logprobs_vocab_idx"])
            label_mask[idx][item_prompt_len - 1:item_seq_len - 1] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "topk_logprobs": topk_logprobs,
            "topk_logprobs_indices": topk_logprobs_indices,
        }
    return collate_fn


@torch.no_grad()
def evaluate_language(
    lang: str, logprobs_dir: str, teacher_short: str, model_template: str,
    batch_size: int, eval_split_ratio: float, device: str,
) -> float:
    model_id = model_template.format(lang=lang)
    parquet_path = os.path.join(logprobs_dir, f"logprobs-{teacher_short}-{lang}-translation.parquet")
    logger.info(f"[{lang}] Loading student: {model_id}")
    student, tokenizer = load_model(model_id, device=device)
    student.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dev = next(student.parameters()).device

    logger.info(f"[{lang}] Loading parquet: {parquet_path}")
    dataset = datasets.Dataset.from_parquet(parquet_path)
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.filter(lambda r: len(r["logprobs"]) > 0)
    split = dataset.train_test_split(test_size=eval_split_ratio, seed=42)
    eval_dataset = split["test"]
    logger.info(f"[{lang}] Dev examples: {len(eval_dataset)}")

    dataloader = DataLoader(
        eval_dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(dev.type == "cuda"),
        collate_fn=build_collate_fn(tokenizer, student.dtype),
    )

    total_kl = 0.0
    count = 0
    for batch in dataloader:
        kl = compute_kl(student, batch, dev)
        total_kl += kl.item()
        count += 1
    avg_kl = total_kl / max(count, 1)
    logger.info(f"[{lang}] Avg KL: {avg_kl:.6f}")

    del student, tokenizer, dataloader, eval_dataset, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_kl


def main():
    parser = argparse.ArgumentParser(description="Compute KL(teacher || student) on translation dev splits.")
    parser.add_argument("--output", default="kl_results.csv", help="Output CSV path.")
    parser.add_argument("--logprobs-dir", default="logprobs/", help="Directory holding teacher logprobs parquets.")
    parser.add_argument("--teacher-short", default="Qwen3.5-9B",
                        help="Teacher short name used in parquet filenames.")
    parser.add_argument("--model-template",
                        default="lecslab/{lang}-general-Qwen3.5-9B-Qwen3.5-0.8B",
                        help="Student model HF id template; '{lang}' substituted per language.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-split-ratio", type=float, default=0.05,
                        help="Matches train.py default for dev fold.")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = []
    for lang in args.languages:
        try:
            kl = evaluate_language(
                lang=lang,
                logprobs_dir=args.logprobs_dir,
                teacher_short=args.teacher_short,
                model_template=args.model_template,
                batch_size=args.batch_size,
                eval_split_ratio=args.eval_split_ratio,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"[{lang}] Failed: {e}")
            continue
        min_accept = 1.0 - math.sqrt(kl / 2.0) if kl >= 0 else float("nan")
        rows.append({"language_code": lang, "kl_divergence": kl, "min_acceptance_rate": min_accept})

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language_code", "kl_divergence", "min_acceptance_rate"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
