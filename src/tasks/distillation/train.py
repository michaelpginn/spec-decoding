"""
Distillation training loop. Uses a parquet data file created with `scripts/generate_teacher_logprobs`.
"""
import logging
import math
import os
import time
from dataclasses import asdict

import datasets
import torch
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.config import WANDB_ENTITY, DistillConfig
from src.utils import load_model

logger = logging.getLogger(__name__)


def _model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def build_repo_name(config: DistillConfig) -> str:
    student = _model_short_name(config.student_model)
    teacher = _model_short_name(config.teacher_model)
    name = f"{config.language_code}-{config.task}-{teacher}-{student}"
    if config.hf_repo_id:
        return f"{config.hf_repo_id}/{name}"
    return name


def setup_wandb(config: DistillConfig):
    """Initialize wandb for distillation run tracking."""
    teacher_short = _model_short_name(config.teacher_model)
    student_short = _model_short_name(config.student_model)

    group = f"distill_{teacher_short}__{config.language_code}"

    tags = [
        "distillation",
        config.language_code,
        config.task,
        teacher_short,
        student_short,
        f"lr={config.learning_rate}",
        f"steps={config.max_steps}",
        f"ga={config.grad_accum_steps}",
    ]

    run = wandb.init(
        project=config.wandb_project,
        entity=WANDB_ENTITY,
        config=asdict(config),
        group=group,
        job_type=f"distill-{config.task}",
        tags=tags,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    return run


def _build_scheduler(optimizer, config: DistillConfig) -> LambdaLR:
    """Build LR scheduler with linear warmup then cosine/linear/constant decay."""
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if config.lr_scheduler == "constant":
            return 1.0
        progress = (current_step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        if config.lr_scheduler == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # linear
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(student, batch, device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    topk_logprobs = batch["topk_logprobs"].to(device, non_blocking=True)
    topk_logprobs_idx = batch["topk_logprobs_indices"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device, non_blocking=True)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :].contiguous(), dim=-1)
        student_logprobs = logprobs.gather(dim=-1, index=topk_logprobs_idx)
        loss = -(torch.exp(topk_logprobs) * student_logprobs).sum(-1)
        loss = (loss * label_mask).sum() / label_mask.sum().clamp(min=1)
    return loss

@torch.no_grad()
def _compute_eval_loss(student, eval_dataloader, device) -> float:
    """Run a forward pass over the eval split and return average loss."""
    student.eval()
    total_loss = 0.0
    count = 0
    for batch in eval_dataloader:
        loss = compute_loss(student, batch, device)
        total_loss += loss.item()
        count += 1
    student.train()
    return total_loss / max(count, 1)


def run_distillation(config: DistillConfig):
    """
    Train student via distillation.

    - task_specific: cross-entropy on teacher translations (SeqKD).
    - general: causal LM on monolingual text.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(f"Loading student model: {config.student_model}")
    student, tokenizer = load_model(config.student_model, device=config.device)

    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming student from checkpoint: {config.resume_from}")
        student, _ = load_model(config.resume_from, device=config.device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    num_embeddings: int = student.get_input_embeddings().num_embeddings  # type: ignore[assignment]
    if len(tokenizer) > num_embeddings:
        student.resize_token_embeddings(len(tokenizer))

    device = next(student.parameters()).device

    assert config.dataset_path
    dataset = datasets.Dataset.from_parquet(config.dataset_path)
    assert isinstance(dataset, datasets.Dataset)
    # There's a few one-token samples which we can't use for training
    dataset = dataset.filter(lambda r: len(r['logprobs']) > 0)
    repo_name = build_repo_name(config)
    logger.info(f"HF repo: {repo_name}")

    # Train / eval split (randomized, seeded for reproducibility).
    if config.eval_split_ratio > 0 and len(dataset) > 1:
        split = dataset.train_test_split(
            test_size=config.eval_split_ratio, seed=42,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = dataset.select([])
    logger.info(
        f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval examples"
    )

    def collate_fn(batch):
        # Build input IDs and full logits
        bs = len(batch)
        seq_len = max([len(r["token_ids"]) for r in batch])
        topk = len(batch[0]["logprobs"][0])

        input_ids = torch.full((bs, seq_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        # Avoid materializing these as full vocab dim
        # Note: shifted on seq dim (first item is logprobs for second token)
        topk_logprobs = torch.zeros((bs, seq_len - 1, topk), dtype=student.dtype)
        topk_logprobs_indices = torch.zeros((bs, seq_len - 1, topk), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len - 1), dtype=student.dtype) # Mask positions that shouldn't be trained

        for idx in range(bs):
            item_seq_len = len(batch[idx]["token_ids"])
            item_prompt_len = batch[idx]["prompt_length"]
            input_ids[idx][0:item_seq_len] = torch.as_tensor(batch[idx]["token_ids"])
            attention_mask[idx][0:item_seq_len] = 1
            topk_logprobs[idx][item_prompt_len-1:item_seq_len-1] = torch.as_tensor(batch[idx]["logprobs"])
            topk_logprobs_indices[idx][item_prompt_len-1:item_seq_len-1] = torch.as_tensor(batch[idx]["logprobs_vocab_idx"])
            label_mask[idx][item_prompt_len-1:item_seq_len-1] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "topk_logprobs": topk_logprobs,
            "topk_logprobs_indices": topk_logprobs_indices,
        }

    dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Optimizer with weight decay (exclude bias and LayerNorm)
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    param_groups = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(param_groups, lr=config.learning_rate)
    scheduler = _build_scheduler(optimizer, config)
    start_step = _restore_training_state(config, optimizer, scheduler, device)

    # AMP scaler (only needed for float16, not bfloat16)
    use_scaler = device.type == "cuda" and student.dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)

    # Training loop
    step = start_step
    target_step = start_step + config.max_steps
    accum_count = 0
    log_accum_loss = 0.0
    log_micro_count = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    epoch = 0

    logger.info(f"Training from step {start_step} to {target_step} (optimizer steps)")

    while step < target_step:
        for batch in dataloader:
            if step >= target_step:
                break

            loss = compute_loss(student, batch, device)
            if torch.isnan(loss):
                logger.warning(f"Step {step}, micro-batch NaN — skipping accumulation window")
                optimizer.zero_grad()
                accum_count = 0
                continue

            scaler.scale(loss / config.grad_accum_steps).backward()
            accum_count += 1
            log_accum_loss += loss.item()
            log_micro_count += 1

            if accum_count < config.grad_accum_steps:
                continue

            # Optimizer step (this is one "step")
            scaler.unscale_(optimizer)
            unclipped_grad_norm = grad_norm(student)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            accum_count = 0
            step += 1

            if step % config.log_every == 0 and log_micro_count > 0:
                avg_loss = log_accum_loss / log_micro_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
                )
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/grad_norm": unclipped_grad_norm,
                    "step": step,
                })
                log_accum_loss = 0.0
                log_micro_count = 0
                start_time = time.time()

            if step % config.eval_every == 0 and len(eval_dataset) > 0:
                eval_loss = _compute_eval_loss(student, eval_dataloader, device)
                logger.info(f"Step {step} | Eval loss: {eval_loss:.4f}")
                wandb.log({"eval/loss": eval_loss, "step": step})
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(
                        student, tokenizer, optimizer, config.output_dir, "best", repo_name,
                        push_to_hub=False, scheduler=scheduler,
                    )

            if step >= target_step:
                break

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    wandb.log({"eval/best_loss": best_eval_loss})

    if config.hf_repo_id:
        logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    else:
        logger.info("Training complete! Saving final checkpoint locally (HF Hub push disabled).")
    _save_checkpoint(
        student, tokenizer, optimizer, config.output_dir, "final", repo_name,
        push_to_hub=bool(config.hf_repo_id), scheduler=scheduler,
    )
    wandb.finish()


def _restore_training_state(config: DistillConfig, optimizer, scheduler, device) -> int:
    """Restore optimizer and scheduler state from checkpoint; return starting step."""
    start_step = 0
    if config.resume_from:
        checkpoint_name = os.path.basename(config.resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            logger.info(f"Loading optimizer state from {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        else:
            logger.warning("No optimizer state found — learning rates will reset")

        sched_path = os.path.join(config.resume_from, "scheduler.pt")
        if os.path.exists(sched_path):
            logger.info(f"Loading scheduler state from {sched_path}")
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))
        elif start_step > 0:
            logger.warning(
                f"No scheduler state found — fast-forwarding scheduler to step {start_step}"
            )
            for _ in range(start_step):
                scheduler.step()
    return start_step


def _save_checkpoint(student, tokenizer, optimizer, output_dir, label,
                     repo_name=None, push_to_hub=False, scheduler=None):
    """Save model, tokenizer, optimizer, and scheduler state; optionally push to HF Hub."""
    run_name = wandb.run.name # type:ignore
    path = os.path.join(output_dir, f"{run_name}-{label}.ckpt")
    os.makedirs(path, exist_ok=True)

    student.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    logger.info(f"Saved checkpoint: {path}")

    if push_to_hub and repo_name:
        hub_repo = f"{repo_name}-{label}" if isinstance(label, int) else repo_name
        logger.info(f"Pushing to HF Hub: {hub_repo}")
        student.push_to_hub(hub_repo, commit_message=f"Distilled model (step {label})")
        tokenizer.push_to_hub(hub_repo, commit_message=f"Tokenizer (step {label})")
        logger.info(f"Pushed: https://huggingface.co/{hub_repo}")

def grad_norm(model):
    # Log grad norm
    grad_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm**0.5
    return grad_norm
