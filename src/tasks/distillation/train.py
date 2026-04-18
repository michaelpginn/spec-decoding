"""
Distillation training loop.

Supports two modes (controlled by DistillConfig.distill_mode):
  - task_specific: SeqKD on pre-generated teacher translations (bilingual).
  - general: causal LM fine-tuning on raw monolingual text.
"""
import logging
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from src.config.config import DistillConfig
from src.utils import load_model
from src.tasks.distillation.data_loader import (
    load_general_dataset,
    load_seqkd_dataset,
    tokenize_general,
    tokenize_seqkd,
)

logger = logging.getLogger(__name__)


def _model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def build_repo_name(config: DistillConfig, dataset_len: int) -> str:
    student = _model_short_name(config.student_model)
    if config.distill_mode == "general":
        prefix = "general-kd"
    else:
        teacher = _model_short_name(config.teacher_model)
        prefix = f"seqkd-{teacher}"
    name = f"{prefix}-{student}-{config.language_code}-{dataset_len}"
    if config.hf_repo_id:
        return f"{config.hf_repo_id}/{name}"
    return name


def setup_wandb(config: DistillConfig):
    """Initialize wandb for distillation run tracking."""
    teacher_short = _model_short_name(config.teacher_model)
    student_short = _model_short_name(config.student_model)

    name = (
        f"{config.distill_mode}_{config.language_code}"
        f"_lr{config.learning_rate}_steps{config.max_steps}_ga{config.grad_accum_steps}"
    )
    group = f"distill_{teacher_short}__{config.language_code}"

    tags = [
        "distillation",
        config.language_code,
        config.distill_mode,
        teacher_short,
        student_short,
        f"lr={config.learning_rate}",
        f"steps={config.max_steps}",
        f"ga={config.grad_accum_steps}",
    ]
    # Set by scripts/sweep_distill.sh: learning_rate_sweep_runs | steps_grad_accum_sweep
    _sweep_tag = os.environ.get("WANDB_DISTILL_SWEEP_TAG", "").strip()
    if _sweep_tag:
        tags.append(_sweep_tag)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "spec-decoding"),
        entity=os.environ.get("WANDB_ENTITY", "lecs-general"),
        config=asdict(config),
        group=group,
        job_type=f"distill-{config.distill_mode}",
        name=name,
        tags=tags,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")


def run_distillation(config: DistillConfig):
    """
    Train student via distillation.

    - task_specific: cross-entropy on teacher translations (SeqKD).
    - general: causal LM on monolingual text.
    """
    setup_wandb(config)

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

    # Data — dispatch on distill_mode
    logger.info(f"Distillation mode: {config.distill_mode}")
    if config.distill_mode == "general":
        raw_dataset = load_general_dataset(config)
        dataset_len = len(raw_dataset)
        tokenized = tokenize_general(raw_dataset, tokenizer, config)
    else:
        raw_dataset = load_seqkd_dataset(config)
        dataset_len = len(raw_dataset)
        tokenized = tokenize_seqkd(raw_dataset, tokenizer, config)

    repo_name = build_repo_name(config, dataset_len)
    logger.info(f"HF repo: {repo_name}")

    dataloader = DataLoader(
        tokenized,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=config.learning_rate)
    start_step = _restore_optimizer(config, optimizer, device)

    # AMP scaler (only needed for float16, not bfloat16)
    use_scaler = device.type == "cuda" and student.dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)

    # Training loop
    step = start_step
    target_step = start_step + config.max_steps
    accum_count = 0
    log_accum_loss = 0.0
    log_step_count = 0
    start_time = time.time()
    epoch = 0

    logger.info(f"Training from step {start_step} to {target_step}")

    while step < target_step:
        for batch in dataloader:
            if step >= target_step:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            if torch.isnan(loss):
                logger.warning(f"Skipping step {step}: loss is NaN")
                optimizer.zero_grad()
                accum_count = 0
                step += 1
                continue

            scaler.scale(loss / config.grad_accum_steps).backward()
            accum_count += 1
            log_accum_loss += loss.item()
            log_step_count += 1

            if accum_count >= config.grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            if (step + 1) % config.log_every == 0 and log_step_count > 0:
                avg_loss = log_accum_loss / log_step_count
                elapsed = time.time() - start_time
                logger.info(f"Step {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
                wandb.log({"train/loss": avg_loss, "train/epoch": epoch, "step": step + 1})
                log_accum_loss = 0.0
                log_step_count = 0
                start_time = time.time()

            if (step + 1) % config.save_every == 0:
                _save_checkpoint(
                    student, tokenizer, optimizer, config.output_dir,
                    step + 1, repo_name, push_to_hub=bool(config.hf_repo_id),
                )

            step += 1

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    if config.hf_repo_id:
        logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    else:
        logger.info("Training complete! Saving final checkpoint locally (HF Hub push disabled).")
    _save_checkpoint(
        student, tokenizer, optimizer, config.output_dir, "final", repo_name,
        push_to_hub=bool(config.hf_repo_id),
    )
    wandb.finish()


def _restore_optimizer(config: DistillConfig, optimizer, device) -> int:
    """Restore optimizer state from checkpoint and return the starting step."""
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
    return start_step


def _save_checkpoint(student, tokenizer, optimizer, output_dir, label,
                     repo_name=None, push_to_hub=False):
    """Save model, tokenizer, and optimizer state; optionally push to HF Hub."""
    path = os.path.join(output_dir, f"checkpoint-{label}" if isinstance(label, int) else str(label))
    os.makedirs(path, exist_ok=True)

    student.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    logger.info(f"Saved checkpoint: {path}")

    if push_to_hub and repo_name:
        hub_repo = f"{repo_name}-{label}" if isinstance(label, int) else repo_name
        logger.info(f"Pushing to HF Hub: {hub_repo}")
        student.push_to_hub(hub_repo, commit_message=f"Distilled model (step {label})")
        tokenizer.push_to_hub(hub_repo, commit_message=f"Tokenizer (step {label})")
        logger.info(f"Pushed: https://huggingface.co/{hub_repo}")
