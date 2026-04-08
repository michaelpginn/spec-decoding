"""
Distillation training loop.

Supports two modes (controlled by DistillConfig.distill_mode):
  - task_specific: SeqKD on pre-generated teacher translations (bilingual).
  - general: causal LM fine-tuning on raw monolingual text.
"""
import logging
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
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
    if config.hf_repo_id and config.hf_repo_id != "None":
        return f"{config.hf_repo_id}/{name}"
    return name


def run_distillation(config: DistillConfig):
    """
    Train student via distillation.

    - task_specific: cross-entropy on teacher translations (SeqKD).
    - general: causal LM on monolingual text.
    """

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(f"Loading student model: {config.student_model}")
    student, tokenizer = load_model(config.student_model, device=config.device)

    if config.resume_from and config.resume_from != "None" and os.path.exists(config.resume_from):
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
                log_accum_loss = 0.0
                log_step_count = 0
                start_time = time.time()

            if (step + 1) % config.save_every == 0:
                _save_checkpoint(student, tokenizer, optimizer, config.output_dir, step + 1, repo_name)

            step += 1

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    _save_checkpoint(student, tokenizer, optimizer, config.output_dir, "final", repo_name, push_to_hub=True)


def _restore_optimizer(config: DistillConfig, optimizer, device) -> int:
    """Restore optimizer state from checkpoint and return the starting step."""
    start_step = 0
    if config.resume_from and config.resume_from != "None":
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
        logger.info(f"Pushing to HF Hub: {repo_name}")
        student.push_to_hub(repo_name, commit_message=f"SeqKD distilled model ({label})")
        tokenizer.push_to_hub(repo_name, commit_message=f"Tokenizer ({label})")
        logger.info(f"Pushed: https://huggingface.co/{repo_name}")
