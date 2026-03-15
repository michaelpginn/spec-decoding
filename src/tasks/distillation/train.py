"""
Knowledge distillation training loop.
"""
import logging
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config.config import DistillConfig
from src.decoding.models import load_model
from src.tasks.distillation.data_loader import load_distillation_dataset

logger = logging.getLogger(__name__)

def _model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def build_repo_name(config: DistillConfig, dataset_len: int) -> str:
    """Build a full HF repo ID: {hf_repo_id}/distill-{teacher}-{student}-{lang}-{len}.

    ``hf_repo_id`` (e.g. ``lecslab``).
    """
    teacher = _model_short_name(config.teacher_model)
    student = _model_short_name(config.student_model)
    name = f"distill-{teacher}-{student}-{config.language_code}-{dataset_len}"
    if config.hf_repo_id and config.hf_repo_id != "None":
        return f"{config.hf_repo_id}/{name}"
    return name

def compute_distillation_loss(student_logits, teacher_logits, attention_mask, temperature):
    """Compute KL divergence loss between student and teacher distributions."""
    # Shift so that tokens < n predict n
    shift_student_logits = student_logits[..., :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()

    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

    # This handles edge cases where the teacher and student vocabularies might differ
    if student_log_probs.size(-1) != teacher_probs.size(-1):
        min_vocab = min(student_log_probs.size(-1), teacher_probs.size(-1))
        student_log_probs = student_log_probs[..., :min_vocab]
        teacher_probs = teacher_probs[..., :min_vocab]

    kl_per_token_vocab = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    )

    expanded_mask = shift_mask.unsqueeze(-1)  # [batch, seq_len-1, 1]
    kl_per_token = (kl_per_token_vocab * expanded_mask).sum(dim=-1)

    if shift_mask.sum() > 0:
        loss = (temperature ** 2) * (kl_per_token.sum() / shift_mask.sum())
    else:
        loss = torch.tensor(0.0, device=student_logits.device)

    return loss


def compute_hard_label_loss(student_logits, input_ids, attention_mask=None, ignore_index=-100):
    """Compute cross-entropy loss for next-token prediction."""
    input_ids = input_ids.to(student_logits.device)

    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    if attention_mask is not None:
        attention_mask = attention_mask.to(student_logits.device)
        shift_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, ignore_index)

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def run_distillation(config: DistillConfig):
    """Main distillation training loop."""

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(f"Loading teacher model: {config.teacher_model}")
    teacher, tokenizer = load_model(config.teacher_model, device=config.device)
    teacher.eval()

    if config.resume_from and config.resume_from != "None" and os.path.exists(config.resume_from):
        logger.info(f"Resuming student from checkpoint: {config.resume_from}")
        student, _ = load_model(config.resume_from, device=config.device)
    else:
        logger.info(f"Loading fresh student model: {config.student_model}")
        student, _ = load_model(config.student_model, device=config.device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for student model")

    if len(tokenizer) > student.get_input_embeddings().num_embeddings:
        student.resize_token_embeddings(len(tokenizer))

    device = next(student.parameters()).device

    dataset, text_col = load_distillation_dataset(config)
    dataset_len = len(dataset)

    repo_name = build_repo_name(config, dataset_len)
    logger.info(f"HF repo: {repo_name}")

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(student.parameters(), lr=config.learning_rate)

    start_step = 0
    if config.resume_from and config.resume_from != "None":
        checkpoint_name = os.path.basename(config.resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            logger.info(f"Loading optimizer state from {opt_path}")
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=device)
            )
        else:
            logger.warning("No optimizer state found — adaptive learning rates will reset")

    step = start_step
    target_step = start_step + config.max_steps
    accum_loss = 0.0
    accum_count = 0
    log_accum_loss = 0.0
    log_step_count = 0
    start_time = time.time()
    epoch = 0

    use_scaler = (device.type == "cuda" and student.dtype == torch.float16)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    logger.info(f"Training from step {start_step} to {target_step}")

    while step < target_step:
        for batch in dataloader:
            if step >= target_step:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    teacher_logits = teacher(**inputs).logits
                student_logits = student(**inputs).logits

                loss_distill = compute_distillation_loss(
                    student_logits, teacher_logits, attention_mask, config.temperature,
                )
                loss_hard = compute_hard_label_loss(student_logits, input_ids, attention_mask)
                loss = (config.alpha * loss_distill) + ((1 - config.alpha) * loss_hard)

            if torch.isnan(loss):
                logger.warning(f"Skipping step {step}: loss is NaN")
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_count = 0
                step += 1
                continue

            scaler.scale(loss / config.grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_count += 1
            log_accum_loss += loss.item()
            log_step_count += 1

            if accum_count >= config.grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_count = 0

            if (step + 1) % config.log_every == 0 and log_step_count > 0:
                avg_loss = log_accum_loss / log_step_count
                elapsed = time.time() - start_time
                logger.info(f"Step {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
                log_accum_loss = 0.0
                log_step_count = 0
                start_time = time.time()

            if (step + 1) % config.save_every == 0:
                _save_checkpoint(
                    student, tokenizer, optimizer,
                    config.output_dir, step + 1, repo_name,
                )

            step += 1

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    _save_checkpoint(
        student, tokenizer, optimizer,
        config.output_dir, "final", repo_name, push_to_hub=True,
    )


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
        student.push_to_hub(repo_name, commit_message=f"Final distilled model ({label})")
        tokenizer.push_to_hub(repo_name, commit_message=f"Tokenizer for distilled model ({label})")
        logger.info(f"Successfully pushed to: https://huggingface.co/{repo_name}")
