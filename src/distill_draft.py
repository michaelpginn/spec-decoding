"""
Knowledge Distillation 
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge distillation"
    )
    # Model configuration
    parser.add_argument(
        "--teacher-model", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID for teacher"
    )
    parser.add_argument(
        "--student-model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for student"
    )
    
    # Dataset configuration
    parser.add_argument("--dataset-name", type=str, default="uonlp/CulturaX",
                       help="HuggingFace dataset name or path to local dataset")
    parser.add_argument("--dataset-config", type=str, default=None,
                       help="Dataset config name (e.g., 'ne' for CulturaX)")
    parser.add_argument("--dataset-text-column", type=str, default="text",
                       help="Column name containing text (default: 'text')")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to local dataset file (JSON, CSV, etc.)")
    parser.add_argument(
        "--min-text-length", 
        type=int, 
        default=100,
        help="Minimum text length to filter short examples"
    )
    
    # Training configuration
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    
    # Distillation parameters
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss (1-alpha for hard labels)")
    
    # Checkpointing
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    
    return parser.parse_args()

def load_training_dataset(args):
    """Load dataset from various sources."""

    if args.dataset_path:
        # Load from local file
        print(f"Loading dataset from local file: {args.dataset_path}")
        if args.dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=args.dataset_path, 
                                 split=args.dataset_split, streaming=False)
        elif args.dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=args.dataset_path,
                                 split=args.dataset_split, streaming=False)
        else:
            raise ValueError(f"Unsupported file format: {args.dataset_path}")
    else:
        # Load from HuggingFace
        config_info = f" (config: {args.dataset_config})" if args.dataset_config else ""
        print(f"Loading dataset: {args.dataset_name}{config_info}...")
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset_name, 
                args.dataset_config,
                split=args.dataset_split,
                streaming=False
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                split=args.dataset_split,
                streaming=False
            )
    
    # Filter by length
    dataset = dataset.filter(lambda x: len(x[args.dataset_text_column]) > args.min_text_length)
    
    return dataset


def load_models(args):
    """Load teacher and student models with appropriate configurations."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device} | Precision: {dtype}")
    
    # Load tokenizer from teacher
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load teacher (4-bit quantized to save memory)
    print("Loading teacher model (frozen, 4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    teacher.eval()
    
    # Load student (from checkpoint or fresh)
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        student = AutoModelForCausalLM.from_pretrained(
            args.resume_from, dtype=dtype, device_map="auto"
        )
    else:
        print(f"Loading fresh student model: {args.student_model}")
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model, dtype=dtype, device_map="auto"
        )
    
    student.train()
    
    # Resize embeddings if needed
    if len(tokenizer) > student.get_input_embeddings().num_embeddings:
        student.resize_token_embeddings(len(tokenizer))
    
    return teacher, student, tokenizer, device


def compute_distillation_loss(student_logits, teacher_logits, attention_mask, temperature):
    """Compute KL divergence loss between student and teacher distributions."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Compute per-token, per-vocabulary KL divergence
    kl_per_token_vocab = F.kl_div(
        student_log_probs, teacher_probs, reduction='none'
    )
    
    mask = attention_mask.float().to(student_logits.device)
    expanded_mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
    kl_per_token = (kl_per_token_vocab * expanded_mask).sum(dim=-1)
    
    if mask.sum() > 0:
        loss = (temperature ** 2) * (kl_per_token.sum() / mask.sum())
    else:
        loss = torch.tensor(0.0, device=student_logits.device)
    
    return loss


def compute_hard_label_loss(student_logits, input_ids, attention_mask=None, ignore_index: int = -100):
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
        ignore_index=ignore_index
    )


def train(args):
    """Main training loop."""
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    teacher, student, tokenizer, device = load_models(args)
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate)
    
    print(f"Loading dataset: {args.dataset_name} (config: {args.dataset_config or 'default'})...")
    # Load dataset
    dataset = load_training_dataset(args)

    # pretokenize the dataset
    print("pretokenizing the dataset....")
    
    def tokenize_function(examples):
        """"Tokenize a batch of examples"""
        return tokenizer(
            examples[args.dataset_text_column],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[args.dataset_text_column],
        desc="Tokenizing dataset"
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    
    # Determine starting step
    start_step = 0
    if args.resume_from:
        # Extract step number from checkpoint name
        checkpoint_name = os.path.basename(args.resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])

        # load the optimizer state
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer state from {optimizer_path}")
            model_device = next(student.parameters()).device
            optimizer_state = torch.load(optimizer_path, map_location=model_device)
            optimizer.load_state_dict(optimizer_state)
            print("Successfully loaded optimizer state - training will resume with previous state")
        else:
            print("No optimizer state found at checkpoint. adaptive learning rates will reset !!!")
    
    # Training loop
    step = start_step
    target_step = start_step + args.max_steps
    accum_loss = 0.0
    accum_count = 0
    log_accum_loss = 0.0
    log_step_count = 0
    start_time = time.time()
    epoch = 0
    
    print(f"Training from step {start_step} to {target_step}")
    
    while step < target_step:
        # loop through the dataset
        for batch in dataloader:
            if step >= target_step:
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            model_device = next(student.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
            # Forward passes
            with torch.no_grad():
                teacher_logits = teacher(**inputs).logits
            student_logits = student(**inputs).logits
            
            # Loss computation
            loss_distill = compute_distillation_loss(
                student_logits, teacher_logits, 
                attention_mask, args.temperature
            )
            loss_hard = compute_hard_label_loss(student_logits, input_ids, attention_mask)
            loss = (args.alpha * loss_distill) + ((1 - args.alpha) * loss_hard)
            
            # Safety check
            if torch.isnan(loss):
                print(f"Skipping step {step}: Loss is NaN")
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_count = 0
                step += 1
                continue
        
            # Backward
            (loss / args.grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_count += 1

            log_accum_loss += loss.item()
            log_step_count += 1
            
            # Optimizer step
            if accum_count >= args.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                accum_loss = 0.0
                accum_count = 0
                
                # Logging
            if (step + 1) % args.log_every == 0:
                avg_loss = log_accum_loss / log_step_count  # Average over log_every steps
                elapsed = time.time() - start_time
                print(f"Step {step+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
                log_accum_loss = 0.0
                log_step_count = 0
                start_time = time.time()
            
            # Checkpoint
            if (step + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step+1}")
                print(f"Saving checkpoint: {checkpoint_path}")
                student.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                # save optimizer state
                optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
                torch.save(optimizer.state_dict(), optimizer_path)
                print(f"Saved optimizer state to {optimizer_path}")
        
            step += 1
        epoch += 1
        if step < target_step:
            print(f"Completed epoch {epoch}. Continuing to step {target_step}...")
    
    # Final save
    final_path = os.path.join(args.output_dir, "final")
    print(f"Training complete! Saving final model to: {final_path}")
    student.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    final_optimizer_path = os.path.join(final_path, "optimizer.pt")
    torch.save(optimizer.state_dict(), final_optimizer_path)
    print(f"Saved final optimizer state to {final_optimizer_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
