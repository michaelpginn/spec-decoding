"""
Knowledge Distillation 
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge distillation from 8-1 B"
    )
    # Model configuration
    parser.add_argument(
        "--teacher_model", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID for teacher"
    )
    parser.add_argument(
        "--student_model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for student"
    )
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="uonlp/CulturaX",
                       help="HuggingFace dataset name or path to local dataset")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset config name (e.g., 'ne' for CulturaX)")
    parser.add_argument("--dataset_text_column", type=str, default="text",
                       help="Column name containing text (default: 'text')")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_streaming", type=lambda x: (str(x).lower() == 'true'), default=True,
                   help="Use streaming mode (no full download). Pass 'true' or 'false'")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to local dataset file (JSON, CSV, etc.)")
    parser.add_argument(
        "--min_text_length", 
        type=int, 
        default=100,
        help="Minimum text length to filter short examples"
    )
    
    # Training configuration
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    
    # Distillation parameters
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss (1-alpha for hard labels)")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    
    return parser.parse_args()

def load_training_dataset(args):
    """Load dataset from various sources."""

    streaming = args.dataset_streaming    
    if args.dataset_path:
        # Load from local file
        print(f"Loading dataset from local file: {args.dataset_path}")
        if args.dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=args.dataset_path, 
                                 split=args.dataset_split, streaming=streaming)
        elif args.dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=args.dataset_path,
                                 split=args.dataset_split, streaming=streaming)
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
                streaming=streaming
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                split=args.dataset_split,
                streaming=streaming
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
            args.resume_from, torch_dtype=dtype, device_map="auto"
        )
    else:
        print(f"Loading fresh student model: {args.student_model}")
        student = AutoModelForCausalLM.from_pretrained(
            args.student_model, torch_dtype=dtype, device_map="auto"
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
    
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
    
    mask = attention_mask.float()
    if mask.sum() > 0:
        loss = (temperature ** 2) * ((kl_per_token * mask).sum() / mask.sum())
    else:
        loss = torch.tensor(0.0, device=student_logits.device)
    
    return loss


def compute_hard_label_loss(student_logits, input_ids):
    """Compute cross-entropy loss for next-token prediction."""
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )


def train(args):
    """Main training loop."""
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    teacher, student, tokenizer, device = load_models(args)
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate)
    
    # Dataset (streaming mode - no full download needed)
    print(f"Loading dataset: {args.dataset_name} (config: {args.dataset_config or 'default'})...")
    # Load dataset
    dataset = load_training_dataset(args)
    
    # Determine starting step
    start_step = 0
    if args.resume_from:
        # Extract step number from checkpoint name
        checkpoint_name = os.path.basename(args.resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])
            # Fast-forward dataset
            items_to_skip = start_step * args.batch_size
            print(f"Fast-forwarding dataset: skipping {items_to_skip} examples...")
            dataset = dataset.skip(items_to_skip)
    
    data_iter = iter(dataset)
    
    # Training loop
    step = start_step
    target_step = start_step + args.max_steps
    accum_loss = 0.0
    start_time = time.time()
    
    print(f"Training from step {start_step} to {target_step}")
    
    while step < target_step:
        # Prepare batch
        batch_texts = []
        while len(batch_texts) < args.batch_size:
            try:
                text = next(data_iter)[args.dataset_text_column]
                if text.strip():
                    batch_texts.append(text)
            except StopIteration:
                print("End of dataset stream.")
                break
        
        if not batch_texts:
            break
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)
        
        # Forward passes
        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits
        student_logits = student(**inputs).logits
        
        # Loss computation
        loss_distill = compute_distillation_loss(
            student_logits, teacher_logits, 
            inputs["attention_mask"], args.temperature
        )
        loss_hard = compute_hard_label_loss(student_logits, inputs["input_ids"])
        loss = (args.alpha * loss_distill) + ((1 - args.alpha) * loss_hard)
        
        # Safety check
        if torch.isnan(loss):
            print(f"Skipping step {step}: Loss is NaN")
            optimizer.zero_grad()
            continue
        
        # Backward
        (loss / args.grad_accum_steps).backward()
        accum_loss += loss.item()
        
        # Optimizer step
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            if (step + 1) % args.log_every == 0:
                avg_loss = accum_loss / args.grad_accum_steps
                elapsed = time.time() - start_time
                print(f"Step {step+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
                accum_loss = 0.0
                start_time = time.time()
            
            # Checkpoint
            if (step + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step+1}")
                print(f"Saving checkpoint: {checkpoint_path}")
                student.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
        
        step += 1
    
    # Final save
    final_path = os.path.join(args.output_dir, "final")
    print(f"Training complete! Saving final model to: {final_path}")
    student.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)