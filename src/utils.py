"""
Load models for speculative decoding tasks.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def load_model(model_name: str, device: str = "auto"):
    """
    Load a HuggingFace model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        device: "cuda", "cpu", or "auto"
    
    Returns:
        tuple of (model, tokenizer)
    """
    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        device_map="auto" if dev.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    if dev.type == "cpu" and getattr(model, "device", None) != dev:
        model = model.to(dev)
    return model, tokenizer
