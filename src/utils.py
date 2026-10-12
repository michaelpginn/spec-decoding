"""
Load models for speculative decoding tasks.
"""
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_model(model_name: str, device: str = "auto"):
    """
    Load a HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path
        device: "cuda", "mps", "cpu", or "auto"
                "auto" prefers cuda > mps (Apple Silicon) > cpu

    Returns:
        tuple of (model, tokenizer)
    """
    dev = _resolve_device(device)
    logger.info(f"Loading {model_name} on device={dev}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # bfloat16 on cuda and mps (supported since PyTorch 2.0 on Apple Silicon).
    # On MPS/CPU: load to CPU first, then .to(device).
    if dev.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if dev.type == "mps" else torch.float32,
            trust_remote_code=True,
        )
        model = model.to(dev)

    model.eval()
    return model, tokenizer
