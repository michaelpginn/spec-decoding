"""
load the target and the draft models for speculative translation tasks
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def load_target_model(model_name: str, device: str = "auto"):
    """
    load huggingface model and the tokenizer for the target model.
    returns model and the tokenizer.
    """
    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        device_map="auto" if dev.type == "cuda" else None,
        trust_remote_code=True,
    )
    if dev.type == "cpu" and getattr(model, "device", None) != dev:
        model = model.to(dev)
    return model, tokenizer

def load_draft_model(draft_model_type: str, draft_model_path: str | None, **kwargs):
    raise NotImplementedError(
        "Draft model not implemented yet. Use target model only for now."
    )


def translate_target(model, tokenizer, source: str, target_lang: str, max_length: int = 512, device=None, debug: bool = False):
    """Translate using Qwen chat template."""
    # Qwen's chat template
    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text from English to {target_lang}. Maintain the original style and tone. Only output the translation."
        },
        {
            "role": "user",
            "content": f"Translate this:\n\n{source}"
        }
    ]
    
    # Apply chat template (handles <|im_start|> and <|im_end|> automatically)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Adds <|im_start|>assistant\n
    )
    
    # Debug: print the prompt (only when debug=True)
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'='*60}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    out = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the new tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return decoded.strip()