"""
speculative decoding for translation
"""
import torch
import time


def assisted_decode(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,
    source: str,
    target_lang: str,
    max_length: int = 512,
    device=None,
    return_metrics: bool = True,
):
    """
    Use HuggingFace's optimized assisted generation (speculative decoding).
    """
    if device is None:
        device = next(target_model.parameters()).device

    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text from English to {target_lang.upper()}. Maintain the original style and tone. Only output the translation."
        },
        {
            "role": "user",
            "content": f"Translate this:\n\n{source}"
        }
    ]

    prompt = target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    start_time = time.time()
    
    with torch.no_grad():
        outputs = target_model.generate(
            input_ids,
            attention_mask=attention_mask,
            assistant_model=draft_model,
            tokenizer=target_tokenizer,
            assistant_tokenizer=draft_tokenizer,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=target_tokenizer.eos_token_id,
        )
    
    total_time = time.time() - start_time
    generated_tokens = outputs.shape[1] - prompt_len

    translation = target_tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    if return_metrics:
        metrics = {
            "total_time": total_time,
            "generated_tokens": generated_tokens,
            "decode_tps": generated_tokens / total_time if total_time > 0 else 0,
        }
        return translation, metrics
    else:
        return translation
