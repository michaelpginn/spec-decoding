"""
Translation module for speculative decoding experiments.
"""
from .data_loader import load_tatoeba_data, get_language_name
from .models import load_model, load_target_model, translate_target
from .spec_decode import speculative_decode_greedy, speculative_decode_translate, assisted_decode_hf
from .evaluate import compute_bleu, compute_spec_metrics

__all__ = [
    "load_tatoeba_data",
    "get_language_name",
    "load_model",
    "load_target_model",
    "translate_target",
    "speculative_decode_greedy",
    "speculative_decode_translate",
    "assisted_decode_hf",
    "compute_bleu",
    "compute_spec_metrics",
]
