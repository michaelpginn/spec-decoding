from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    target_model: str
    draft_model: str | None
    language_code: str
    
    draft_model_type: Literal["none", "neural", "statistical"]
    decoding_mode: Literal["greedy", "top_k", "top_p"]

    task: str = "translation"
    data_source: str = "tatoeba"
    max_samples: int = 5
    max_new_tokens: int = 256
    gamma: int = 5
    use_hf_assisted: bool = False
    device: str = "auto"