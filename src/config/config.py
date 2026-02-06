from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    target_model: str
    draft_model: str | None
    language_code: str
    
    draft_model_type: Literal["none", "neural", "statistical"]
    decoding_mode: Literal["greedy", "top_k", "top_p"]
