from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    task: Literal['translation']
    language_code: str

    target_model: str
    draft_model: str | None
    draft_model_type: Literal["none", "neural", "ngram"]

    decoding_mode: Literal["greedy", "top_k", "top_p"]
    gamma: int = 5
    track_iterations: bool = False # If true, will log per-iteration of SD

    ngram_n: int = 3

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

    data_source: str = "tatoeba"
    max_samples: int = 5
    max_new_tokens: int = 512
    device: str = "auto"

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if self.draft_model_type == 'neural':
            assert self.gamma > 0
            assert self.draft_model is not None
