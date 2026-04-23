from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    task: Literal['translation']
    language_code: str
    include_aya: bool

    target_model: str
    draft_model: str | None
    draft_model_type: Literal["none", "neural", "ngram"]
    decoding_mode: Literal["greedy", "sample"]
    top_k: int = 0
    top_p: float = 0.0
    gamma: int = 5
    track_iterations: bool = False # If true, will log per-iteration of SD

    ngram_n: int = 3

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

    data_source: str = "tatoeba"
    data_start: int = 0
    data_end: int = 0  # 0 means "no explicit end"
    max_samples: int = 5
    max_new_tokens: int = 512
    device: str = "auto"
    
    wandb_tag: str | None = None

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if self.draft_model_type == 'neural':
            assert self.gamma > 0
            assert self.draft_model is not None
