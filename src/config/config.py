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

@dataclass
class DistillConfig:
    teacher_model: str
    student_model: str
    language_code: str

    distill_mode: Literal["task_specific", "general"] = "task_specific"

    # SeqKD dataset — HF dataset ID or local path with teacher translations
    seqkd_data_path: str | None = None
    max_samples: int = 5000

    # Training
    max_steps: int = 3000
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    max_length: int = 512
    eval_split_ratio: float = 0.05
    eval_every: int = 200

    # Checkpointing & output
    hf_repo_id: str | None = None
    output_dir: str = "../distilled_models"
    resume_from: str | None = None
    log_every: int = 50

    device: str = "auto"

    def __post_init__(self):
        if self.seqkd_data_path == "None":
            self.seqkd_data_path = None
        if self.hf_repo_id == "None":
            self.hf_repo_id = None
        if self.resume_from == "None":
            self.resume_from = None
