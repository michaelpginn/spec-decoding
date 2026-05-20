from dataclasses import dataclass
from typing import Literal

WANDB_PROJECT = "speculative decoding v2"
WANDB_ENTITY = "lecs-general"


@dataclass
class ExperimentConfig:
    task: Literal['translation', 'story_gen']
    language_code: str

    target_model: str
    draft_model: str | None
    draft_model_type: Literal["none", "neural", "ngram"]
    decoding_mode: Literal["greedy", "sample"]
    top_k: int = 0
    top_p: float = 0.0

    repetition_penalty: float = 1.1
    repetition_penalty_window: int = 16

    gamma: int = 5
    track_iterations: bool = False # If true, will log per-iteration of SD

    ngram_n: int = 3

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

    data_source: str = "tatoeba"
    max_samples: int = 6000
    max_samples_mono: int = 20000
    max_new_tokens: int = 128
    device: str = "auto"

    wandb_tag: str | None = None
    wandb_project: str = WANDB_PROJECT

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if self.draft_model_type == 'neural':
            assert self.gamma > 0
            assert self.draft_model is not None

@dataclass
class DistillConfig:
    task: Literal['general', 'translation']
    teacher_model: str
    student_model: str
    language_code: str

    # SeqKD dataset — HF dataset ID or local path with teacher logits, created with generate_
    dataset_path: str | None = None
    max_samples: int = 5000
    top_k: int = 20 # How many teacher logits to keep per token

    # Training
    max_steps: int = 3000
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    max_length: int = 128
    eval_split_ratio: float = 0.05
    eval_every: int = 50

    # Checkpointing & output
    hf_repo_id: str | None = None
    output_dir: str = "../distilled_models"
    resume_from: str | None = None
    log_every: int = 5

    device: str = "auto"
    wandb_project: str = "spec-dec-distill"

    def __post_init__(self):
        if self.dataset_path == "None":
            self.dataset_path = None
        if self.hf_repo_id == "None":
            self.hf_repo_id = None
        if self.resume_from == "None":
            self.resume_from = None
