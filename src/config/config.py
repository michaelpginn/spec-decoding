from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    task: Literal['translation']
    language_code: str

    target_model: str
    draft_model: str | None
    draft_model_type: Literal["none", "neural", "statistical"]
    decoding_mode: Literal["greedy", "top_k", "top_p"]
    gamma: int = 5
    track_iterations: bool = False

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

    data_source: str = "tatoeba"
    max_samples: int = 5
    max_new_tokens: int = 512
    device: str = "auto"

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if self.draft_model_type != 'none':
            assert self.draft_model is not None
            assert self.gamma > 0


@dataclass
class DistillConfig:
    teacher_model: str
    student_model: str
    language_code: str

    # SeqKD dataset — HF dataset ID or local path with teacher translations
    seqkd_data_path: str | None = None
    max_samples: int = 5000

    # Training
    max_steps: int = 3000
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512

    # Checkpointing & output
    hf_repo_id: str | None = None
    output_dir: str = "../distilled_models"
    resume_from: str | None = None
    save_every: int = 500
    log_every: int = 50

    device: str = "auto"
