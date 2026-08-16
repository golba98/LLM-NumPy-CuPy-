from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    """
    Hyperparameter configuration for TinyLLM training runs.
    """
    batch_size: int = 4
    context_length: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    max_steps: int = 100
    eval_interval: int = 10
    checkpoint_interval: int = 50
    accumulation_steps: int = 1
    micro_batch_size: int = 0
    seed: int = 42
    log_csv: str = "logs/train.csv"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    cosine_decay_steps: int = 0
    checkpoint_dir: str = ""
    tokenizer_path: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_sha256: str = ""
    eval_max_batches: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    loss_scale: float = 1.0
    min_gradient_norm: float = 0.0
    max_consecutive_small_gradient_steps: int = 0
    target_tokens: int = 0
    progress: bool = False
    run_name: str = "training"

    def __post_init__(self):
        if self.batch_size <= 0 or self.context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")
        if self.learning_rate <= 0 or self.adam_eps <= 0:
            raise ValueError("learning_rate and adam_eps must be positive")
        if not 0 <= self.weight_decay:
            raise ValueError("weight_decay must be non-negative")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam betas must be between zero and one")
        if self.max_grad_norm <= 0 or self.max_steps <= 0:
            raise ValueError("max_grad_norm and max_steps must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if self.micro_batch_size < 0:
            raise ValueError("micro_batch_size must be non-negative")
        if self.eval_max_batches < 0:
            raise ValueError("eval_max_batches must be non-negative")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")
        if self.dtype not in {"float16", "float32", "float64"}:
            raise ValueError("dtype must be float16, float32, or float64")
        if self.loss_scale < 1.0:
            raise ValueError("loss_scale must be >= 1")
        if self.min_gradient_norm < 0:
            raise ValueError("min_gradient_norm must be non-negative")
        if self.max_consecutive_small_gradient_steps < 0:
            raise ValueError("max_consecutive_small_gradient_steps must be non-negative")
        if self.target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")

    @property
    def effective_batch_size(self) -> int:
        return (self.micro_batch_size or self.batch_size) * self.accumulation_steps

    def to_dict(self):
        return asdict(self)
