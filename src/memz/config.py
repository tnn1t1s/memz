"""Configuration for the continuous PEFT service."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_rslora: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 1e-4
    max_steps: int = 200
    eval_every: int = 1
    max_seq_length: int = 512
    warmup_steps: int = 10
    optimizer: str = "adafactor"


@dataclass
class BatchConfig:
    max_examples: int = 128
    max_age_minutes: int = 30


@dataclass
class EvaluationConfig:
    anchor_provider: str = "hf"  # hf | local | none
    anchor_dataset: str = "tnn1t1s/news-100-sept-2023"
    anchor_split: str = "train"
    anchor_sample: int = 1000
    rollback_mode: str = "off"  # off | warn | auto_rollback
    catastrophic_delta: float = 0.25


@dataclass
class ServiceConfig:
    backend: str = "mps"
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    update_mode: str = "batched"  # immediate | batched
    data_dir: str = "."
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> ServiceConfig:
    """Load ServiceConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return ServiceConfig()

    lora = LoRAConfig(**raw.pop("lora", {}))
    training = TrainingConfig(**raw.pop("training", {}))
    batch = BatchConfig(**raw.pop("batch", {}))
    evaluation = EvaluationConfig(**raw.pop("evaluation", {}))

    return ServiceConfig(
        lora=lora,
        training=training,
        batch=batch,
        evaluation=evaluation,
        **raw,
    )
