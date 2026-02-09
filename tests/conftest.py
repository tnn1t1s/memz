"""Shared fixtures for memz tests."""

import pytest

from memz.config import (
    BatchConfig,
    EvaluationConfig,
    LoRAConfig,
    ServiceConfig,
    TrainingConfig,
)


@pytest.fixture
def default_config():
    """Return a ServiceConfig with defaults."""
    return ServiceConfig()


@pytest.fixture
def minimal_config():
    """Return a config suitable for fast tests (few steps, small batch)."""
    return ServiceConfig(
        base_model="sshleifer/tiny-gpt2",
        backend="cpu",
        update_mode="immediate",
        lora=LoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
        training=TrainingConfig(
            batch_size=1,
            grad_accum=1,
            lr=1e-3,
            max_steps=2,
            eval_every=1,
            max_seq_length=32,
            warmup_steps=0,
            optimizer="adamw",
        ),
        batch=BatchConfig(max_examples=4, max_age_minutes=1),
        evaluation=EvaluationConfig(
            anchor_provider="none",
            rollback_mode="off",
        ),
    )


@pytest.fixture
def sample_examples():
    """Return a small list of training examples."""
    return [
        {"input": "What is Python?", "output": "A programming language."},
        {"input": "What is 2+2?", "output": "4"},
        {"input": "Capital of France?", "output": "Paris"},
    ]


@pytest.fixture
def sample_yaml_content():
    """Return sample YAML config content."""
    return """\
backend: cpu
base_model: sshleifer/tiny-gpt2
update_mode: immediate

lora:
  r: 4
  alpha: 8
  dropout: 0.0
  target_modules: [c_attn]

training:
  batch_size: 1
  grad_accum: 1
  lr: 0.001
  max_steps: 2
  eval_every: 1
  max_seq_length: 32
  warmup_steps: 0
  optimizer: adamw

batch:
  max_examples: 4
  max_age_minutes: 1

evaluation:
  anchor_provider: none
  rollback_mode: "off"
  catastrophic_delta: 0.25
"""
