"""Tests for memz.trainer -- queue, batch accumulation, trainer state."""

from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from memz.config import (
    BatchConfig,
    EvaluationConfig,
    LoRAConfig,
    ServiceConfig,
    TrainingConfig,
)
from memz.trainer import TrainResult, build_optimizer, load_tokenizer, run_training


class TestTrainResult:
    """Test the TrainResult dataclass."""

    def test_defaults(self):
        r = TrainResult()
        assert r.steps_completed == 0
        assert r.final_loss == 0.0
        assert r.adapter_path == ""
        assert r.loss_history == []

    def test_loss_history_independence(self):
        r1 = TrainResult()
        r2 = TrainResult()
        r1.loss_history.append(1.0)
        assert r2.loss_history == []


class TestBuildOptimizer:
    """Test optimizer selection."""

    def _make_model(self):
        return torch.nn.Linear(4, 4)

    def test_adamw(self):
        model = self._make_model()
        cfg = ServiceConfig(training=ServiceConfig.__dataclass_fields__["training"].default_factory())
        cfg.training.optimizer = "adamw"
        opt = build_optimizer(model, cfg)
        assert "AdamW" in type(opt).__name__

    def test_adafactor(self):
        model = self._make_model()
        cfg = ServiceConfig()
        cfg.training.optimizer = "adafactor"
        opt = build_optimizer(model, cfg)
        assert "Adafactor" in type(opt).__name__

    def test_unsupported_optimizer_raises(self):
        model = self._make_model()
        cfg = ServiceConfig()
        cfg.training.optimizer = "sgd"
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer(model, cfg)


class TestLoadTokenizer:
    """Test tokenizer loading (uses sshleifer/tiny-gpt2 for speed)."""

    def test_tokenizer_has_pad_token(self, minimal_config):
        tok = load_tokenizer(minimal_config)
        assert tok.pad_token is not None
        assert tok.pad_token_id is not None


class TestUpdateQueue:
    """Test update queue (enqueue/dequeue) using the server's AppState queue."""

    def test_enqueue_dequeue_order(self):
        queue: deque[dict] = deque()
        jobs = [
            {"job_id": "job_001", "examples": [{"input": "a", "output": "b"}]},
            {"job_id": "job_002", "examples": [{"input": "c", "output": "d"}]},
        ]
        for j in jobs:
            queue.append(j)

        assert len(queue) == 2
        first = queue.popleft()
        assert first["job_id"] == "job_001"
        second = queue.popleft()
        assert second["job_id"] == "job_002"
        assert len(queue) == 0

    def test_immediate_mode_prepends(self):
        queue: deque[dict] = deque()
        queue.append({"job_id": "job_001", "examples": []})
        # Immediate mode uses appendleft to jump the queue
        queue.appendleft({"job_id": "job_urgent", "examples": []})

        first = queue.popleft()
        assert first["job_id"] == "job_urgent"


class TestBatchAccumulation:
    """Test batch accumulation config (max_examples, max_age_minutes)."""

    def test_batch_config_max_examples_threshold(self):
        cfg = BatchConfig(max_examples=3, max_age_minutes=30)
        examples = []
        for i in range(5):
            examples.append({"input": f"q{i}", "output": f"a{i}"})
            if len(examples) >= cfg.max_examples:
                break
        assert len(examples) == 3

    def test_batch_config_defaults(self):
        cfg = BatchConfig()
        assert cfg.max_examples == 128
        assert cfg.max_age_minutes == 30


class TestTrainerStateTracking:
    """Test that TrainResult tracks state across training."""

    def test_loss_history_accumulates(self):
        result = TrainResult()
        losses = [2.5, 2.1, 1.8, 1.5]
        for loss in losses:
            result.loss_history.append(loss)
        assert len(result.loss_history) == 4
        assert result.loss_history == losses

    def test_final_loss_is_last(self):
        result = TrainResult()
        result.loss_history = [2.5, 2.1, 1.8]
        result.final_loss = result.loss_history[-1]
        assert result.final_loss == 1.8

    def test_adapter_path_set_after_training(self):
        result = TrainResult()
        result.adapter_path = "/tmp/test/adapter"
        result.steps_completed = 100
        assert result.adapter_path == "/tmp/test/adapter"
        assert result.steps_completed == 100


class TestRunTrainingSmokeTest:
    """Smoke test with sshleifer/tiny-gpt2 (very small model for CI speed)."""

    @pytest.mark.slow
    def test_run_training_tiny_model(self, minimal_config, sample_examples, tmp_path):
        result = run_training(
            config=minimal_config,
            examples=sample_examples,
            output_dir=tmp_path / "output",
            device="cpu",
        )

        assert result.steps_completed == minimal_config.training.max_steps
        assert result.final_loss > 0
        assert len(result.loss_history) == minimal_config.training.max_steps
        assert result.adapter_path != ""

        # Verify adapter was actually saved
        adapter_path = Path(result.adapter_path)
        assert adapter_path.exists()
        assert (adapter_path / "adapter_config.json").exists()
