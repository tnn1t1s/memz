"""Tests for memz.config -- loading, defaults, validation."""

import pytest
import yaml

from memz.config import (
    BatchConfig,
    EvaluationConfig,
    LoRAConfig,
    ServiceConfig,
    TrainingConfig,
    load_config,
)


class TestDefaults:
    """Verify all dataclass defaults match the spec."""

    def test_lora_defaults(self):
        c = LoRAConfig()
        assert c.r == 8
        assert c.alpha == 16
        assert c.dropout == 0.05
        assert c.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_training_defaults(self):
        c = TrainingConfig()
        assert c.batch_size == 1
        assert c.grad_accum == 16
        assert c.lr == 1e-4
        assert c.max_steps == 200
        assert c.eval_every == 1
        assert c.max_seq_length == 512
        assert c.warmup_steps == 10
        assert c.optimizer == "adafactor"

    def test_batch_defaults(self):
        c = BatchConfig()
        assert c.max_examples == 128
        assert c.max_age_minutes == 30

    def test_evaluation_defaults(self):
        c = EvaluationConfig()
        assert c.anchor_provider == "hf"
        assert c.anchor_dataset == "tnn1t1s/news-100-sept-2023"
        assert c.anchor_split == "train"
        assert c.anchor_sample == 1000
        assert c.rollback_mode == "off"
        assert c.catastrophic_delta == 0.25

    def test_service_config_defaults(self):
        c = ServiceConfig()
        assert c.backend == "mps"
        assert c.base_model == "mistralai/Mistral-7B-Instruct-v0.2"
        assert c.update_mode == "batched"
        assert isinstance(c.lora, LoRAConfig)
        assert isinstance(c.training, TrainingConfig)
        assert isinstance(c.batch, BatchConfig)
        assert isinstance(c.evaluation, EvaluationConfig)


class TestLoadConfig:
    """Test YAML loading via load_config."""

    def test_load_full_config(self, tmp_path, sample_yaml_content):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(sample_yaml_content)

        cfg = load_config(cfg_path)
        assert cfg.backend == "cpu"
        assert cfg.base_model == "sshleifer/tiny-gpt2"
        assert cfg.update_mode == "immediate"
        assert cfg.lora.r == 4
        assert cfg.lora.alpha == 8
        assert cfg.training.lr == 0.001
        assert cfg.training.optimizer == "adamw"
        assert cfg.batch.max_examples == 4
        assert cfg.evaluation.anchor_provider == "none"

    def test_load_empty_yaml(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")

        cfg = load_config(cfg_path)
        # Should return defaults when YAML is empty
        assert cfg.backend == "mps"
        assert cfg.lora.r == 8

    def test_load_partial_yaml(self, tmp_path):
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text("backend: cuda\nbase_model: gpt2\n")

        cfg = load_config(cfg_path)
        assert cfg.backend == "cuda"
        assert cfg.base_model == "gpt2"
        # Nested configs should use defaults
        assert cfg.lora.r == 8
        assert cfg.training.max_steps == 200

    def test_load_with_nested_overrides(self, tmp_path):
        content = """\
lora:
  r: 16
  alpha: 32
training:
  lr: 0.01
"""
        cfg_path = tmp_path / "nested.yaml"
        cfg_path.write_text(content)

        cfg = load_config(cfg_path)
        assert cfg.lora.r == 16
        assert cfg.lora.alpha == 32
        # Non-overridden nested defaults
        assert cfg.lora.dropout == 0.05
        assert cfg.training.lr == 0.01
        assert cfg.training.batch_size == 1

    def test_load_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")

    def test_load_invalid_field_raises(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("lora:\n  nonexistent_field: 42\n")

        with pytest.raises(TypeError):
            load_config(cfg_path)


class TestConfigValidation:
    """Test config edge cases and invalid values via YAML loading."""

    def test_invalid_rollback_mode_loads_but_stores_value(self, tmp_path):
        cfg_path = tmp_path / "bad_rollback.yaml"
        cfg_path.write_text("evaluation:\n  rollback_mode: invalid_mode\n")
        cfg = load_config(cfg_path)
        # Dataclass accepts any string; validation is at usage time
        assert cfg.evaluation.rollback_mode == "invalid_mode"

    def test_negative_lr_loads_but_stores_value(self, tmp_path):
        cfg_path = tmp_path / "neg_lr.yaml"
        cfg_path.write_text("training:\n  lr: -0.001\n")
        cfg = load_config(cfg_path)
        assert cfg.training.lr == -0.001

    def test_zero_batch_size(self, tmp_path):
        cfg_path = tmp_path / "zero_batch.yaml"
        cfg_path.write_text("training:\n  batch_size: 0\n")
        cfg = load_config(cfg_path)
        assert cfg.training.batch_size == 0

    def test_unknown_top_level_field_raises(self, tmp_path):
        cfg_path = tmp_path / "bad_top.yaml"
        cfg_path.write_text("totally_unknown: 42\n")
        with pytest.raises(TypeError):
            load_config(cfg_path)

    def test_lora_use_rslora_from_yaml(self, tmp_path):
        cfg_path = tmp_path / "rslora.yaml"
        cfg_path.write_text("lora:\n  use_rslora: false\n")
        cfg = load_config(cfg_path)
        assert cfg.lora.use_rslora is False

    def test_evaluation_config_all_modes(self):
        for mode in ("off", "warn", "auto_rollback"):
            ec = EvaluationConfig(rollback_mode=mode)
            assert ec.rollback_mode == mode


class TestConfigComposition:
    """Test that nested config composition works correctly."""

    def test_independent_instances(self):
        c1 = ServiceConfig()
        c2 = ServiceConfig()
        c1.lora.r = 32
        assert c2.lora.r == 8  # unaffected

    def test_custom_nested_config(self):
        cfg = ServiceConfig(
            lora=LoRAConfig(r=16),
            training=TrainingConfig(lr=0.01),
        )
        assert cfg.lora.r == 16
        assert cfg.training.lr == 0.01
        # Other fields keep defaults
        assert cfg.batch.max_examples == 128
