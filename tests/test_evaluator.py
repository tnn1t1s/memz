"""Tests for memz.evaluator -- metric computation shapes, evaluate_checkpoint."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from memz.config import EvaluationConfig, ServiceConfig, TrainingConfig
from memz.evaluator import (
    cache_base_outputs,
    evaluate_checkpoint,
    finetune_loss,
    forgetting_loss,
    forgetting_loss_from_cache,
)


class _FakeModelOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class _FakeModel:
    """Minimal model stand-in that returns random logits with correct shapes."""

    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size
        self._training = False

    def eval(self):
        self._training = False
        return self

    def train(self, mode=True):
        self._training = mode
        return self

    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.randn(batch, seq, self.vocab_size)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return _FakeModelOutput(logits=logits, loss=loss)


def _make_fake_model(vocab_size=32):
    return _FakeModel(vocab_size=vocab_size)


def _make_fake_batch(batch_size=2, seq_len=8, vocab_size=32):
    """Create a fake tokenized batch."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TestCacheBaseOutputs:
    """Test that cache_base_outputs returns correctly shaped data."""

    def test_output_shape(self):
        model = _make_fake_model()
        batches = [_make_fake_batch() for _ in range(3)]
        loader = batches  # just a list, iterable

        cached = cache_base_outputs(model, loader, max_batches=2, device="cpu")
        assert len(cached) == 2
        assert "input_ids" in cached[0]
        assert "attention_mask" in cached[0]
        assert "log_probs_base" in cached[0]
        # All on CPU
        assert cached[0]["log_probs_base"].device.type == "cpu"

    def test_empty_dataloader(self):
        model = _make_fake_model()
        cached = cache_base_outputs(model, [], device="cpu")
        assert cached == []


class TestForgettingLossFromCache:
    """Test forgetting loss from cached base outputs."""

    def test_returns_expected_keys(self):
        model = _make_fake_model()
        cached = [{
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "log_probs_base": torch.randn(2, 8, 32),
        }]
        result = forgetting_loss_from_cache(model, cached, device="cpu")
        assert "forgetting_loss" in result
        assert "perplexity_finetuned" in result
        assert "n_tokens" in result
        assert result["n_tokens"] > 0

    def test_empty_cache_returns_zero(self):
        model = _make_fake_model()
        result = forgetting_loss_from_cache(model, [], device="cpu")
        assert result["forgetting_loss"] == 0.0
        assert result["n_tokens"] == 0


class TestForgettingLoss:
    """Test dual-model forgetting loss."""

    def test_returns_expected_keys(self):
        ft_model = _make_fake_model()
        base_model = _make_fake_model()
        batches = [_make_fake_batch()]
        result = forgetting_loss(ft_model, base_model, batches, device="cpu")
        assert "forgetting_loss" in result
        assert "perplexity_finetuned" in result
        assert "perplexity_base" in result
        assert "n_tokens" in result

    def test_max_batches_limits(self):
        ft_model = _make_fake_model()
        base_model = _make_fake_model()
        batches = [_make_fake_batch() for _ in range(5)]
        result = forgetting_loss(ft_model, base_model, batches, max_batches=1, device="cpu")
        # Should only process 1 batch; n_tokens = batch_size * (seq_len - 1)
        assert result["n_tokens"] <= 2 * 7  # batch_size=2, seq_len-1=7

    def test_empty_dataloader(self):
        ft_model = _make_fake_model()
        base_model = _make_fake_model()
        result = forgetting_loss(ft_model, base_model, [], device="cpu")
        assert result["n_tokens"] == 0
        assert result["forgetting_loss"] == 0.0


class TestFinetuneLoss:
    """Test finetune loss computation."""

    def test_returns_expected_keys(self):
        model = _make_fake_model()
        batches = [_make_fake_batch()]
        result = finetune_loss(model, batches, device="cpu")
        assert "finetune_loss" in result
        assert "n_tokens" in result
        assert result["n_tokens"] > 0
        assert isinstance(result["finetune_loss"], float)

    def test_empty_dataloader(self):
        model = _make_fake_model()
        result = finetune_loss(model, [], device="cpu")
        assert result["finetune_loss"] == 0.0
        assert result["n_tokens"] == 0


class TestEvaluateCheckpoint:
    """Test the high-level evaluate_checkpoint function."""

    def test_anchors_none_skips_forgetting(self):
        cfg = ServiceConfig(
            evaluation=EvaluationConfig(anchor_provider="none"),
        )
        model = _make_fake_model()
        metrics = evaluate_checkpoint(cfg, model, device="cpu")
        assert "forgetting_loss" not in metrics

    def test_with_cached_base_outputs(self):
        cfg = ServiceConfig(
            evaluation=EvaluationConfig(anchor_provider="hf"),
        )
        model = _make_fake_model()
        cached = [{
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "log_probs_base": torch.randn(2, 8, 32),
        }]
        metrics = evaluate_checkpoint(
            cfg, model, cached_base_outputs=cached, device="cpu"
        )
        assert "forgetting_loss" in metrics
        assert "perplexity_finetuned" in metrics

    def test_with_train_dataloader(self):
        cfg = ServiceConfig(
            evaluation=EvaluationConfig(anchor_provider="none"),
        )
        model = _make_fake_model()
        train_batches = [_make_fake_batch()]
        metrics = evaluate_checkpoint(
            cfg, model, train_dataloader=train_batches, device="cpu"
        )
        assert "finetune_loss" in metrics

    def test_dual_model_requires_tokenizer(self):
        cfg = ServiceConfig(
            evaluation=EvaluationConfig(anchor_provider="hf"),
        )
        ft_model = _make_fake_model()
        base_model = _make_fake_model()
        with pytest.raises(ValueError, match="tokenizer required"):
            evaluate_checkpoint(
                cfg, ft_model, base_model=base_model, device="cpu"
            )


class TestAnchorLoading:
    """Test anchor data loading via evaluate_checkpoint with mocked HF datasets."""

    def test_evaluate_checkpoint_with_base_model_and_tokenizer(self):
        from datasets import Dataset
        from memz.data import load_anchor_data

        fake_dataset = Dataset.from_list([
            {"text": "anchor text " * 50},
        ])

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", 8)
                torch.manual_seed(42)
                n = min(30, max_length * 3)
                ids = torch.randint(1, 100, (n,))
                return {"input_ids": ids.unsqueeze(0), "attention_mask": ids.new_ones(1, n)}

        cfg = ServiceConfig(
            evaluation=EvaluationConfig(
                anchor_provider="hf",
                anchor_dataset="fake/dataset",
                anchor_sample=2,
            ),
            training=TrainingConfig(max_seq_length=8, batch_size=2),
        )

        ft_model = _make_fake_model(vocab_size=100)
        base_model = _make_fake_model(vocab_size=100)

        with patch("memz.data.load_dataset", return_value=fake_dataset):
            metrics = evaluate_checkpoint(
                cfg, ft_model,
                base_model=base_model,
                tokenizer=FakeTokenizer(),
                device="cpu",
            )

        assert "forgetting_loss" in metrics
        assert "perplexity_finetuned" in metrics
        assert "perplexity_base" in metrics
        assert isinstance(metrics["forgetting_loss"], float)


class TestMetricsSerialization:
    """Test that metric dicts are JSON-serializable (no tensors, no special types)."""

    def test_forgetting_loss_json_safe(self):
        import json
        model = _make_fake_model()
        cached = [{
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "log_probs_base": torch.randn(2, 8, 32),
        }]
        result = forgetting_loss_from_cache(model, cached, device="cpu")
        # Should not raise
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped["forgetting_loss"] == pytest.approx(result["forgetting_loss"])
        assert roundtripped["n_tokens"] == result["n_tokens"]

    def test_finetune_loss_json_safe(self):
        import json
        model = _make_fake_model()
        batches = [_make_fake_batch()]
        result = finetune_loss(model, batches, device="cpu")
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped["finetune_loss"] == pytest.approx(result["finetune_loss"])

    def test_dual_model_forgetting_json_safe(self):
        import json
        ft_model = _make_fake_model()
        base_model = _make_fake_model()
        batches = [_make_fake_batch()]
        result = forgetting_loss(ft_model, base_model, batches, device="cpu")
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped["forgetting_loss"] == pytest.approx(result["forgetting_loss"])
        assert roundtripped["perplexity_base"] == pytest.approx(result["perplexity_base"])
