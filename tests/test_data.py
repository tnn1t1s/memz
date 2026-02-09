"""Tests for memz.data module."""

from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset

from memz.data import create_dataloader, load_anchor_data, tokenize_examples


class FakeTokenizer:
    """Minimal tokenizer mock for testing without loading real models."""

    def __init__(self, vocab_size=100, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", 16)
        return_tensors = kwargs.get("return_tensors")
        truncation = kwargs.get("truncation", False)
        padding = kwargs.get("padding", False)

        # Produce deterministic token ids from text hash
        torch.manual_seed(hash(text) % 2**31)
        n_real = min(max(4, len(text) // 2), 50)
        if truncation and n_real > max_length:
            n_real = max_length

        ids = torch.randint(1, self.vocab_size, (n_real,))

        if padding == "max_length" and n_real < max_length:
            pad = torch.zeros(max_length - n_real, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
            mask = torch.cat([
                torch.ones(n_real, dtype=torch.long),
                torch.zeros(max_length - n_real, dtype=torch.long),
            ])
        else:
            mask = torch.ones(len(ids), dtype=torch.long)

        if return_tensors == "pt":
            ids = ids.unsqueeze(0)
            mask = mask.unsqueeze(0)

        return {"input_ids": ids, "attention_mask": mask}


# -- tokenize_examples tests --


def test_tokenize_examples_returns_dataset():
    tokenizer = FakeTokenizer()
    examples = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "Hello", "output": "Hi there"},
    ]
    dataset = tokenize_examples(examples, tokenizer, max_length=16)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2


def test_tokenize_examples_has_required_columns():
    tokenizer = FakeTokenizer()
    examples = [{"input": "test", "output": "result"}]
    dataset = tokenize_examples(examples, tokenizer, max_length=16)

    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names


def test_tokenize_examples_labels_mask_padding():
    tokenizer = FakeTokenizer()
    examples = [{"input": "short", "output": "x"}]
    dataset = tokenize_examples(examples, tokenizer, max_length=32)

    row = dataset[0]
    labels = row["labels"]
    attention_mask = row["attention_mask"]

    # Where attention_mask is 0 (padding), labels should be -100
    padding_positions = (attention_mask == 0)
    if padding_positions.any():
        assert (labels[padding_positions] == -100).all()

    # Where attention_mask is 1, labels should NOT be -100
    real_positions = (attention_mask == 1)
    assert (labels[real_positions] != -100).all()


def test_tokenize_examples_respects_max_length():
    tokenizer = FakeTokenizer()
    examples = [{"input": "a" * 200, "output": "b" * 200}]
    max_len = 16
    dataset = tokenize_examples(examples, tokenizer, max_length=max_len)

    row = dataset[0]
    assert row["input_ids"].shape[0] == max_len
    assert row["attention_mask"].shape[0] == max_len
    assert row["labels"].shape[0] == max_len


def test_tokenize_examples_uses_chat_template():
    tokenizer = FakeTokenizer()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

    examples = [{"input": "hello", "output": "world"}]
    tokenize_examples(examples, tokenizer, max_length=16)

    tokenizer.apply_chat_template.assert_called_once()
    call_args = tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_tokenize_examples_fallback_when_chat_template_fails():
    tokenizer = FakeTokenizer()
    tokenizer.apply_chat_template = MagicMock(side_effect=Exception("fail"))

    examples = [{"input": "hello", "output": "world"}]
    # Should not raise, falls back to plain concatenation
    dataset = tokenize_examples(examples, tokenizer, max_length=16)
    assert len(dataset) == 1


# -- load_anchor_data tests --


def test_load_anchor_data_chunks_correctly():
    fake_dataset = Dataset.from_list([
        {"text": "A" * 500},
        {"text": "B" * 500},
    ])

    tokenizer = FakeTokenizer()

    with patch("memz.data.load_dataset", return_value=fake_dataset):
        chunks = load_anchor_data(
            tokenizer,
            dataset_name="fake/dataset",
            max_length=8,
            stride=8,
            max_examples=100,
        )

    assert len(chunks) > 0
    for chunk in chunks:
        assert "input_ids" in chunk
        assert "attention_mask" in chunk
        assert chunk["input_ids"].shape[0] == 8
        assert chunk["attention_mask"].shape[0] == 8
        # attention_mask should be all ones (no padding in anchor chunks)
        assert chunk["attention_mask"].sum() == 8


def test_load_anchor_data_respects_max_examples():
    fake_dataset = Dataset.from_list([
        {"text": "word " * 1000},
    ])

    tokenizer = FakeTokenizer()

    with patch("memz.data.load_dataset", return_value=fake_dataset):
        chunks = load_anchor_data(
            tokenizer,
            dataset_name="fake/dataset",
            max_length=8,
            stride=8,
            max_examples=3,
        )

    assert len(chunks) == 3


def test_load_anchor_data_skips_empty_text():
    fake_dataset = Dataset.from_list([
        {"text": ""},
        {"text": "   "},
        {"text": "valid text " * 50},
    ])

    tokenizer = FakeTokenizer()

    with patch("memz.data.load_dataset", return_value=fake_dataset):
        chunks = load_anchor_data(
            tokenizer,
            dataset_name="fake/dataset",
            max_length=8,
            stride=8,
            max_examples=100,
        )

    # Should have chunks only from the third row
    assert len(chunks) > 0


# -- create_dataloader tests --


def test_create_dataloader_batches():
    items = [
        {
            "input_ids": torch.randint(0, 50, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
        }
        for _ in range(10)
    ]
    loader = create_dataloader(items, batch_size=4, shuffle=False)

    batches = list(loader)
    assert len(batches) == 3  # 10 items / 4 = 2 full + 1 partial
    assert batches[0]["input_ids"].shape == (4, 8)
    assert batches[0]["attention_mask"].shape == (4, 8)
    assert batches[2]["input_ids"].shape == (2, 8)


def test_create_dataloader_includes_labels():
    items = [
        {
            "input_ids": torch.randint(0, 50, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "labels": torch.randint(0, 50, (8,)),
        }
        for _ in range(4)
    ]
    loader = create_dataloader(items, batch_size=2)

    batch = next(iter(loader))
    assert "labels" in batch
    assert batch["labels"].shape == (2, 8)


def test_create_dataloader_no_labels():
    items = [
        {
            "input_ids": torch.randint(0, 50, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
        }
        for _ in range(4)
    ]
    loader = create_dataloader(items, batch_size=2)

    batch = next(iter(loader))
    assert "labels" not in batch
