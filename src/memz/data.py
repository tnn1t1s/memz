"""Data loading and encoding for training examples."""

from __future__ import annotations

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


def tokenize_examples(
    examples: list[dict],
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Tokenize input/output example pairs for causal LM training.

    Each example should have "input" and "output" keys.
    Labels use -100 for padding tokens.
    """
    texts = []
    for ex in examples:
        inp = ex.get("input", "")
        out = ex.get("output", "")
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out},
            ]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                text = f"{inp}\n{out}"
        else:
            text = f"{inp}\n{out}"
        texts.append(text)

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[tokens["attention_mask"] == 0] = -100
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    dataset = Dataset.from_list([{"text": t} for t in texts])
    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


def load_anchor_data(
    tokenizer,
    dataset_name: str = "tnn1t1s/news-100-sept-2023",
    split: str = "train",
    text_field: str = "text",
    max_length: int = 512,
    stride: int = 512,
    max_examples: int | None = 1000,
) -> list[dict]:
    """Load anchor dataset from HF for forgetting evaluation.

    Returns a list of dicts with input_ids and attention_mask tensors,
    windowed into fixed-length chunks (no cross-document leakage).
    """
    dataset = load_dataset(dataset_name, split=split)

    chunks = []
    for row in dataset:
        text = (row.get(text_field) or "").strip()
        if not text:
            continue
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"][0]
        for i in range(0, len(input_ids) - max_length + 1, stride):
            chunk = input_ids[i : i + max_length]
            chunks.append({
                "input_ids": chunk,
                "attention_mask": chunk.new_ones(chunk.shape),
            })
            if max_examples and len(chunks) >= max_examples:
                break
        if max_examples and len(chunks) >= max_examples:
            break

    return chunks


def create_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader with proper collation for token batches."""

    def collate_fn(batch):
        result = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }
        if "labels" in batch[0]:
            result["labels"] = torch.stack([b["labels"] for b in batch])
        return result

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
