"""Evaluator for forgetting metrics and anchor loss.

Key: CE(finetuned || base) not CE(finetuned || ground_truth),
following Kalajdzievski (2024).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from memz.config import ServiceConfig
from memz.data import create_dataloader, load_anchor_data
from memz.device import get_device

logger = logging.getLogger(__name__)


def cache_base_outputs(
    model,
    dataloader: DataLoader,
    max_batches: int | None = None,
    device: str | None = None,
) -> list[dict]:
    """Cache base model log-probs for memory-efficient forgetting computation.

    Stores log-probs on CPU to save GPU/MPS memory so the base model
    can be unloaded before loading the fine-tuned model.
    """
    device = device or get_device()
    model.eval()
    cached = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            log_probs = F.log_softmax(logits, dim=-1).cpu()
            cached.append({
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "log_probs_base": log_probs,
            })

    return cached


def forgetting_loss_from_cache(
    model,
    cached_outputs: list[dict],
    device: str | None = None,
) -> dict:
    """Compute forgetting loss using cached base model outputs.

    This is the low-memory variant: base model outputs were cached
    and stored on CPU, so only the fine-tuned model needs to be loaded.
    """
    device = device or get_device()
    model.eval()

    total_ce_forgetting = 0.0
    total_ce_finetuned = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in cached_outputs:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            log_probs_base = batch["log_probs_base"].to(device)

            logits_ft = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # Shift for next-token prediction
            shift_logits_ft = logits_ft[..., :-1, :].contiguous()
            shift_log_probs_base = log_probs_base[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            vocab_size = shift_logits_ft.size(-1)
            flat_logits_ft = shift_logits_ft.view(-1, vocab_size)
            flat_log_probs_base = shift_log_probs_base.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1).bool()

            valid_logits_ft = flat_logits_ft[flat_mask]
            valid_log_probs_base = flat_log_probs_base[flat_mask]
            valid_labels = flat_labels[flat_mask]

            n_tokens = valid_labels.numel()
            if n_tokens == 0:
                continue

            # CE(finetuned || base) using cached base log-probs
            valid_probs_base = valid_log_probs_base.exp()
            valid_log_probs_ft = F.log_softmax(valid_logits_ft, dim=-1)
            ce_forgetting = -(valid_probs_base * valid_log_probs_ft).sum(dim=-1).sum()

            ce_finetuned = F.cross_entropy(
                valid_logits_ft, valid_labels, reduction="sum"
            )

            total_ce_forgetting += ce_forgetting.item()
            total_ce_finetuned += ce_finetuned.item()
            total_tokens += n_tokens

    if total_tokens == 0:
        return {"forgetting_loss": 0.0, "perplexity_finetuned": 0.0, "n_tokens": 0}

    return {
        "forgetting_loss": total_ce_forgetting / total_tokens,
        "perplexity_finetuned": torch.exp(
            torch.tensor(total_ce_finetuned / total_tokens)
        ).item(),
        "n_tokens": total_tokens,
    }


def forgetting_loss(
    finetuned_model,
    base_model,
    dataloader: DataLoader,
    max_batches: int | None = None,
    device: str | None = None,
) -> dict:
    """Compute forgetting loss: CE(p_finetuned || p_base).

    Standard variant where both models are in memory.
    """
    device = device or get_device()
    finetuned_model.eval()
    base_model.eval()

    total_ce_forgetting = 0.0
    total_ce_finetuned = 0.0
    total_ce_base = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_ft = finetuned_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_base = base_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # Shift for next-token prediction
            shift_logits_ft = logits_ft[..., :-1, :].contiguous()
            shift_logits_base = logits_base[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            vocab_size = shift_logits_ft.size(-1)
            flat_logits_ft = shift_logits_ft.view(-1, vocab_size)
            flat_logits_base = shift_logits_base.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1).bool()

            valid_logits_ft = flat_logits_ft[flat_mask]
            valid_logits_base = flat_logits_base[flat_mask]
            valid_labels = flat_labels[flat_mask]

            n_tokens = valid_labels.numel()
            if n_tokens == 0:
                continue

            # CE(finetuned || base)
            valid_probs_base = F.softmax(valid_logits_base, dim=-1)
            valid_log_probs_ft = F.log_softmax(valid_logits_ft, dim=-1)
            ce_forgetting = -(valid_probs_base * valid_log_probs_ft).sum(dim=-1).sum()

            ce_finetuned = F.cross_entropy(
                valid_logits_ft, valid_labels, reduction="sum"
            )
            ce_base = F.cross_entropy(
                valid_logits_base, valid_labels, reduction="sum"
            )

            total_ce_forgetting += ce_forgetting.item()
            total_ce_finetuned += ce_finetuned.item()
            total_ce_base += ce_base.item()
            total_tokens += n_tokens

    if total_tokens == 0:
        return {
            "forgetting_loss": 0.0,
            "perplexity_finetuned": 0.0,
            "perplexity_base": 0.0,
            "n_tokens": 0,
        }

    return {
        "forgetting_loss": total_ce_forgetting / total_tokens,
        "perplexity_finetuned": torch.exp(
            torch.tensor(total_ce_finetuned / total_tokens)
        ).item(),
        "perplexity_base": torch.exp(
            torch.tensor(total_ce_base / total_tokens)
        ).item(),
        "n_tokens": total_tokens,
    }


def finetune_loss(
    model,
    dataloader: DataLoader,
    max_batches: int | None = None,
    device: str | None = None,
) -> dict:
    """Compute fine-tuning loss on training data."""
    device = device or get_device()
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    return {
        "finetune_loss": total_loss / total_tokens if total_tokens > 0 else 0.0,
        "n_tokens": total_tokens,
    }


def evaluate_checkpoint(
    config: ServiceConfig,
    finetuned_model,
    base_model=None,
    cached_base_outputs: list[dict] | None = None,
    train_dataloader: DataLoader | None = None,
    tokenizer=None,
    device: str | None = None,
) -> dict:
    """Run full evaluation for a checkpoint.

    Computes forgetting loss (if anchors enabled) and finetune loss.
    Supports both standard mode (two models) and low-memory mode (cached outputs).

    Returns a metrics dict suitable for saving alongside the checkpoint.
    """
    device = device or get_device()
    metrics: dict = {}

    # Forgetting metrics (optional, depends on anchor config)
    if config.evaluation.anchor_provider != "none":
        if cached_base_outputs is not None:
            fg = forgetting_loss_from_cache(finetuned_model, cached_base_outputs, device)
        elif base_model is not None:
            if tokenizer is None:
                raise ValueError("tokenizer required when using dual-model forgetting")
            anchor_chunks = load_anchor_data(
                tokenizer,
                dataset_name=config.evaluation.anchor_dataset,
                split=config.evaluation.anchor_split,
                max_examples=config.evaluation.anchor_sample,
                max_length=config.training.max_seq_length,
            )
            anchor_loader = create_dataloader(anchor_chunks, config.training.batch_size)
            fg = forgetting_loss(finetuned_model, base_model, anchor_loader, device=device)
        else:
            fg = None

        if fg is not None:
            metrics["forgetting_loss"] = fg["forgetting_loss"]
            metrics["perplexity_finetuned"] = fg["perplexity_finetuned"]
            if "perplexity_base" in fg:
                metrics["perplexity_base"] = fg["perplexity_base"]

    # Finetune loss
    if train_dataloader is not None:
        ft = finetune_loss(finetuned_model, train_dataloader, device=device)
        metrics["finetune_loss"] = ft["finetune_loss"]

    return metrics
