"""Trainer worker for LoRA fine-tuning."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    Adafactor,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from memz.config import ServiceConfig
from memz.data import create_dataloader, tokenize_examples
from memz.device import get_device, get_dtype

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a training run."""

    steps_completed: int = 0
    final_loss: float = 0.0
    adapter_path: str = ""
    loss_history: list[float] = field(default_factory=list)


def load_tokenizer(config: ServiceConfig):
    """Load tokenizer for the base model."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: ServiceConfig, device: str | None = None):
    """Load frozen base model."""
    device = device or get_device()
    dtype = get_dtype(device)
    device_map = device if device == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_lora_model(config: ServiceConfig, device: str | None = None, tokenizer=None):
    """Load model with LoRA adapter applied."""
    device = device or get_device()
    dtype = get_dtype(device)

    if tokenizer is None:
        tokenizer = load_tokenizer(config)

    device_map = device if device == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
    )
    model = get_peft_model(model, lora_config)
    logger.info("Trainable parameters:")
    model.print_trainable_parameters()
    return model, tokenizer


def unload_model(model) -> None:
    """Unload model and free memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_optimizer(model, config: ServiceConfig):
    """Build optimizer from config."""
    name = config.training.optimizer.lower()
    if name == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=config.training.lr,
            relative_step=False,
            scale_parameter=False,
        )
    if name == "adamw":
        return AdamW(model.parameters(), lr=config.training.lr)
    raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")


def run_training(
    config: ServiceConfig,
    examples: list[dict],
    output_dir: str | Path,
    device: str | None = None,
) -> TrainResult:
    """Run a LoRA fine-tuning job on the provided examples.

    Args:
        config: Service configuration.
        examples: List of dicts with "input" and "output" keys.
        output_dir: Directory to save the adapter checkpoint.
        device: Device override (auto-detected if None).

    Returns:
        TrainResult with training metrics and adapter path.
    """
    device = device or get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    logger.info("Loading model with LoRA r=%d ...", config.lora.r)
    tokenizer = load_tokenizer(config)
    model, _ = load_lora_model(config, device, tokenizer)

    # Tokenize examples and create dataloader
    train_data = tokenize_examples(
        examples, tokenizer, config.training.max_seq_length
    )
    train_loader = create_dataloader(
        train_data, config.training.batch_size, shuffle=True
    )

    optimizer = build_optimizer(model, config)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )

    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)
    result = TrainResult()

    while step < config.training.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device) if "labels" in batch else input_ids

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        (loss / config.training.grad_accum).backward()

        if (step + 1) % config.training.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1
        loss_val = loss.item()
        result.loss_history.append(loss_val)

        if step % 10 == 0:
            logger.info("Step %d/%d  loss=%.4f", step, config.training.max_steps, loss_val)

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    logger.info("Adapter saved to %s", adapter_path)

    result.steps_completed = step
    result.final_loss = result.loss_history[-1] if result.loss_history else 0.0
    result.adapter_path = str(adapter_path)

    # Cleanup
    unload_model(model)

    return result
