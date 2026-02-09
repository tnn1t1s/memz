# memz

A continuous PEFT (Parameter-Efficient Fine-Tuning) service that runs locally and adapts a frozen base model via LoRA updates. It exposes a simple API for searching with the latest fine-tuned checkpoint, submitting new training examples, and inspecting loss curves and forgetting metrics. Designed for research and experimentation on a Mac Mini (MPS backend), with optional evaluation, anchoring, and rollback policies.

## Quickstart

```bash
# Clone and install
git clone <repo-url> && cd memz
pip install -e .

# Edit config.yaml to set your base model
# (default: mistralai/Mistral-7B-Instruct-v0.2)

# Start the service
memz serve
```

The server starts on `http://127.0.0.1:8000` by default. Override with `--host` and `--port`:

```bash
memz serve --host 0.0.0.0 --port 9000 --config path/to/config.yaml
```

## API Reference

### POST /search

Generate a response using the latest stable LoRA checkpoint.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is LoRA fine-tuning?",
    "max_tokens": 256,
    "temperature": 0.2
  }'
```

Response:

```json
{
  "response": "LoRA (Low-Rank Adaptation) is ...",
  "checkpoint": "A_2026-02-09_0012",
  "checkpoint_status": "stable",
  "latency_ms": 823
}
```

### POST /update

Submit training examples to enqueue a LoRA fine-tuning batch.

```bash
curl -X POST http://127.0.0.1:8000/update \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"input": "Explain gradient descent.", "output": "Gradient descent is ..."}
    ],
    "tags": ["user:alice"]
  }'
```

Response:

```json
{
  "job_id": "job_000128",
  "queued": true
}
```

### GET /info

Return service status, loss curves, and forgetting metrics.

```bash
curl http://127.0.0.1:8000/info
```

Response:

```json
{
  "base_model": "mistral-7b",
  "current_checkpoint": "A_2026-02-09_0012",
  "queue_depth": 3,
  "last_update": "2026-02-09T21:12:55Z",
  "loss_curves": {
    "finetune_loss": [],
    "forgetting_loss": [],
    "eval_loss": []
  },
  "forgetting_summary": {
    "anchor_ce_delta": 0.18,
    "retention_score": 0.92
  },
  "hardware": {
    "device": "mac-mini",
    "backend": "mps"
  }
}
```

If anchors are disabled (`anchor_provider: none`), `forgetting_summary` and `forgetting_loss` will be `null`.

## Configuration

All settings live in `config.yaml`. Key parameters:

```yaml
backend: mps                          # mps | cuda | cpu
base_model: mistralai/Mistral-7B-Instruct-v0.2

update_mode: batched                  # immediate | batched

lora:
  r: 8                               # LoRA rank
  alpha: 16                           # LoRA alpha
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  batch_size: 1
  grad_accum: 16                      # gradient accumulation steps
  lr: 1e-4
  max_steps: 200
  eval_every: 1                       # evaluate every N steps

batch:
  max_examples: 128                   # trigger training after N examples
  max_age_minutes: 30                 # or after N minutes

evaluation:
  anchor_provider: hf                 # hf | local | none
  anchor_dataset: tnn1t1s/news-100-sept-2023
  anchor_sample: 1000
  rollback_mode: "off"                # off | warn | auto_rollback
  catastrophic_delta: 0.25
```

**Update modes**: `immediate` triggers a LoRA run on every `/update` call. `batched` accumulates examples until `max_examples` or `max_age_minutes` is reached.

**Rollback modes**: `off` logs metrics only. `warn` marks risky checkpoints without reverting. `auto_rollback` reverts to the previous checkpoint if `forgetting_loss_delta` exceeds `catastrophic_delta`.

## Architecture

```
+------------------+       +------------------+
|   API Server     | ----> |  Trainer Worker  |
|  (FastAPI)       |       |  (LoRA updates)  |
|  /search         |       +--------+---------+
|  /update         |                |
|  /info           |                v
+------------------+       +------------------+
                           | Checkpoint Mgr   |
                           | (adapter lineage) |
                           +--------+---------+
                                    |
                                    v
                           +------------------+
                           |   Evaluator      |
                           | (loss, anchors)  |
                           +------------------+
```

**API Server** handles HTTP requests and routes to internal components.

**Trainer Worker** consumes queued update batches and runs LoRA fine-tuning against the frozen base model, producing a new adapter checkpoint per batch.

**Checkpoint Manager** maintains adapter lineage, exposes the latest stable adapter, and optionally rolls back based on configured policy.

**Evaluator** computes fine-tune loss and optional forgetting metrics (anchor CE delta, retention score) per checkpoint.

## Development

### Project layout

```
memz/
  config.yaml              # service configuration
  pyproject.toml           # package metadata and dependencies
  src/memz/
    cli.py                 # CLI entry point (typer)
    server.py              # FastAPI application
    config.py              # configuration dataclasses and loader
    trainer.py             # LoRA training worker
    evaluator.py           # evaluation and forgetting metrics
    checkpoint.py          # checkpoint manager and rollback
    data.py                # data loading and encoding
    device.py              # device detection (MPS/CUDA/CPU)
  tests/
```

### Running tests

```bash
pip install -e ".[dev]"
pytest
```

### Requirements

- Python 3.11+
- PyTorch 2.0+ with MPS support (macOS) or CUDA
- See `pyproject.toml` for the full dependency list
