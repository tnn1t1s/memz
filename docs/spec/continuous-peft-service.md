# Continuous PEFT Service (MPS) - Specification

## Purpose
Design a local, always-on service that continuously adapts a frozen base model
via LoRA updates and exposes a RAG-like API. The system must be exploratory:
all heuristics (anchors, rollback, update timing) are optional and configurable.

## Goals
- Provide a simple API:
  - `search(prompt)` to answer using the latest checkpoint.
  - `update(examples)` to enqueue LoRA training data.
  - `info()` to return loss curves, forgetting vs base, and status.
- Support MPS backend for Mac mini.
- Keep model choice and forgetting policy optional and configurable.
- Allow immediate (test) and batched (production) update modes.
- Make evaluation and rollback opt-in, not baked-in heuristics.

## Non-Goals
- Full managed serving platform (K8s, autoscaling).
- One fixed model or dataset.
- Hard-coded forgetting thresholds or rollback rules.

---

## System Overview
The service consists of four cooperating components:

1) **API Server** (FastAPI or Flask)
   - Handles `search`, `update`, `info`.
   - Provides status, metrics, and checkpoint metadata.

2) **Trainer Worker**
   - Consumes a queue of update batches.
   - Runs LoRA fine-tuning against the frozen base model.
   - Produces a new adapter checkpoint per batch.

3) **Evaluator**
   - Computes fine-tune loss and optional forgetting metrics.
   - Writes JSON metrics for curves and dashboards.

4) **Checkpoint Manager**
   - Maintains adapter lineage.
   - Exposes latest stable adapter.
   - Optional rollback based on configured policy.

---

## Storage Layout (suggested)
```
service/
  app.py
  trainer.py
  evaluator.py
  config.yaml
  models/
    base/                # frozen base model
    adapters/            # adapter checkpoints
  data/
    raw/                 # raw user updates
    curated/             # cleaned + validated updates
    anchors/             # optional anchor data
  metrics/
  state/                 # current checkpoint pointer, queue status
```

---

## API Specification

### POST /search
**Input**
```json
{ "prompt": "...", "max_tokens": 256, "temperature": 0.2 }
```

**Behavior**
- Default: use latest **stable** adapter checkpoint.
- Optional: `use_training_adapter=true` to allow live weights.

**Output**
```json
{
  "response": "...",
  "checkpoint": "A_2026-02-09_0012",
  "checkpoint_status": "stable",
  "latency_ms": 823
}
```

### POST /update
**Input**
```json
{
  "examples": [{ "input": "...", "output": "..." }],
  "tags": ["user:alice"]
}
```

**Behavior**
- Validate and store examples in `data/curated/`.
- Enqueue a LoRA update batch.

**Output**
```json
{ "job_id": "job_000128", "queued": true }
```

### GET /info
**Output**
```json
{
  "base_model": "mistral-7b",
  "current_checkpoint": "A_2026-02-09_0012",
  "queue_depth": 3,
  "last_update": "2026-02-09T21:12:55Z",
  "loss_curves": {
    "finetune_loss": [ ... ],
    "forgetting_loss": [ ... ],
    "eval_loss": [ ... ]
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

If anchors are disabled, `forgetting_summary` and `forgetting_loss` are `null`.

---

## Update Modes

### Immediate (testing)
- Each `update()` triggers a LoRA run.

### Batched (production)
- Accumulate examples until:
  - `max_examples` reached, or
  - `max_age_minutes` elapsed.

Both modes are controlled by config or CLI flags.

---

## Model and Adapter Lifecycle
1) Base model `M0` is frozen and loaded once.
2) Each update trains a LoRA adapter `A_t`.
3) New adapter checkpoint becomes `A_{t+1}`.
4) `search()` uses the latest stable checkpoint.
5) Optional rollback policy can revert to `A_{t-1}`.

---

## Evaluation and Forgetting (Optional)

### Anchor Set (Optional)
Anchors are used to measure forgetting. Default example:
- HF dataset: `tnn1t1s/news-100-sept-2023` (News 100)

Anchors are optional; if disabled, forgetting metrics are skipped.

### Metrics
- `finetune_loss`: CE on update batch.
- `forgetting_loss`: CE(M0 || M0+A_t) on anchors.
- `drift_loss`: CE(A_{t-1} || A_t) on anchors.

All metrics should be saved per checkpoint.

---

## Rollback Policy (Optional)
Rollback is **off by default**.
Three modes:
- `off`: log metrics only.
- `warn`: mark checkpoint as risky, no rollback.
- `auto_rollback`: revert to previous checkpoint.

Example trigger (tunable):
```
if forgetting_loss_delta > catastrophic_delta:
    rollback()
```

---

## Config Schema (Draft)
```yaml
backend: mps
base_model: mistral-7b-instruct  # optional, set by user

update_mode: batched             # immediate|batched

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  batch_size: 1
  grad_accum: 16
  lr: 1e-4
  max_steps: 200
  eval_every: 1

batch:
  max_examples: 128
  max_age_minutes: 30

evaluation:
  anchor_provider: hf            # hf|local|none
  anchor_dataset: tnn1t1s/news-100-sept-2023
  anchor_split: train
  anchor_sample: 1000
  rollback_mode: off             # off|warn|auto_rollback
  catastrophic_delta: 0.25
```

---

## CLI (Draft)
```
radiance-peft run --backend mps --base-model mistral-7b-instruct \
  --update-mode batched --batch-max-examples 128 --batch-max-age-min 30

radiance-peft eval --anchor-provider hf \
  --anchor-dataset tnn1t1s/news-100-sept-2023
```

---

## Open Research Questions
- What thresholds best define "catastrophic" forgetting?
- When should rollback be enabled vs warn-only?
- How should anchor sets be curated for different domains?
- How should LoRA rank scale with update frequency?

---

## Implementation Notes
- Default to stable checkpoints for `search()`.
- Allow optional in-training adapter usage for low-latency updates.
- Ensure all outputs include checkpoint id and status.
- Log metrics and runtime metadata for reproducibility.
