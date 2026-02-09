"""FastAPI server for the continuous PEFT service."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memz.checkpoint import CheckpointManager
from memz.config import ServiceConfig, load_config
from memz.device import get_device, get_dtype
from memz.trainer import load_tokenizer, run_training

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    use_training_adapter: bool = False


class SearchResponse(BaseModel):
    response: str
    checkpoint: str | None
    checkpoint_status: str | None
    latency_ms: float


class Example(BaseModel):
    input: str
    output: str


class UpdateRequest(BaseModel):
    examples: list[Example]
    tags: list[str] = Field(default_factory=list)


class UpdateResponse(BaseModel):
    job_id: str
    queued: bool


# ---------------------------------------------------------------------------
# Application state (set during lifespan)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.checkpoint_mgr = CheckpointManager(checkpoints_dir="./checkpoints")
        self.tokenizer = load_tokenizer(config)
        self.model = None  # lazy loaded on first search
        self.queue: deque[dict] = deque()
        self.last_update: str | None = None
        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

    def _worker_loop(self) -> None:
        """Background worker that drains the training queue."""
        while not self.shutdown_event.is_set():
            if not self.queue:
                self.shutdown_event.wait(timeout=1.0)
                continue

            job = self.queue.popleft()
            try:
                logger.info("Training job %s started (%d examples)", job["job_id"], len(job["examples"]))
                result = run_training(
                    config=self.config,
                    examples=job["examples"],
                    output_dir=f"./training_runs/{job['job_id']}",
                    device=self.device,
                )
                # Register checkpoint
                metrics = {
                    "finetune_loss": result.final_loss,
                    "steps": result.steps_completed,
                }
                meta = self.checkpoint_mgr.save_checkpoint(
                    adapter_source_path=result.adapter_path,
                    steps=result.steps_completed,
                    metrics=metrics,
                )
                # Apply rollback policy
                self.checkpoint_mgr.apply_rollback_policy(
                    checkpoint_id=meta.checkpoint_id,
                    metrics=metrics,
                    rollback_mode=self.config.evaluation.rollback_mode,
                    catastrophic_delta=self.config.evaluation.catastrophic_delta,
                )
                self.last_update = datetime.now(timezone.utc).isoformat()
                logger.info("Training job %s complete: checkpoint %s", job["job_id"], meta.checkpoint_id)

                # Unload cached model so next search picks up the new adapter
                self.model = None

            except Exception:
                logger.exception("Training job %s failed", job["job_id"])

    def start_worker(self) -> None:
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop_worker(self) -> None:
        self.shutdown_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=10)

    def ensure_model(self):
        """Lazy-load or reload the model with the latest adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if self.model is not None:
            return self.model

        logger.info("Loading base model %s ...", self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Load latest adapter if available
        latest = self.checkpoint_mgr.get_latest_stable()
        if latest is not None:
            adapter_path = latest.adapter_path
            if Path(adapter_path).exists():
                logger.info("Loading adapter from %s", adapter_path)
                model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model
        return model


# ---------------------------------------------------------------------------
# Global state reference (set in lifespan)
# ---------------------------------------------------------------------------

_state: AppState | None = None


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("Server not initialized")
    return _state


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Create the FastAPI application with the given config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _state
        cfg = config or load_config("config.yaml")
        _state = AppState(cfg)
        _state.start_worker()
        logger.info("memz started: backend=%s model=%s", cfg.backend, cfg.base_model)
        yield
        _state.stop_worker()
        logger.info("memz shutting down")

    return FastAPI(title="memz", description="Continuous PEFT service", lifespan=lifespan)


app = create_app()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    state = get_state()

    t0 = time.monotonic()

    model = state.ensure_model()
    tokenizer = state.tokenizer

    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(state.device)
    attention_mask = inputs["attention_mask"].to(state.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else None,
            do_sample=req.temperature > 0,
        )

    generated = outputs[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    latency_ms = (time.monotonic() - t0) * 1000

    latest = state.checkpoint_mgr.get_latest_stable()
    return SearchResponse(
        response=response_text,
        checkpoint=latest.checkpoint_id if latest else None,
        checkpoint_status=latest.status if latest else None,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/update", response_model=UpdateResponse)
async def update(req: UpdateRequest):
    state = get_state()

    # Validate
    for i, ex in enumerate(req.examples):
        if not ex.input.strip() or not ex.output.strip():
            raise HTTPException(status_code=422, detail=f"Example {i} has empty input or output")

    # Store to curated JSONL
    curated_dir = Path("data/curated")
    curated_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    curated_path = curated_dir / f"{job_id}.jsonl"
    with open(curated_path, "w") as f:
        for ex in req.examples:
            f.write(json.dumps({"input": ex.input, "output": ex.output}) + "\n")

    # Enqueue
    examples_dicts = [{"input": ex.input, "output": ex.output} for ex in req.examples]

    if state.config.update_mode == "immediate":
        state.queue.appendleft({"job_id": job_id, "examples": examples_dicts, "tags": req.tags})
    else:
        state.queue.append({"job_id": job_id, "examples": examples_dicts, "tags": req.tags})

    logger.info("Enqueued job %s with %d examples", job_id, len(req.examples))

    return UpdateResponse(job_id=job_id, queued=True)


@app.get("/info")
async def info():
    state = get_state()

    ckpt_info = state.checkpoint_mgr.get_latest_stable()

    # Build loss curves from checkpoint history
    all_ckpts = state.checkpoint_mgr.list_checkpoints()
    finetune_loss = []
    forgetting_loss = []
    for c in all_ckpts:
        if "finetune_loss" in c.metrics:
            finetune_loss.append({"checkpoint": c.checkpoint_id, "value": c.metrics["finetune_loss"]})
        if "forgetting_loss" in c.metrics:
            forgetting_loss.append({"checkpoint": c.checkpoint_id, "value": c.metrics["forgetting_loss"]})

    # Forgetting summary
    forgetting_summary = None
    if state.config.evaluation.anchor_provider != "none" and ckpt_info and "forgetting_loss" in ckpt_info.metrics:
        baseline = next((c for c in all_ckpts if "forgetting_loss" in c.metrics), None)
        if baseline:
            delta = ckpt_info.metrics["forgetting_loss"] - baseline.metrics["forgetting_loss"]
            forgetting_summary = {
                "anchor_ce_delta": round(delta, 4),
                "retention_score": round(max(0.0, 1.0 - delta) if delta >= 0 else 1.0, 4),
            }

    return {
        "base_model": state.config.base_model,
        "current_checkpoint": ckpt_info.checkpoint_id if ckpt_info else None,
        "queue_depth": len(state.queue),
        "last_update": state.last_update,
        "loss_curves": {
            "finetune_loss": finetune_loss or None,
            "forgetting_loss": forgetting_loss or None,
            "eval_loss": None,
        },
        "forgetting_summary": forgetting_summary,
        "hardware": {
            "device": state.device,
            "backend": state.config.backend,
        },
    }
