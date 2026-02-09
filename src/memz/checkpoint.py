"""Checkpoint manager for adapter lineage and rollback."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata for a single adapter checkpoint."""

    checkpoint_id: str
    created_at: str
    adapter_path: str
    parent_id: str | None = None
    status: str = "stable"  # stable | risky | rolled_back
    steps: int = 0
    metrics: dict = field(default_factory=dict)


class CheckpointManager:
    """Manages adapter checkpoint lineage, latest pointer, and rollback.

    Storage layout:
        checkpoints_dir/
            A_<timestamp>/
                adapter/         # peft adapter weights
                meta.json        # CheckpointMeta serialized
            latest.json          # pointer to current stable checkpoint
    """

    def __init__(self, checkpoints_dir: str | Path) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self.checkpoints_dir / "latest.json"

    def _generate_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        return f"A_{ts}"

    def _load_latest(self) -> str | None:
        if not self._latest_path.exists():
            return None
        data = json.loads(self._latest_path.read_text())
        return data.get("checkpoint_id")

    def _save_latest(self, checkpoint_id: str) -> None:
        self._latest_path.write_text(
            json.dumps({"checkpoint_id": checkpoint_id}, indent=2)
        )

    def get_latest_stable(self) -> CheckpointMeta | None:
        """Return metadata for the latest stable checkpoint, or None."""
        ckpt_id = self._load_latest()
        if ckpt_id is None:
            return None
        return self.load_meta(ckpt_id)

    def load_meta(self, checkpoint_id: str) -> CheckpointMeta | None:
        """Load metadata for a specific checkpoint."""
        meta_path = self.checkpoints_dir / checkpoint_id / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return CheckpointMeta(**data)

    def list_checkpoints(self) -> list[CheckpointMeta]:
        """List all checkpoints, sorted by creation time."""
        metas = []
        for d in sorted(self.checkpoints_dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                meta = self.load_meta(d.name)
                if meta is not None:
                    metas.append(meta)
        return metas

    def save_checkpoint(
        self,
        adapter_source_path: str | Path,
        steps: int,
        metrics: dict | None = None,
    ) -> CheckpointMeta:
        """Register a new checkpoint from a saved adapter.

        Copies the adapter into the managed directory, writes metadata,
        and updates the latest pointer.
        """
        checkpoint_id = self._generate_id()
        parent_id = self._load_latest()

        ckpt_dir = self.checkpoints_dir / checkpoint_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Copy adapter files
        dest_adapter = ckpt_dir / "adapter"
        source = Path(adapter_source_path)
        if source.is_dir():
            shutil.copytree(source, dest_adapter, dirs_exist_ok=True)
        else:
            dest_adapter.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_adapter)

        meta = CheckpointMeta(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            adapter_path=str(dest_adapter),
            parent_id=parent_id,
            status="stable",
            steps=steps,
            metrics=metrics or {},
        )

        meta_path = ckpt_dir / "meta.json"
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

        self._save_latest(checkpoint_id)
        logger.info("Saved checkpoint %s (parent=%s)", checkpoint_id, parent_id)
        return meta

    def apply_rollback_policy(
        self,
        checkpoint_id: str,
        metrics: dict,
        rollback_mode: str = "off",
        catastrophic_delta: float = 0.25,
        baseline_forgetting: float | None = None,
    ) -> bool:
        """Apply rollback policy based on forgetting metrics.

        Returns True if rollback was triggered.
        """
        if rollback_mode == "off":
            return False

        forgetting = metrics.get("forgetting_loss")
        if forgetting is None:
            return False

        if baseline_forgetting is None:
            # Use parent checkpoint forgetting as baseline
            meta = self.load_meta(checkpoint_id)
            if meta and meta.parent_id:
                parent = self.load_meta(meta.parent_id)
                if parent and "forgetting_loss" in parent.metrics:
                    baseline_forgetting = parent.metrics["forgetting_loss"]

        if baseline_forgetting is None:
            return False

        delta = forgetting - baseline_forgetting
        if delta <= catastrophic_delta:
            return False

        logger.warning(
            "Forgetting delta %.4f exceeds threshold %.4f for %s",
            delta,
            catastrophic_delta,
            checkpoint_id,
        )

        if rollback_mode == "warn":
            # Mark as risky but do not rollback
            self._mark_status(checkpoint_id, "risky")
            return False

        if rollback_mode == "auto_rollback":
            return self.rollback(checkpoint_id)

        return False

    def rollback(self, checkpoint_id: str) -> bool:
        """Roll back to the parent of the given checkpoint.

        Returns True if rollback succeeded.
        """
        meta = self.load_meta(checkpoint_id)
        if meta is None or meta.parent_id is None:
            logger.warning("Cannot rollback %s: no parent", checkpoint_id)
            return False

        parent = self.load_meta(meta.parent_id)
        if parent is None:
            logger.warning("Cannot rollback: parent %s not found", meta.parent_id)
            return False

        self._mark_status(checkpoint_id, "rolled_back")
        self._save_latest(meta.parent_id)
        logger.info("Rolled back from %s to %s", checkpoint_id, meta.parent_id)
        return True

    def _mark_status(self, checkpoint_id: str, status: str) -> None:
        """Update the status field in a checkpoint's metadata."""
        meta = self.load_meta(checkpoint_id)
        if meta is None:
            return
        meta.status = status
        meta_path = self.checkpoints_dir / checkpoint_id / "meta.json"
        meta_path.write_text(json.dumps(asdict(meta), indent=2))
