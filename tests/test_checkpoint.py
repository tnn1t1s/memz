"""Tests for memz.checkpoint -- lifecycle, lineage, and rollback."""

import json
import time

import pytest

from memz.checkpoint import CheckpointManager, CheckpointMeta


@pytest.fixture
def ckpt_mgr(tmp_path):
    """Return a CheckpointManager backed by a temp directory."""
    return CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")


@pytest.fixture
def adapter_dir(tmp_path):
    """Create a fake adapter directory with a dummy file."""
    d = tmp_path / "fake_adapter"
    d.mkdir()
    (d / "adapter_model.bin").write_bytes(b"fake weights")
    (d / "adapter_config.json").write_text('{"r": 8}')
    return d


class TestCheckpointCreation:
    """Test saving and loading checkpoints."""

    def test_save_creates_meta(self, ckpt_mgr, adapter_dir):
        meta = ckpt_mgr.save_checkpoint(adapter_dir, steps=100, metrics={"loss": 0.5})
        assert meta.checkpoint_id.startswith("A_")
        assert meta.status == "stable"
        assert meta.steps == 100
        assert meta.metrics == {"loss": 0.5}
        assert meta.parent_id is None  # first checkpoint

    def test_save_copies_adapter_files(self, ckpt_mgr, adapter_dir):
        meta = ckpt_mgr.save_checkpoint(adapter_dir, steps=50)
        dest = ckpt_mgr.checkpoints_dir / meta.checkpoint_id / "adapter"
        assert dest.exists()
        assert (dest / "adapter_model.bin").exists()
        assert (dest / "adapter_config.json").exists()

    def test_load_meta_roundtrip(self, ckpt_mgr, adapter_dir):
        meta = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        loaded = ckpt_mgr.load_meta(meta.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == meta.checkpoint_id
        assert loaded.steps == meta.steps
        assert loaded.status == meta.status

    def test_load_meta_nonexistent_returns_none(self, ckpt_mgr):
        assert ckpt_mgr.load_meta("nonexistent_id") is None


class TestLineage:
    """Test parent-child checkpoint lineage."""

    def test_first_checkpoint_has_no_parent(self, ckpt_mgr, adapter_dir):
        meta = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        assert meta.parent_id is None

    def test_second_checkpoint_parent_is_first(self, ckpt_mgr, adapter_dir):
        m1 = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        # Need a small delay so the timestamp-based ID differs
        time.sleep(1.1)
        m2 = ckpt_mgr.save_checkpoint(adapter_dir, steps=20)
        assert m2.parent_id == m1.checkpoint_id

    def test_lineage_chain(self, ckpt_mgr, adapter_dir):
        ids = []
        for i in range(3):
            if i > 0:
                time.sleep(1.1)
            m = ckpt_mgr.save_checkpoint(adapter_dir, steps=(i + 1) * 10)
            ids.append(m.checkpoint_id)

        m2 = ckpt_mgr.load_meta(ids[1])
        m3 = ckpt_mgr.load_meta(ids[2])
        assert m2.parent_id == ids[0]
        assert m3.parent_id == ids[1]


class TestGetLatestStable:
    """Test latest stable checkpoint retrieval."""

    def test_no_checkpoints_returns_none(self, ckpt_mgr):
        assert ckpt_mgr.get_latest_stable() is None

    def test_single_checkpoint(self, ckpt_mgr, adapter_dir):
        m = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        latest = ckpt_mgr.get_latest_stable()
        assert latest is not None
        assert latest.checkpoint_id == m.checkpoint_id

    def test_latest_tracks_most_recent(self, ckpt_mgr, adapter_dir):
        ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        time.sleep(1.1)
        m2 = ckpt_mgr.save_checkpoint(adapter_dir, steps=20)
        latest = ckpt_mgr.get_latest_stable()
        assert latest.checkpoint_id == m2.checkpoint_id


class TestListCheckpoints:
    """Test listing all checkpoints."""

    def test_empty_dir(self, ckpt_mgr):
        assert ckpt_mgr.list_checkpoints() == []

    def test_lists_all(self, ckpt_mgr, adapter_dir):
        ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        time.sleep(1.1)
        ckpt_mgr.save_checkpoint(adapter_dir, steps=20)
        all_ckpts = ckpt_mgr.list_checkpoints()
        assert len(all_ckpts) == 2


class TestRollbackPolicy:
    """Test rollback modes: off, warn, auto_rollback."""

    def _create_two_checkpoints(self, ckpt_mgr, adapter_dir):
        m1 = ckpt_mgr.save_checkpoint(
            adapter_dir, steps=10, metrics={"forgetting_loss": 1.0}
        )
        time.sleep(1.1)
        m2 = ckpt_mgr.save_checkpoint(
            adapter_dir, steps=20, metrics={"forgetting_loss": 1.5}
        )
        return m1, m2

    def test_off_mode_never_rolls_back(self, ckpt_mgr, adapter_dir):
        _, m2 = self._create_two_checkpoints(ckpt_mgr, adapter_dir)
        result = ckpt_mgr.apply_rollback_policy(
            checkpoint_id=m2.checkpoint_id,
            metrics={"forgetting_loss": 10.0},
            rollback_mode="off",
        )
        assert result is False

    def test_warn_mode_marks_risky(self, ckpt_mgr, adapter_dir):
        _, m2 = self._create_two_checkpoints(ckpt_mgr, adapter_dir)
        result = ckpt_mgr.apply_rollback_policy(
            checkpoint_id=m2.checkpoint_id,
            metrics={"forgetting_loss": 10.0},
            rollback_mode="warn",
            catastrophic_delta=0.25,
        )
        assert result is False
        updated = ckpt_mgr.load_meta(m2.checkpoint_id)
        assert updated.status == "risky"

    def test_auto_rollback_reverts_to_parent(self, ckpt_mgr, adapter_dir):
        m1, m2 = self._create_two_checkpoints(ckpt_mgr, adapter_dir)
        result = ckpt_mgr.apply_rollback_policy(
            checkpoint_id=m2.checkpoint_id,
            metrics={"forgetting_loss": 10.0},
            rollback_mode="auto_rollback",
            catastrophic_delta=0.25,
        )
        assert result is True
        latest = ckpt_mgr.get_latest_stable()
        assert latest.checkpoint_id == m1.checkpoint_id
        rolled_back = ckpt_mgr.load_meta(m2.checkpoint_id)
        assert rolled_back.status == "rolled_back"

    def test_no_rollback_when_delta_below_threshold(self, ckpt_mgr, adapter_dir):
        _, m2 = self._create_two_checkpoints(ckpt_mgr, adapter_dir)
        result = ckpt_mgr.apply_rollback_policy(
            checkpoint_id=m2.checkpoint_id,
            metrics={"forgetting_loss": 1.1},  # delta = 0.1, below 0.25
            rollback_mode="auto_rollback",
            catastrophic_delta=0.25,
        )
        assert result is False

    def test_no_rollback_without_forgetting_metric(self, ckpt_mgr, adapter_dir):
        _, m2 = self._create_two_checkpoints(ckpt_mgr, adapter_dir)
        result = ckpt_mgr.apply_rollback_policy(
            checkpoint_id=m2.checkpoint_id,
            metrics={},  # no forgetting_loss
            rollback_mode="auto_rollback",
        )
        assert result is False


class TestRollbackDirect:
    """Test the direct rollback method."""

    def test_rollback_with_no_parent_fails(self, ckpt_mgr, adapter_dir):
        m1 = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        assert ckpt_mgr.rollback(m1.checkpoint_id) is False

    def test_rollback_nonexistent_fails(self, ckpt_mgr):
        assert ckpt_mgr.rollback("nonexistent") is False

    def test_rollback_updates_latest(self, ckpt_mgr, adapter_dir):
        m1 = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        time.sleep(1.1)
        m2 = ckpt_mgr.save_checkpoint(adapter_dir, steps=20)
        assert ckpt_mgr.rollback(m2.checkpoint_id) is True
        latest = ckpt_mgr.get_latest_stable()
        assert latest.checkpoint_id == m1.checkpoint_id


class TestStatusTransitions:
    """Test checkpoint status changes."""

    def test_new_checkpoint_is_stable(self, ckpt_mgr, adapter_dir):
        m = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        assert m.status == "stable"

    def test_mark_risky(self, ckpt_mgr, adapter_dir):
        m = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        ckpt_mgr._mark_status(m.checkpoint_id, "risky")
        updated = ckpt_mgr.load_meta(m.checkpoint_id)
        assert updated.status == "risky"

    def test_mark_rolled_back(self, ckpt_mgr, adapter_dir):
        m = ckpt_mgr.save_checkpoint(adapter_dir, steps=10)
        ckpt_mgr._mark_status(m.checkpoint_id, "rolled_back")
        updated = ckpt_mgr.load_meta(m.checkpoint_id)
        assert updated.status == "rolled_back"
