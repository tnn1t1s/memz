"""Tests for memz.server -- FastAPI endpoints via TestClient."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memz.checkpoint import CheckpointManager, CheckpointMeta
from memz.config import EvaluationConfig, ServiceConfig


@pytest.fixture
def mock_state(tmp_path):
    """Create an AppState with mocked heavy dependencies."""
    from memz.server import AppState

    cfg = ServiceConfig(
        base_model="sshleifer/tiny-gpt2",
        backend="cpu",
        data_dir=str(tmp_path),
        evaluation=EvaluationConfig(anchor_provider="none", rollback_mode="off"),
    )

    with patch("memz.server.load_tokenizer") as mock_tok, \
         patch("memz.server.get_device", return_value="cpu"), \
         patch("memz.server.get_dtype"):
        mock_tok.return_value = MagicMock()
        state = AppState(cfg)
        state.checkpoint_mgr = CheckpointManager(checkpoints_dir=tmp_path / "ckpts")
    return state


@pytest.fixture
def client(mock_state):
    """Create a TestClient using the module-level app with mocked state."""
    import memz.server as server_mod

    # Use the module-level app which has the routes registered
    original_state = server_mod._state
    server_mod._state = mock_state

    # Use raise_server_exceptions=False so we can test error codes
    yield TestClient(server_mod.app, raise_server_exceptions=False)

    server_mod._state = original_state


class TestInfoEndpoint:
    """Test GET /info."""

    def test_info_returns_expected_fields(self, client, mock_state):
        resp = client.get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "base_model" in body
        assert "current_checkpoint" in body
        assert "queue_depth" in body
        assert "last_update" in body
        assert "loss_curves" in body
        assert "forgetting_summary" in body
        assert "hardware" in body

    def test_info_no_checkpoints(self, client):
        resp = client.get("/info")
        body = resp.json()
        assert body["current_checkpoint"] is None
        assert body["queue_depth"] == 0

    def test_info_hardware_section(self, client, mock_state):
        resp = client.get("/info")
        body = resp.json()
        assert body["hardware"]["backend"] == "cpu"

    def test_info_loss_curves_structure(self, client):
        resp = client.get("/info")
        body = resp.json()
        curves = body["loss_curves"]
        assert "finetune_loss" in curves
        assert "forgetting_loss" in curves
        assert "eval_loss" in curves


class TestUpdateEndpoint:
    """Test POST /update."""

    def test_update_valid_examples(self, client, mock_state):
        resp = client.post("/update", json={
            "examples": [
                {"input": "hello", "output": "world"},
                {"input": "foo", "output": "bar"},
            ],
            "tags": ["test"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["queued"] is True
        assert body["job_id"].startswith("job_")

    def test_update_empty_input_rejected(self, client):
        resp = client.post("/update", json={
            "examples": [{"input": "", "output": "something"}],
        })
        assert resp.status_code == 422

    def test_update_empty_output_rejected(self, client):
        resp = client.post("/update", json={
            "examples": [{"input": "something", "output": ""}],
        })
        assert resp.status_code == 422

    def test_update_whitespace_only_rejected(self, client):
        resp = client.post("/update", json={
            "examples": [{"input": "   ", "output": "ok"}],
        })
        assert resp.status_code == 422

    def test_update_no_examples_field(self, client):
        resp = client.post("/update", json={})
        assert resp.status_code == 422

    def test_update_enqueues_job(self, client, mock_state):
        initial_depth = len(mock_state.queue)
        client.post("/update", json={
            "examples": [{"input": "a", "output": "b"}],
        })
        assert len(mock_state.queue) == initial_depth + 1


class TestSearchEndpoint:
    """Test POST /search."""

    def test_search_missing_prompt(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422

    def test_search_with_mocked_model(self, client, mock_state):
        """Test search with a mocked model to verify response shape."""
        import torch

        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        mock_state.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.decode = MagicMock(return_value="generated text")
        mock_state.tokenizer = mock_tokenizer

        resp = client.post("/search", json={"prompt": "test prompt"})
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "checkpoint" in body
        assert "checkpoint_status" in body
        assert "latency_ms" in body
        assert body["response"] == "generated text"

    def test_search_returns_checkpoint_info(self, client, mock_state, tmp_path):
        """When a checkpoint exists, search should return its id and status."""
        import torch

        # Create a fake checkpoint
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "weights.bin").write_bytes(b"fake")
        meta = mock_state.checkpoint_mgr.save_checkpoint(adapter_dir, steps=50)

        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        mock_state.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        mock_tokenizer.decode = MagicMock(return_value="ok")
        mock_state.tokenizer = mock_tokenizer

        resp = client.post("/search", json={"prompt": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["checkpoint"] == meta.checkpoint_id
        assert body["checkpoint_status"] == "stable"


class TestResponseModels:
    """Test request/response model validation."""

    def test_update_request_defaults(self):
        from memz.server import UpdateRequest
        req = UpdateRequest(examples=[{"input": "a", "output": "b"}])
        assert req.tags == []

    def test_search_request_defaults(self):
        from memz.server import SearchRequest
        req = SearchRequest(prompt="hello")
        assert req.max_tokens == 256
        assert req.temperature == 0.2
        assert req.use_training_adapter is False
