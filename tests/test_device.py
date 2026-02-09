"""Tests for memz.device -- device detection and dtype selection."""

from unittest.mock import patch

import torch

from memz.device import get_device, get_dtype


class TestGetDevice:
    """Test device auto-detection with mocked backends."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_preferred(self, _mock_cuda):
        assert get_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_when_no_cuda(self, _mock_mps, _mock_cuda):
        assert get_device() == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback(self, _mock_mps, _mock_cuda):
        assert get_device() == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_cuda_preferred_over_mps(self, _mock_mps, _mock_cuda):
        assert get_device() == "cuda"


class TestGetDtype:
    """Test dtype selection per device."""

    def test_mps_float16(self):
        assert get_dtype("mps") == torch.float16

    def test_cuda_bfloat16(self):
        assert get_dtype("cuda") == torch.bfloat16

    def test_cpu_float32(self):
        assert get_dtype("cpu") == torch.float32

    def test_unknown_device_defaults_to_float32(self):
        assert get_dtype("tpu") == torch.float32
