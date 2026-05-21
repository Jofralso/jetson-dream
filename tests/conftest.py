#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for jetson-dream tests.

Test markers:
  - @pytest.mark.unit        : Fast, no GPU/hardware needed
  - @pytest.mark.integration : Needs PyTorch (can run on CPU)
  - @pytest.mark.gpu         : Needs CUDA GPU
  - @pytest.mark.hardware    : Needs camera/MIDI hardware
  - @pytest.mark.slow        : Takes >5 seconds

Usage:
  pytest                           # Run all unit tests
  pytest -m unit                   # Only unit tests
  pytest -m integration            # Tests needing PyTorch
  pytest -m "not gpu"              # Skip GPU tests
  pytest -m "not hardware"         # Skip hardware tests
  pytest --run-slow                # Include slow tests
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
#  MARKERS
# ═══════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests (no external deps)")
    config.addinivalue_line("markers", "integration: needs PyTorch (CPU ok)")
    config.addinivalue_line("markers", "gpu: needs CUDA GPU")
    config.addinivalue_line("markers", "hardware: needs physical camera or MIDI")
    config.addinivalue_line("markers", "slow: takes >5 seconds")


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Include slow tests")
    parser.addoption("--run-hardware", action="store_true", default=False,
                     help="Include hardware-dependent tests")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on markers and environment."""
    skip_slow = pytest.mark.skip(reason="use --run-slow to run")
    skip_hardware = pytest.mark.skip(reason="use --run-hardware to run")
    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "hardware" in item.keywords and not config.getoption("--run-hardware"):
            item.add_marker(skip_hardware)
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_frame():
    """A 160×120 BGR uint8 test frame with some structure."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    # Add some gradients and shapes for meaningful processing
    frame[:, :, 0] = np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (120, 1))
    frame[:, :, 1] = np.tile(np.linspace(0, 255, 120, dtype=np.uint8).reshape(-1, 1), (1, 160))
    frame[:, :, 2] = 128
    # Add a white circle
    cv2 = pytest.importorskip("cv2")
    cv2.circle(frame, (80, 60), 30, (255, 255, 255), -1)
    return frame


@pytest.fixture
def sample_frame_small():
    """A tiny 32×24 frame for fast tests."""
    return np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_720p():
    """A 1280×720 frame for resolution tests."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def mock_torch():
    """Mock torch module for tests that don't need real PyTorch."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.device.return_value = "cpu"
    return mock


@pytest.fixture
def temp_models_dir(tmp_path):
    """Temporary directory for model files."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def mock_midi_state():
    """Factory for creating MIDI state dicts."""
    def _make(**kwargs):
        state = {
            "pads_held": {},
            "pad_triggers": {},
            "top_buttons": {cc: False for cc in range(91, 99)},
            "top_triggers": {},
            "active_row": -1,
            "active_col": -1,
            "active_velocity": 0.0,
            "num_pads_held": 0,
        }
        state.update(kwargs)
        return state
    return _make


# ═══════════════════════════════════════════════════════════════
#  TORCH AVAILABILITY HELPERS
# ═══════════════════════════════════════════════════════════════


def has_torch():
    """Check if PyTorch is importable."""
    try:
        import torch
        return True
    except ImportError:
        return False


def has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not has_torch(), reason="PyTorch not installed"
)

requires_cuda = pytest.mark.skipif(
    not has_cuda(), reason="CUDA not available"
)
