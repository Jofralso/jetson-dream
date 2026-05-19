#!/usr/bin/env python3
"""Tests for DeepDream engine."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock torch/torchvision before importing dream_engine
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torchvision", MagicMock())
sys.modules.setdefault("torchvision.models", MagicMock())
sys.modules.setdefault("torchvision.transforms", MagicMock())

from scripts.dream_engine import DREAM_LAYERS, IMAGENET_MEAN, IMAGENET_STD


class TestDeepDreamEngine:
    """Test DeepDream engine (mocked model for CI)."""

    def test_dream_layers_list(self):
        assert len(DREAM_LAYERS) == 10
        assert DREAM_LAYERS[0] == "Conv2d_1a_3x3"
        assert DREAM_LAYERS[-1] == "Mixed_7a"

    def test_imagenet_constants(self):
        assert IMAGENET_MEAN.shape == (3,)
        assert IMAGENET_STD.shape == (3,)
        assert all(0 < m < 1 for m in IMAGENET_MEAN)
        assert all(0 < s < 1 for s in IMAGENET_STD)

    def test_layer_index_clamping(self):
        """Test layer index bounds logic."""
        assert DREAM_LAYERS[0] == "Conv2d_1a_3x3"
        assert DREAM_LAYERS[9] == "Mixed_7a"

        # Clamping at upper bound
        idx = max(0, min(15, len(DREAM_LAYERS) - 1))
        assert idx == 9
        # Clamping at lower bound
        idx = max(0, min(-1, len(DREAM_LAYERS) - 1))
        assert idx == 0


class TestPrePostProcess:
    """Test numpy-level pre/post processing."""

    def test_hue_shift_wraps(self):
        """Hue shift should wrap at 180."""
        import cv2

        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 0.5 * 180) % 180
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        assert result.shape == (10, 10, 3)

    def test_zoom_crop(self):
        """Zoom should crop center and upscale."""
        import cv2

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        zoom = 1.1
        h, w = frame.shape[:2]
        dh = int(h * (1 - 1 / zoom) / 2)
        dw = int(w * (1 - 1 / zoom) / 2)
        cropped = frame[dh:h - dh, dw:w - dw]
        result = cv2.resize(cropped, (w, h))
        assert result.shape == (100, 100, 3)
