#!/usr/bin/env python3
"""Tests for LiteDreamEngine (MobileNetV2)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check if torch is real (not mocked by another test)
try:
    import torch
    HAS_REAL_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_REAL_TORCH = False


class TestMobileNetDreamModel:
    """Test MobileNetV2 dream model structure."""

    @pytest.mark.skipif(not HAS_REAL_TORCH, reason="Real PyTorch not available")
    def test_mobilenet_layers_list(self):
        from scripts.lite_dream_engine import MOBILENET_LAYERS
        assert len(MOBILENET_LAYERS) == 8
        assert MOBILENET_LAYERS[0] == "features.0"
        assert MOBILENET_LAYERS[-1] == "features.18"

    @pytest.mark.skipif(not HAS_REAL_TORCH, reason="Real PyTorch not available")
    def test_layer_index_resolve(self):
        from scripts.lite_dream_engine import MobileNetDreamModel
        # Test static method mapping and boundary clamping
        assert MobileNetDreamModel._resolve_layer_idx(0) == 0    # features.0
        assert MobileNetDreamModel._resolve_layer_idx(4) == 10   # features.10
        assert MobileNetDreamModel._resolve_layer_idx(7) == 18   # features.18
        assert MobileNetDreamModel._resolve_layer_idx(100) == 18 # clamp high
        assert MobileNetDreamModel._resolve_layer_idx(-1) == 0   # clamp low


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="Real PyTorch not available")
class TestLiteDreamEngineIntegration:
    """Integration tests (require PyTorch, can run on CPU)."""

    @pytest.fixture
    def engine(self):
        from scripts.lite_dream_engine import LiteDreamEngine
        return LiteDreamEngine(resolution=(64, 48))

    @pytest.mark.integration
    def test_create_engine(self, engine):
        assert engine.device is not None
        assert engine.layer_index == 4

    @pytest.mark.integration
    def test_process_frame(self, engine):
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result = engine.process_frame(frame)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    @pytest.mark.integration
    def test_frame_skip(self, engine):
        engine.frame_skip = 1
        frame1 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result1 = engine.process_frame(frame1)
        # Second call should return cached
        frame2 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result2 = engine.process_frame(frame2)
        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.integration
    def test_update_params(self, engine):
        engine.update_params({
            "layer_index": 2,
            "intensity": 0.05,
            "iterations": 3,
            "feedback": 0.5,
        })
        assert engine.layer_index == 2
        assert engine.intensity == 0.05
        assert engine.iterations == 3
        assert engine.feedback == 0.5

    @pytest.mark.integration
    def test_get_info(self, engine):
        info = engine.get_info()
        assert "engine" in info
        assert "MobileNetV2" in info["engine"]
        assert "params" in info

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_frames_consistency(self, engine):
        """Process multiple frames to test feedback loop stability."""
        engine.feedback = 0.3
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        for _ in range(5):
            result = engine.process_frame(frame)
            assert result.shape == (48, 64, 3)
            assert not np.isnan(result.astype(float)).any()
