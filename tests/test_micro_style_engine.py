#!/usr/bin/env python3
"""Tests for MicroStyleEngine."""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")  # Detect mock vs real
except ImportError:
    HAS_TORCH = False


class TestMicroTransformNet:
    """Test the micro architecture."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_model_size(self):
        from scripts.micro_style_engine import MicroTransformNet
        model = MicroTransformNet()
        params = sum(p.numel() for p in model.parameters())
        # Should be well under 1M params
        assert params < 1_000_000
        # Should be at least 100K (sanity)
        assert params > 100_000

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_forward_shape(self):
        import torch
        from scripts.micro_style_engine import MicroTransformNet
        model = MicroTransformNet()
        model.eval()
        x = torch.randn(1, 3, 96, 160)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 96, 160)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_different_input_sizes(self):
        import torch
        from scripts.micro_style_engine import MicroTransformNet
        model = MicroTransformNet()
        model.eval()
        # Test various sizes (all must be divisible by 4 due to 2 downsamples)
        for h, w in [(48, 64), (96, 128), (120, 160), (240, 320)]:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (1, 3, h, w), f"Failed for {h}×{w}"


class TestMicroStyleEngine:
    """Test the engine wrapper."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_create_engine(self, tmp_path):
        from scripts.micro_style_engine import MicroStyleEngine
        engine = MicroStyleEngine(models_dir=str(tmp_path / "models"))
        assert engine.style_names == []  # No models yet

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_process_frame_no_model(self, tmp_path):
        from scripts.micro_style_engine import MicroStyleEngine
        engine = MicroStyleEngine(models_dir=str(tmp_path / "models"))
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        # Should return original frame when no model loaded
        result = engine.process_frame(frame)
        np.testing.assert_array_equal(result, frame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_save_and_load_model(self, tmp_path):
        import torch
        from scripts.micro_style_engine import MicroStyleEngine, MicroTransformNet

        # Save a random model
        models_dir = tmp_path / "models" / "micro"
        models_dir.mkdir(parents=True)
        model = MicroTransformNet()
        torch.save(model.state_dict(), str(models_dir / "test_style.pth"))

        # Load it
        engine = MicroStyleEngine(models_dir=str(models_dir))
        assert "test_style" in engine.style_names
        assert engine.load_style("test_style")

        # Process a frame
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result = engine.process_frame(frame)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_update_params(self, tmp_path):
        from scripts.micro_style_engine import MicroStyleEngine
        engine = MicroStyleEngine(models_dir=str(tmp_path / "models"))
        engine.update_params({"style_blend": 0.5, "feedback": 0.3})
        assert engine.style_blend == 0.5
        assert engine.feedback == 0.3

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_get_info(self, tmp_path):
        from scripts.micro_style_engine import MicroStyleEngine
        engine = MicroStyleEngine(models_dir=str(tmp_path / "models"))
        info = engine.get_info()
        assert info["engine"] == "MicroStyle"
        assert "0.4M" in info["params"]
