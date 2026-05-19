#!/usr/bin/env python3
"""Tests for turbo_engine module."""

import time

import numpy as np
import pytest


class TestProcessingResolutionManager:
    """Test ProcessingResolutionManager scaling logic."""

    def test_balanced_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "balanced")
        assert mgr.display_w == 1280
        assert mgr.display_h == 720
        assert mgr.proc_w == 480
        assert mgr.proc_h == 270
        assert mgr.process_resolution == (480, 270)
        assert mgr.display_resolution == (1280, 720)

    def test_ultra_fast_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "ultra_fast")
        assert mgr.proc_w == 320
        assert mgr.proc_h == 180

    def test_fast_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "fast")
        assert mgr.proc_w == 426
        assert mgr.proc_h == 240

    def test_quality_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "quality")
        assert mgr.proc_w == 640
        assert mgr.proc_h == 360

    def test_native_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "native")
        assert mgr.proc_w == 1280
        assert mgr.proc_h == 720

    def test_invalid_preset(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        with pytest.raises(ValueError, match="Unknown preset"):
            ProcessingResolutionManager((1280, 720), "imaginary")

    def test_downscale(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "balanced")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = mgr.downscale(frame)
        assert result.shape == (270, 480, 3)

    def test_upscale(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "balanced")
        frame = np.zeros((270, 480, 3), dtype=np.uint8)
        result = mgr.upscale(frame)
        assert result.shape == (720, 1280, 3)

    def test_downscale_noop_at_native(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((640, 480), "native")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mgr.downscale(frame)
        assert result is frame  # No copy, same object

    def test_upscale_noop_at_native(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((640, 480), "native")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mgr.upscale(frame)
        assert result is frame

    def test_pixel_reduction_ratio(self):
        from scripts.turbo_engine import ProcessingResolutionManager

        mgr = ProcessingResolutionManager((1280, 720), "balanced")
        display_pixels = mgr.display_w * mgr.display_h
        proc_pixels = mgr.proc_w * mgr.proc_h
        ratio = display_pixels / proc_pixels
        # 1280*720 / 480*270 ≈ 7.1x fewer pixels
        assert ratio > 5.0


class TestPerformanceProfiler:
    """Test PerformanceProfiler timing logic."""

    def test_basic_timing(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        prof.start("test")
        time.sleep(0.01)
        prof.stop("test")

        avg = prof.avg_ms("test")
        assert avg > 5  # At least 5ms (sleep 10ms but timer imprecision)

    def test_fps_counting(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        for _ in range(10):
            prof.tick()

        # FPS depends on how fast the loop runs, just check it's positive
        assert prof.fps() > 0

    def test_unknown_section_returns_zero(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        assert prof.avg_ms("nonexistent") == 0.0

    def test_budget_remaining(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        # No sections timed, full budget available
        budget = prof.budget_remaining_ms(30.0)
        assert abs(budget - 33.33) < 0.1  # ~33.3ms at 30fps

    def test_summary_returns_dict(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        prof.start("a")
        prof.stop("a")
        prof.tick()

        summary = prof.summary()
        assert "fps" in summary
        assert "a" in summary
        assert "budget" in summary

    def test_report_returns_string(self):
        from scripts.turbo_engine import PerformanceProfiler

        prof = PerformanceProfiler()
        prof.start("test_section")
        prof.stop("test_section")

        report = prof.report()
        assert "test_section" in report
        assert "FPS" in report
