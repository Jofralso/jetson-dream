#!/usr/bin/env python3
"""Tests for async_pipeline module."""

import threading
import time

import numpy as np
import pytest

from scripts.async_pipeline import FrameBuffer


class TestFrameBuffer:
    """Test thread-safe FrameBuffer."""

    def test_put_and_get(self):
        buf = FrameBuffer()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        buf.put(frame)
        result = buf.get(timeout=1.0)
        assert result is not None
        assert result.shape == (100, 200, 3)

    def test_get_blocks_on_empty(self):
        buf = FrameBuffer()
        result = buf.get(timeout=0.05)
        assert result is None

    def test_latest_frame_wins(self):
        """Only the latest frame is returned."""
        buf = FrameBuffer()
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 200

        buf.put(frame1)
        buf.put(frame2)

        result = buf.get(timeout=1.0)
        assert result is not None
        assert result[0, 0, 0] == 200  # Got the latest frame

    def test_peek_does_not_consume(self):
        buf = FrameBuffer()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        buf.put(frame)

        peek1 = buf.peek()
        peek2 = buf.peek()
        assert peek1 is not None
        assert peek2 is not None

    def test_peek_empty_returns_none(self):
        buf = FrameBuffer()
        assert buf.peek() is None

    def test_frame_id_increments(self):
        buf = FrameBuffer()
        assert buf.frame_id == 0

        buf.put(np.zeros((10, 10, 3), dtype=np.uint8))
        assert buf.frame_id == 1

        buf.put(np.zeros((10, 10, 3), dtype=np.uint8))
        assert buf.frame_id == 2

    def test_thread_safety(self):
        """Test concurrent put/get from different threads."""
        buf = FrameBuffer()
        results = []
        errors = []

        def writer():
            try:
                for i in range(50):
                    frame = np.full((10, 10, 3), i % 256, dtype=np.uint8)
                    buf.put(frame)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    frame = buf.get(timeout=0.1)
                    if frame is not None:
                        results.append(frame[0, 0, 0])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors
        assert len(results) > 0  # Got at least some frames

    def test_get_clears_event(self):
        """After get(), subsequent get() blocks until new put()."""
        buf = FrameBuffer()
        buf.put(np.zeros((10, 10, 3), dtype=np.uint8))

        result1 = buf.get(timeout=1.0)
        assert result1 is not None

        # Should block and return None (no new frame)
        result2 = buf.get(timeout=0.05)
        assert result2 is None


class TestCaptureThread:
    """Test CaptureThread with a mock VideoCapture."""

    def test_capture_thread_basics(self):
        """Test that CaptureThread can be created."""
        from scripts.async_pipeline import CaptureThread

        buf = FrameBuffer()
        # We don't actually start it (no real camera), just verify construction
        thread = CaptureThread(None, buf, 1280, 720)
        assert thread.name == "capture"
        assert thread.daemon is True
        assert thread.fps == 0.0


class TestProcessThread:
    """Test ProcessThread with a mock processor."""

    def test_process_thread_basics(self):
        from scripts.async_pipeline import ProcessThread

        in_buf = FrameBuffer()
        out_buf = FrameBuffer()

        def passthrough(frame):
            return frame

        thread = ProcessThread(in_buf, out_buf, passthrough)
        assert thread.name == "process"
        assert thread.daemon is True

    def test_process_thread_runs(self):
        from scripts.async_pipeline import ProcessThread

        in_buf = FrameBuffer()
        out_buf = FrameBuffer()

        def double_brightness(frame):
            return np.clip(frame.astype(np.int16) * 2, 0, 255).astype(np.uint8)

        thread = ProcessThread(in_buf, out_buf, double_brightness)
        thread.start()

        try:
            # Feed a frame
            input_frame = np.full((10, 10, 3), 50, dtype=np.uint8)
            in_buf.put(input_frame)

            # Wait for output
            result = out_buf.get(timeout=2.0)
            assert result is not None
            assert result[0, 0, 0] == 100  # doubled
        finally:
            thread.stop()
            thread.join(timeout=2)

    def test_process_with_scaling(self):
        from scripts.async_pipeline import ProcessThread

        in_buf = FrameBuffer()
        out_buf = FrameBuffer()

        import cv2

        def downscale(frame):
            return cv2.resize(frame, (5, 5))

        def upscale(frame):
            return cv2.resize(frame, (20, 20))

        def passthrough(frame):
            return frame

        thread = ProcessThread(in_buf, out_buf, passthrough, downscale, upscale)
        thread.start()

        try:
            in_buf.put(np.zeros((10, 10, 3), dtype=np.uint8))
            result = out_buf.get(timeout=2.0)
            assert result is not None
            assert result.shape == (20, 20, 3)
        finally:
            thread.stop()
            thread.join(timeout=2)

    def test_update_process_fn(self):
        from scripts.async_pipeline import ProcessThread

        in_buf = FrameBuffer()
        out_buf = FrameBuffer()

        def fn1(frame):
            return np.full_like(frame, 10)

        def fn2(frame):
            return np.full_like(frame, 200)

        thread = ProcessThread(in_buf, out_buf, fn1)
        thread.start()

        try:
            # First fn
            in_buf.put(np.zeros((5, 5, 3), dtype=np.uint8))
            r1 = out_buf.get(timeout=2.0)
            assert r1 is not None
            assert r1[0, 0, 0] == 10

            # Swap to second fn
            thread.update_process_fn(fn2)
            time.sleep(0.05)

            in_buf.put(np.zeros((5, 5, 3), dtype=np.uint8))
            r2 = out_buf.get(timeout=2.0)
            assert r2 is not None
            assert r2[0, 0, 0] == 200
        finally:
            thread.stop()
            thread.join(timeout=2)
