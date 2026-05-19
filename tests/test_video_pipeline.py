#!/usr/bin/env python3
"""Tests for video pipeline."""

import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from scripts.video_pipeline import build_gst_pipeline


class TestGstPipeline:

    def test_csi_pipeline(self):
        result = build_gst_pipeline("csi", 640, 480, 30)
        assert isinstance(result, str)
        assert "nvarguscamerasrc" in result
        assert "640" in result
        assert "480" in result

    def test_usb_camera_index(self):
        result = build_gst_pipeline(0)
        assert result == 0

    def test_http_url_passthrough(self):
        url = "http://192.168.1.100:81/stream"
        result = build_gst_pipeline(url)
        assert result == url

    def test_custom_string_passthrough(self):
        custom = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
        result = build_gst_pipeline(custom)
        assert result == custom


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestVideoPipelineUnit:

    def test_test_frame_generation(self):
        from scripts.video_pipeline import VideoPipeline
        vp = VideoPipeline(width=160, height=120)
        frame = vp.generate_test_frame()
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
