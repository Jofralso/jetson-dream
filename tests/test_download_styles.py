#!/usr/bin/env python3
"""Tests for the style download module."""

import pytest
from scripts.download_styles import (
    ALL_STYLES,
    PYTORCH_STYLES,
    RRMINA_STYLES,
    STYLE_INFO,
    download_file,
)


class TestStyleRegistry:
    """Test style registry completeness."""

    def test_all_styles_non_empty(self):
        assert len(ALL_STYLES) >= 7

    def test_core_styles_present(self):
        for name in ["candy", "mosaic", "rain_princess", "udnie"]:
            assert name in ALL_STYLES

    def test_extended_styles_present(self):
        for name in ["starry_night"]:
            assert name in ALL_STYLES

    def test_all_styles_have_urls(self):
        for name, url in ALL_STYLES.items():
            assert url.startswith("http"), f"{name} has invalid URL"
            assert url.endswith(".pth"), f"{name} URL doesn't end in .pth"

    def test_all_styles_have_info(self):
        for name in ALL_STYLES:
            assert name in STYLE_INFO, f"{name} missing description"

    def test_no_duplicate_urls(self):
        urls = list(ALL_STYLES.values())
        # Allow variants (same URL under different names is fine)
        # but detect exact duplicates
        assert len(urls) == len(set(urls))


class TestDownloadFile:
    """Test download logic (mocked, no network)."""

    def test_existing_file_skipped(self, tmp_path):
        dest = tmp_path / "exists.pth"
        dest.write_bytes(b"fake model data " * 100)
        result = download_file("http://example.com/model.pth", dest)
        assert result is True
