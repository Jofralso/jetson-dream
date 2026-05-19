#!/usr/bin/env python3
"""
Download pre-trained fast-neural-style models.

Downloads from PyTorch's example models repository.
These are ~7MB each and run fast on Jetson.
"""

import os
import sys
import urllib.request
from pathlib import Path


# PyTorch fast-neural-style pre-trained models
# Source: https://github.com/pytorch/examples/tree/main/fast_neural_style
STYLE_URLS = {
    "candy": "https://web.eecs.umich.edu/~just101/candy.pth",
    "mosaic": "https://web.eecs.umich.edu/~just101/mosaic.pth",
    "rain_princess": "https://web.eecs.umich.edu/~just101/rain_princess.pth",
    "udnie": "https://web.eecs.umich.edu/~just101/udnie.pth",
}


def download_styles(output_dir: str = "models"):
    """Download all available style models."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading style models to {out}/\n")

    for name, url in STYLE_URLS.items():
        dest = out / f"{name}.pth"
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}.pth ({size_mb:.1f} MB) — already exists")
            continue

        print(f"  ↓ {name}.pth ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Models saved to {out}/")
    print("\nTo train your own styles, see:")
    print("  https://github.com/pytorch/examples/tree/main/fast_neural_style")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "models"
    download_styles(output)
