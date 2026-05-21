#!/usr/bin/env python3
"""
Download pre-trained fast-neural-style models.

Downloads from multiple sources:
  - PyTorch examples repository (canonical models)
  - rrmina/fast-neural-style-pytorch (additional styles)

Standard models are ~7MB each (Johnson TransformNet architecture).
"""

import os
import sys
import urllib.request
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  STYLE MODEL SOURCES
# ═══════════════════════════════════════════════════════════════

# PyTorch fast-neural-style examples (canonical, well-tested)
PYTORCH_STYLES = {
    "candy": "https://web.eecs.umich.edu/~just101/candy.pth",
    "mosaic": "https://web.eecs.umich.edu/~just101/mosaic.pth",
    "rain_princess": "https://web.eecs.umich.edu/~just101/rain_princess.pth",
    "udnie": "https://web.eecs.umich.edu/~just101/udnie.pth",
}

# Additional styles from rrmina/fast-neural-style-pytorch
# These use the same Johnson architecture and are directly compatible
RRMINA_BASE = "https://raw.githubusercontent.com/rrmina/fast-neural-style-pytorch/master/models"
RRMINA_STYLES = {
    "starry_night": f"{RRMINA_BASE}/starry_night_10000.pth",
    "mosaic_v2": f"{RRMINA_BASE}/mosaic_10000.pth",
    "candy_v2": f"{RRMINA_BASE}/candy_10000.pth",
}

# Magenta arbitrary style transfer (TF-based, converted to ONNX)
# These enable ANY style image without retraining
MAGENTA_STYLES = {}  # Placeholder for future ONNX arbitrary style model

# All available styles (merged)
ALL_STYLES = {
    **PYTORCH_STYLES,
    **RRMINA_STYLES,
}

# Style metadata for display
STYLE_INFO = {
    "candy": "Bright candy colors (classic)",
    "mosaic": "Classic mosaic tiles",
    "rain_princess": "Dreamy rain painting",
    "udnie": "Abstract geometric (Francis Picabia)",
    "starry_night": "Van Gogh's Starry Night swirls",
    "mosaic_v2": "Mosaic tiles (variant training)",
    "candy_v2": "Candy colors (variant training)",
}


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Download a single file with progress indication."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ {dest.name} ({size_mb:.1f} MB) — already exists")
        return True

    print(f"  ↓ {dest.name} ... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        return False


def download_styles(output_dir: str = "models", subset: str = "all"):
    """Download style models.

    Args:
        output_dir: Directory to save models
        subset: Which styles to download:
            "core" — only PyTorch canonical (4 models)
            "all"  — all available models
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if subset == "core":
        styles = PYTORCH_STYLES
    else:
        styles = ALL_STYLES

    print(f"Downloading {len(styles)} style models to {out}/\n")

    success = 0
    failed = 0
    for name, url in styles.items():
        dest = out / f"{name}.pth"
        info = STYLE_INFO.get(name, "")
        if download_file(url, dest, info):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed")
    print(f"Models saved to {out}/")

    if failed > 0:
        print(f"\n⚠ {failed} models failed to download.")
        print("  Some URLs may be temporarily unavailable.")
        print("  Re-run this script to retry failed downloads.")

    print("\nTo train your own styles, see:")
    print("  https://github.com/pytorch/examples/tree/main/fast_neural_style")
    print("  https://github.com/rrmina/fast-neural-style-pytorch")


def list_styles():
    """Print available styles and their descriptions."""
    print("\nAvailable styles:\n")
    print(f"  {'Name':<20} {'Source':<12} Description")
    print(f"  {'─'*20} {'─'*12} {'─'*40}")
    for name, info in STYLE_INFO.items():
        source = "pytorch" if name in PYTORCH_STYLES else "rrmina"
        print(f"  {name:<20} {source:<12} {info}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download style transfer models")
    p.add_argument("-o", "--output", default="models", help="Output directory")
    p.add_argument("--subset", choices=["core", "all"], default="all",
                   help="Which models to download (default: all)")
    p.add_argument("--list", action="store_true", help="List available styles")
    args = p.parse_args()

    if args.list:
        list_styles()
    else:
        download_styles(args.output, args.subset)

