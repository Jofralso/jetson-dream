#!/usr/bin/env bash
# jetson-dream setup script
# Installs dependencies for live AI video dreaming on Jetson Orin Nano
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  jetson-dream — Setup"
echo "═══════════════════════════════════════════════"

# Detect if we're on Jetson
IS_JETSON=false
if [ -f /etc/nv_tegra_release ] || [ -d /usr/src/jetson_multimedia_api ]; then
    IS_JETSON=true
    echo "Detected: NVIDIA Jetson platform"
else
    echo "Detected: Generic Linux (not Jetson)"
fi

echo ""
echo "[1/4] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-opencv \
    libasound2-dev \
    libjack-dev

echo ""
echo "[2/4] Python packages..."
pip3 install --upgrade pip

# Core dependencies
pip3 install \
    numpy \
    opencv-python \
    python-rtmidi \
    Pillow

# PyTorch — Jetson uses NVIDIA's pre-built wheels
if [ "$IS_JETSON" = true ]; then
    echo ""
    echo "  Jetson detected — using NVIDIA PyTorch wheels"
    echo "  If PyTorch is not installed, follow:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
    echo ""
    # Check if torch is already installed
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
        echo "  PyTorch already installed ✓"
    else
        echo "  ⚠ PyTorch not found — install NVIDIA's Jetson wheel:"
        echo "    pip3 install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/"
    fi
    pip3 install torchvision 2>/dev/null || echo "  Install torchvision from NVIDIA's Jetson wheels"
else
    # Desktop — standard PyTorch
    pip3 install torch torchvision
fi

echo ""
echo "[3/4] Download InceptionV3 weights (for DeepDream)..."
python3 -c "
from torchvision import models
print('Downloading InceptionV3...')
models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
print('InceptionV3 cached ✓')
"

echo ""
echo "[4/4] Style transfer models..."
MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

echo "  Style models directory: $MODELS_DIR"
echo ""
echo "  To download pre-trained style models, run:"
echo "    python3 scripts/download_styles.py"
echo ""
echo "  Or manually place .pth files in $MODELS_DIR/"
echo "  Compatible models: Johnson et al. fast-neural-style format"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Quick start:"
echo "    cd jetson-dream"
echo "    python3 scripts/main.py              # with camera"
echo "    python3 scripts/main.py --no-camera  # test pattern"
echo "    python3 scripts/main.py -c csi -f    # Jetson CSI + fullscreen"
echo ""
echo "  Connect your Launchpad MK3 via USB — it auto-detects."
echo "═══════════════════════════════════════════════"
