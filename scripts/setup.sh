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
    TORCH_VERSION=""
    if python3 -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
        echo "  PyTorch already installed ✓"
        # Extract torch version for matching torchvision
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
    else
        echo "  ⚠ PyTorch not found — install NVIDIA's Jetson wheel:"
        echo "    pip3 install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/"
    fi
    
    # Install compatible torchvision
    if [ ! -z "$TORCH_VERSION" ]; then
        echo ""
        echo "  Installing compatible torchvision for torch $TORCH_VERSION..."
        # Determine torchvision version matching torch
        case "$TORCH_VERSION" in
            2.11*) TV_VERSION="0.26.0" ;;
            2.12*) TV_VERSION="0.27.0" ;;
            2.10*) TV_VERSION="0.25.0" ;;
            *) TV_VERSION="0.27.0" ;;  # Default to latest compatible
        esac
        
        echo "  Installing torchvision==$TV_VERSION (matching torch $TORCH_VERSION)..."
        pip3 install "torchvision==$TV_VERSION" 2>/dev/null && \
            echo "  ✓ torchvision $TV_VERSION installed" || \
            echo "  ⚠ Try installing from NVIDIA's Jetson wheels if pip install fails"
    fi
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
mkdir -p "$MODELS_DIR/trt_cache"

echo "  Style models directory: $MODELS_DIR"
echo ""
echo "  To download pre-trained style models, run:"
echo "    python3 scripts/download_styles.py"
echo ""
echo "  Or manually place .pth files in $MODELS_DIR/"
echo "  Compatible models: Johnson et al. fast-neural-style format"

# ── TensorRT (optional, for turbo mode) ──
echo ""
echo "[bonus] TensorRT check..."
if [ "$IS_JETSON" = true ]; then
    if python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__} ✓')" 2>/dev/null; then
        echo "  ✓ TensorRT found"
        # Install torch-tensorrt if not present
        if python3 -c "import torch_tensorrt" 2>/dev/null; then
            echo "  ✓ torch-tensorrt found — turbo mode will use TensorRT acceleration (2×+ speedup)"
        else
            echo "  Installing torch-tensorrt for turbo mode..."
            if pip3 install torch-tensorrt 2>/dev/null; then
                echo "  ✓ torch-tensorrt installed — turbo mode ready"
            else
                echo "  ⚠ torch-tensorrt install failed — turbo mode will use FP16 fallback"
                echo "    To retry: pip3 install torch-tensorrt"
            fi
        fi
    else
        echo "  ⚠ TensorRT not found — turbo mode will use torch.compile or FP16"
        echo ""
        echo "  To enable TensorRT acceleration for 2× style transfer speedup:"
        echo "    1. Check if TensorRT is in JetPack SDK:"
        echo "       ls /usr/lib/aarch64-linux-gnu/libnvinfer.so"
        echo ""
        echo "    2a. If found — install Python bindings:"
        echo "        pip3 install torch-tensorrt"
        echo ""
        echo "    2b. If not found — install full JetPack SDK:"
        echo "        sudo apt-get install nvidia-jetpack"
        echo ""
        echo "  After install, rerun: python3 -c 'import tensorrt; import torch_tensorrt'"
    fi
else
    echo "  (TensorRT only available on Jetson — skipping)"
fi

# ── Jetson power mode optimization ──
if [ "$IS_JETSON" = true ]; then
    echo ""
    echo "[turbo] Jetson power optimization..."
    echo "  For maximum 720p@30fps performance:"
    echo "    sudo nvpmodel -m 0              # MAX performance mode"
    echo "    sudo jetson_clocks               # Lock clocks to max"
    echo "    sudo jetson_clocks --fan          # Max fan speed"
    echo ""
    echo "  Check current mode: nvpmodel -q"
fi

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
echo "  ── TURBO MODE (720p@30fps) ──"
echo "    python3 scripts/main.py --turbo                    # 720p auto-tuned"
echo "    python3 scripts/main.py --turbo -c csi -f          # CSI + fullscreen"
echo "    python3 scripts/main.py --turbo --process-res quality  # Higher detail"
echo ""
echo "  Connect your Launchpad MK3 via USB — it auto-detects."
echo "═══════════════════════════════════════════════"
