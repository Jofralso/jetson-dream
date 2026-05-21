# Setup Guide

## Prerequisites

### Hardware

| Component | Required | Recommended |
|-----------|----------|-------------|
| NVIDIA Jetson (any with CUDA) | Yes | Orin Nano 8GB |
| Camera | Yes | USB UVC or CSI |
| Display | Yes | HDMI/DP monitor or projector |
| Novation Launchpad MK3 | No (keyboard fallback) | Yes |
| USB hub | No | If using camera + Launchpad |

### Software

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| torchvision | 0.15+ | InceptionV3 pre-trained model |
| OpenCV | 4.5+ | Camera I/O, image processing |
| python-rtmidi | 1.5+ | MIDI controller communication |
| numpy | 1.20+ | Array operations |

**Optional (recommended):**

| Dependency | Purpose |
|-----------|---------|
| TensorRT + torch-tensorrt | 3-5× faster style transfer |
| CUDA 11.4+ | GPU acceleration |
| GStreamer | CSI camera pipeline on Jetson |

---

## Installation

### Automated Setup (Jetson or Desktop Linux)

```bash
git clone https://github.com/Jofralso/jetson-dream.git
cd jetson-dream
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

The setup script:
1. Installs system packages (python3, opencv, ALSA/JACK dev headers)
2. Installs Python packages (numpy, opencv-python, python-rtmidi, Pillow)
3. Installs PyTorch (auto-detects Jetson vs desktop)
4. Downloads InceptionV3 pre-trained weights
5. Creates model directories

### Manual Setup

```bash
# System dependencies
sudo apt-get install python3-pip python3-dev python3-numpy \
    python3-opencv libasound2-dev libjack-dev

# Python packages
pip3 install numpy opencv-python python-rtmidi Pillow

# PyTorch — choose one:
# Desktop:
pip3 install torch torchvision

# Jetson (use NVIDIA wheel):
pip3 install torch torchvision --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/
```

### Download Style Models

```bash
python3 scripts/download_styles.py
```

Downloads 4 pre-trained style transfer models (~7MB each) to `models/`:
- candy.pth
- mosaic.pth
- rain_princess.pth
- udnie.pth

Additional styles (starry_night, la_muse, the_scream, feathers) can be trained from reference images using [PyTorch fast-neural-style](https://github.com/pytorch/examples/tree/main/fast_neural_style).

---

## Camera Setup

### USB Camera

Plug in any UVC-compatible USB camera. Auto-detected at index 0.

```bash
# Verify detection
python scripts/main.py --list-devices

# Run with USB camera
python scripts/main.py -c 0
```

### PS3 Eye Camera

The PS3 Eye offers wide FOV and high frame rates:

```bash
# Verify PS3 Eye is detected
lsusb | grep "1415:2000"

# Auto-detect and run
python scripts/main.py -c ps3eye
```

**Modes:**
- 640×480 @ 60fps (default)
- 320×240 @ 187fps (set with `-W 320 -H 240`)

### Jetson CSI Camera

For cameras connected to the Jetson's CSI connector:

```bash
python scripts/main.py -c csi
```

Uses `nvarguscamerasrc` via GStreamer with hardware-accelerated color conversion.

### No Camera (Test Mode)

```bash
python scripts/main.py --no-camera
```

Generates a synthetic test pattern. Useful for testing MIDI control or AI processing without a physical camera.

---

## Launchpad MK3 Setup

### Connection

1. Connect Launchpad MK3 via USB
2. jetson-dream auto-detects it on startup
3. The Launchpad switches to Programmer Mode automatically

### Verification

```bash
python scripts/main.py --list-devices
# Should show: "Found Launchpad input: [N] Launchpad MK3:..."
```

### Troubleshooting MIDI

| Issue | Solution |
|-------|----------|
| "Launchpad MK3 not found" | Check USB connection, try different port |
| "ALSA sequencer not available" | Running over SSH — use `--no-midi` |
| "python-rtmidi not installed" | `pip3 install python-rtmidi` |
| Multiple MIDI devices | Use `--midi-port "Launchpad"` to filter |

---

## TensorRT Setup (Optional)

TensorRT provides 2-4× additional speedup for style transfer.

### Check Availability

```bash
python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
python3 -c "import torch_tensorrt; print('torch-tensorrt available')"
```

### Install on Jetson (JetPack)

```bash
# If JetPack SDK is installed, TensorRT libraries are already present
# Just install the Python bindings:
pip3 install torch-tensorrt
```

### Install on Desktop

```bash
pip3 install torch-tensorrt
```

### Without TensorRT

The system gracefully falls back to:
1. `torch.compile` + FP16 (if PyTorch 2.0+)
2. Vanilla FP16 inference
3. Standard FP32 inference

---

## Jetson Power Configuration

For maximum performance on Jetson:

```bash
# Set maximum performance mode (0 = MAXN on most Jetsons)
sudo nvpmodel -m 0

# Lock GPU/CPU/EMC clocks at maximum
sudo jetson_clocks

# Verify clock speeds
sudo jetson_clocks --show
```

**Important:** Maximum performance mode increases power consumption and heat. Ensure adequate cooling.

---

## Display Setup

### Windowed Mode (Default)

```bash
python scripts/main.py
```

### Fullscreen

```bash
python scripts/main.py -f
# Or press F during runtime
```

### Projector/External Display

```bash
# Fullscreen on external display (Jetson HDMI)
python scripts/main.py -c csi -f --turbo
```

### Network Streaming (MJPEG)

For remote viewing without a local display, the `MJPEGServer` class in `streaming.py` provides HTTP-based MJPEG output accessible from any browser at `http://<jetson-ip>:8080/`.

---

## Verification

After setup, run the test suite:

```bash
cd jetson-dream
python -m pytest tests/ -v
```

Quick smoke test (no camera, no MIDI):

```bash
python scripts/main.py --no-camera --no-midi
# Should show a window with a test pattern being processed
# Press Q to quit
```

---

## Directory Structure After Setup

```
jetson-dream/
├── models/
│   ├── candy.pth           # Style models (~7MB each)
│   ├── mosaic.pth
│   ├── rain_princess.pth
│   ├── udnie.pth
│   └── trt_cache/          # TensorRT engine cache (auto-generated)
├── scripts/                 # Source code
├── styles/                  # Style reference images
├── tests/                   # Unit tests
├── docs/                    # Documentation (you are here)
└── demo_output/             # Demo output frames
```
