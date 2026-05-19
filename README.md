# jetson-dream

**Live AI video dreaming on Jetson Orin Nano, controlled by a Novation Launchpad MK3.**

Transforms live camera into real-time hallucinatory visuals using classic AI models —
DeepDream (InceptionV3) and fast Neural Style Transfer — with every parameter
controlled by your Launchpad's 64 RGB pads.

## How It Works

```
┌──────────────── Jetson Orin Nano Super ─────────────────┐
│                                                          │
│  [Camera]         → Video Pipeline → frame ──────┐      │
│  [Launchpad MK3]  → MIDI Engine   → pad state ──┤      │
│                                                   │      │
│                                            Param Mapper  │
│                                                   │      │
│                                      ┌────────────┤      │
│                                      ▼            ▼      │
│                                 DeepDream    Style       │
│                                (InceptionV3) Transfer    │
│                                      │            │      │
│                                      └─── mix ────┘      │
│                                            │             │
│                                     Display / Projector  │
└──────────────────────────────────────────────────────────┘
```

## AI Models

| Model | Effect | Speed (480×360) |
|---|---|---|
| **DeepDream** (InceptionV3) | Psychedelic hallucinations — eyes, spirals, animal faces emerge from the image | 5-15 FPS |
| **Fast Neural Style Transfer** | Painterly styles (Van Gogh, Mosaic, Candy, etc.) applied in real-time | 15-30 FPS |
| **Blend mode** | Both engines mixed together for maximum dreaminess | 3-10 FPS |

Lower resolutions = faster. At 320×240 you can push 15+ FPS even in Dream mode.

## Launchpad MK3 Layout

The 8×8 pad grid is mapped for intuitive live control:

```
┌──────────────────────────────────────────────┐
│  Top buttons:                                 │
│  [▲] [▼] [◄] [►] [Ses] [Drm] [Key] [Usr]   │
│  prev next  -   + reset freeze HUD  F/S     │
│  mode mode oct oct                           │
├──────────────────────────────────────────────┤
│                                              │
│  Row 8  LAYER SELECT     dream layer 0-7    │
│  Row 7  STYLE SELECT     style model 0-7    │
│  Row 6  INTENSITY        dream power / blend │
│  Row 5  FEEDBACK         frame recursion     │
│  Row 4  HUE SHIFT        color rotation      │
│  Row 3  ITERATIONS/BLUR  detail / softness   │
│  Row 2  ZOOM SPEED       infinite zoom rate   │
│  Row 1  PRESETS          8 instant presets    │
│                                              │
└──────────────────────────────────────────────┘
```

### Presets (Row 1)

| Pad | Name | Description |
|---|---|---|
| 1 | Subtle | Gentle dream, low feedback |
| 2 | Deep | Full dream with rich detail |
| 3 | Psychedelic | Maximum hallucination + hue cycling |
| 4 | Gentle Style | Light style transfer overlay |
| 5 | Full Style | Full painterly style |
| 6 | Dream+Style | Both engines blended |
| 7 | Inf. Zoom | Classic deep dream zoom |
| 8 | Chaos | Everything maxed out |

## Quick Start

### 1. Setup

```bash
cd jetson-dream
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

### 2. Download Style Models

```bash
python3 scripts/download_styles.py
```

### 3. Run

```bash
# USB camera + auto-detect Launchpad
python3 scripts/main.py

# Jetson CSI camera + fullscreen (for projector)
python3 scripts/main.py -c csi -f

# No camera (test pattern) — good for testing MIDI
python3 scripts/main.py --no-camera

# Lower resolution for faster dreaming
python3 scripts/main.py -W 320 -H 240

# Start in style transfer mode
python3 scripts/main.py --mode style

# Without MIDI (keyboard only)
python3 scripts/main.py --no-midi

# List devices
python3 scripts/main.py --list-devices
```

### 4. Keyboard Controls

| Key | Action |
|---|---|
| `1` | DeepDream mode |
| `2` | Style Transfer mode |
| `3` | Blend mode (both) |
| `F` | Toggle fullscreen |
| `H` | Toggle HUD overlay |
| `Space` | Freeze frame |
| `+` / `-` | Increase / decrease dream intensity |
| `R` | Reset all parameters |
| `Q` / `ESC` | Quit |

## Hardware Requirements

- **Jetson Orin Nano Super 8GB** (or any Jetson with CUDA)
- **Novation Launchpad MK3** (USB, auto-detected)
- **USB camera** or **Jetson CSI camera**
- **Display** (HDMI/DP) or projector

## Dependencies

- Python 3.10+
- PyTorch (NVIDIA Jetson wheel or standard)
- torchvision (InceptionV3 weights)
- OpenCV (camera + display)
- python-rtmidi (MIDI input/output)
- numpy

## Architecture

```
jetson-dream/
├── scripts/
│   ├── main.py              # Main loop — camera → AI → display
│   ├── midi_launchpad.py    # Launchpad MK3 MIDI I/O + LED feedback
│   ├── dream_engine.py      # DeepDream (InceptionV3 gradient ascent)
│   ├── style_engine.py      # Fast Neural Style Transfer
│   ├── video_pipeline.py    # Camera capture + display output
│   ├── param_mapper.py      # Launchpad pads → AI parameters
│   ├── download_styles.py   # Download pre-trained style models
│   └── setup.sh             # Install dependencies
├── models/                  # Style transfer .pth files
├── styles/                  # Style reference images
└── tests/
    ├── test_midi_launchpad.py
    ├── test_param_mapper.py
    ├── test_dream_engine.py
    └── test_video_pipeline.py
```

## Performance Tips

- **Resolution matters most**: 320×240 runs 2-3× faster than 640×480
- **Reduce octaves**: 2 octaves is much faster than 4, still looks dreamy
- **Reduce iterations**: 3-5 iterations per octave is enough for live use
- **Style Transfer is faster** than DeepDream — use it when you need more FPS
- **Feedback** is free (just alpha blending) and adds a lot of visual depth
- **TensorRT**: Convert InceptionV3 to TensorRT for ~2× speedup on Jetson

## Training Custom Styles

You can train your own style transfer models using PyTorch's
[fast-neural-style example](https://github.com/pytorch/examples/tree/main/fast_neural_style).
Place the resulting `.pth` files in the `models/` directory.
