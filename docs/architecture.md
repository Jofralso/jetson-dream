# Architecture Guide

## System Overview

jetson-dream is a real-time AI video processing system that transforms live camera feeds into hallucinatory visuals using DeepDream and Neural Style Transfer, controlled by a Novation Launchpad MK3 MIDI controller. It is optimized for NVIDIA Jetson platforms (Orin Nano, Nano) but runs on any CUDA-capable Linux system.

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Camera (USB/CSI/PS3Eye)     Launchpad MK3 (USB MIDI)               │
│         │                             │                              │
│         ▼                             ▼                              │
│  ┌─────────────────┐         ┌────────────────────┐                 │
│  │  VideoPipeline   │         │  LaunchpadMidi      │                │
│  │  (video_pipeline)│         │  (midi_launchpad)   │                │
│  └────────┬────────┘         └────────┬───────────┘                 │
│           │ BGR frames                │ pad/button state             │
│           ▼                           ▼                              │
│  ┌──────────────────────────────────────────────────┐                │
│  │             AsyncPipeline (turbo mode)            │                │
│  │  ┌──────────┐   ┌───────────┐   ┌────────────┐  │                │
│  │  │ Capture  │──▶│  Process  │──▶│  Display   │  │                │
│  │  │ Thread   │   │  Thread   │   │  Thread    │  │                │
│  │  └──────────┘   └─────┬─────┘   └────────────┘  │                │
│  └────────────────────────┼─────────────────────────┘                │
│                           │                                          │
│           ┌───────────────┼───────────────────┐                      │
│           ▼               ▼                   ▼                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ ParamMapper  │  │ DeepDream    │  │ Style        │                │
│  │              │  │ Engine       │  │ Transfer     │                │
│  │ (MIDI→params)│  │ (InceptionV3)│  │ (TransformNet)│               │
│  └─────────────┘  └──────┬───────┘  └──────┬───────┘                │
│                           │                  │                        │
│                           ▼                  ▼                        │
│                    ┌────────────────────────────────┐                 │
│                    │  TurboEngine (TensorRT/FP16)   │                 │
│                    │  ProcessingResolutionManager   │                 │
│                    └────────────────────────────────┘                 │
│                                                                      │
│                           ▼                                          │
│                    Display / Projector                                │
└──────────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
main.py
├── video_pipeline.py        # Camera I/O, display, GStreamer
├── dream_engine.py          # DeepDream (InceptionV3 gradient ascent)
│   └── torchvision.models   # Pre-trained InceptionV3
├── style_engine.py          # Fast Neural Style Transfer
│   └── turbo_engine.py      # TensorRT/FP16 acceleration
├── async_pipeline.py        # Threaded capture → process → display
├── midi_launchpad.py        # Launchpad MK3 MIDI I/O
│   └── python-rtmidi        # Low-level MIDI access
├── param_mapper.py          # MIDI state → AI engine parameters
│   └── midi_launchpad.py    # Color constants, grid helpers
└── turbo_engine.py          # Resolution manager, profiler, TRT engines
```

## Processing Modes

### Standard Mode (Default)

Single-threaded synchronous loop:

```
while running:
    frame = camera.read()
    output = engine.process(frame)
    display.show(output)
    handle_input()
```

Suitable for low resolutions (320×240 – 480×360) where processing keeps up with the camera frame rate.

### Turbo Mode (`--turbo`)

Three-thread asynchronous pipeline that decouples capture, AI processing, and display:

| Thread | Responsibility | Rate |
|--------|---------------|------|
| **CaptureThread** | Reads camera frames into FrameBuffer | Camera FPS (30-60) |
| **ProcessThread** | Downscales → AI engine → upscales | AI throughput (5-30 FPS) |
| **DisplayThread** (main) | Reads latest processed frame, renders HUD | Monitor refresh |

Key benefit: The display always shows the most recent processed frame, so the UI feels smooth even when AI processing is slower than 30 FPS.

### Nano Mode (`--nano`)

Maximizes throughput on Jetson Nano's limited GPU:

- Processing resolution: 160×96 (15,360 pixels vs 921,600 at 720p = **60× fewer**)
- Single-pass dream: 1 gradient step, no octave pyramid (**6× fewer backward passes**)
- Frame skipping: Reuses cached result every other frame (**2× effective FPS**)
- Shallow target layer: Stops forward pass early (**~3× less computation**)

## AI Engines

### DeepDream (dream_engine.py)

Implements multi-octave gradient ascent on InceptionV3 activations.

**Algorithm:**
1. Build octave pyramid (downscaled versions of input)
2. For each octave (coarse → fine):
   a. Resize input to octave resolution
   b. Apply random jitter (reduces tiling)
   c. Forward pass through InceptionV3 up to target layer
   d. Compute L2 norm of activations as loss
   e. Backpropagate → get gradient w.r.t. input
   f. Add normalized gradient × intensity to input
   g. Extract detail (difference from original at this scale)
3. Accumulate details across octaves
4. Apply post-processing (hue shift, blur, feedback)

**Cost:** O(octaves × iterations × forward_pass_cost). Each iteration requires a full backward pass through the network up to the target layer.

### Style Transfer (style_engine.py)

Uses pre-trained feed-forward transformation networks (Johnson et al. 2016).

**Algorithm:**
1. Single forward pass through TransformNet (3 downsampling → 5 residual → 2 upsampling → output)
2. Blend styled result with original (based on style_blend parameter)
3. Apply saturation, contrast, hue adjustments

**Cost:** O(1 forward pass). No gradients needed. ~7× faster than DeepDream at the same resolution.

### Blend Mode

Runs both engines on the same frame and alpha-blends the results:
```python
output = alpha * dream_result + (1-alpha) * style_result
```

## Acceleration Stack

The system applies optimizations in priority order:

| Priority | Method | Speedup | Applicability |
|----------|--------|---------|---------------|
| 1 | TensorRT FP16 | 3-5× | Style Transfer (no gradients) |
| 2 | torch.compile + reduce-overhead | 1.5-2× | Style Transfer fallback |
| 3 | AMP (torch.amp.autocast) | 1.5-2× | DeepDream gradient computation |
| 4 | FP16 model weights | ~2× | All inference |
| 5 | Resolution downscaling | N² | Both engines (quadratic with pixels) |
| 6 | Frame skipping | N× | DeepDream (reuse cached output) |

## Resolution Presets

| Preset | Resolution | Pixels | Relative Cost | Target Use Case |
|--------|-----------|--------|---------------|-----------------|
| `nano` | 160×96 | 15K | 1× | Jetson Nano maximum FPS |
| `nano_fast` | 213×120 | 26K | 1.7× | Jetson Nano balanced |
| `ultra_fast` | 320×180 | 58K | 3.8× | Orin Nano fast |
| `fast` | 426×240 | 102K | 6.6× | Orin Nano balanced |
| `balanced` | 480×270 | 130K | 8.4× | Orin Nano default |
| `quality` | 640×360 | 230K | 15× | Best visual quality |
| `native` | Display res | Varies | Max | No downscaling |

## Thread Safety

- **FrameBuffer**: Single-frame buffer with lock + event. Writers overwrite; readers get latest.
- **LaunchpadMidi**: Callback-based MIDI with lock-protected state. `get_state()` returns snapshot and clears triggers.
- **ProcessThread**: `update_process_fn()` is a simple attribute swap (Python GIL provides atomicity for reference assignment).

## Camera Backend

The `VideoPipeline` supports multiple camera sources through OpenCV:

| Source | Backend | Notes |
|--------|---------|-------|
| USB camera | V4L2 | Standard UVC, index-based |
| CSI camera | GStreamer (nvarguscamerasrc) | Jetson hardware pipeline |
| PS3 Eye | V4L2 | 320×240@187fps or 640×480@60fps |
| IP camera | HTTP/RTSP | URL-based |

In turbo mode, GStreamer pipelines use `drop=1 sync=0 max-buffers=2` to minimize latency.
