# Performance Tuning Guide

## Understanding the FPS Budget

At 30 FPS, each frame has a **33ms budget**. Here's where that time goes:

| Stage | Typical Cost | Notes |
|-------|-------------|-------|
| Camera capture | 1-3ms | Negligible |
| Downscale to process res | <1ms | CPU resize |
| **AI Processing** | **15-150ms** | **THE bottleneck** |
| Upscale to display res | 1-2ms | CPU resize |
| Display + HUD | 1-2ms | OpenCV imshow |

The AI processing stage dominates. All tuning focuses here.

---

## Quick FPS Reference

Expected FPS for DeepDream by hardware and settings:

| Platform | Resolution | Octaves×Iters | Mode | Expected FPS |
|----------|-----------|---------------|------|-------------|
| Jetson Nano | 160×96 | 1×1 (fast) | nano | 15-25 |
| Jetson Nano | 320×180 | 2×3 | turbo | 4-8 |
| Jetson Nano | 480×360 | 3×5 | standard | 1-3 |
| Orin Nano | 320×180 | 2×3 | turbo | 20-30 |
| Orin Nano | 480×270 | 3×5 | turbo | 10-15 |
| Orin Nano | 640×360 | 3×5 | quality | 5-10 |
| Desktop 3090 | 480×270 | 3×5 | turbo | 30+ |

Style Transfer is roughly 5-7× faster than DeepDream at the same resolution (single forward pass, no gradients).

---

## Optimization Strategies (Ranked by Impact)

### 1. Reduce Processing Resolution (Biggest Impact)

AI cost scales **quadratically** with resolution. Halving both dimensions gives 4× speedup.

```bash
# Jetson Nano — use nano preset (160×96)
python scripts/main.py --nano

# Orin Nano — use ultra_fast (320×180)
python scripts/main.py --turbo --process-res ultra_fast
```

The dreamy effect still looks good when upscaled because the hallucinations are resolution-independent patterns.

### 2. Use Nano Mode / Fast Dream (3-6× Faster)

Nano mode replaces the full multi-octave pipeline with a single gradient step:

```bash
python scripts/main.py --nano
```

| Feature | Standard Dream | Nano Fast Dream |
|---------|---------------|-----------------|
| Octaves | 2-4 | 1 (none) |
| Iterations per octave | 3-10 | 1 |
| Backward passes per frame | 6-40 | 1 |
| Octave pyramid | Yes | No |
| Effect strength | Strong | Subtle/medium |

To get stronger effects with fast dream, increase `intensity` (0.03-0.05) or `feedback` (0.5-0.8).

### 3. Frame Skipping (Linear FPS Multiplier)

Process every Nth frame, reuse the previous result for skipped frames:

```bash
# Process every other frame (2× effective FPS)
python scripts/main.py --nano --frame-skip 1

# Process every 3rd frame (3× effective FPS)
python scripts/main.py --nano --frame-skip 2

# Process every 4th frame
python scripts/main.py --turbo --frame-skip 3
```

With `feedback > 0`, frame skipping is nearly invisible because the dream effect accumulates over time. Higher feedback values mask the skipped frames better.

### 4. Target Shallower Layers (Faster Forward/Backward Pass)

The InceptionV3 forward pass exits early at the target layer. Shallower = faster:

| Layer | Index | Compute Cost | Visual Effect |
|-------|-------|-------------|---------------|
| Conv2d_1a_3x3 | 0 | Cheapest | Subtle edges |
| Conv2d_4a_3x3 | 3 | ~3× of layer 0 | Complex textures |
| Mixed_5b | 4 | ~5× of layer 0 | Eyes, spirals (default) |
| Mixed_6a | 7 | ~10× of layer 0 | Abstract forms |
| Mixed_7a | 9 | ~15× of layer 0 | Full hallucinations |

Nano mode defaults to layer 3 (Conv2d_4a_3x3) for this reason.

To change at runtime: press pads on Row 8 (top row) of the Launchpad, or set `dream.layer_index = 2` programmatically.

### 5. Enable Async Pipeline (Smooth Display)

The async pipeline doesn't increase AI FPS, but decouples display from processing so the UI stays smooth:

```bash
python scripts/main.py --turbo   # async enabled by default
python scripts/main.py --turbo --no-async   # disable for comparison
```

### 6. Use Style Transfer Instead of DeepDream

Style transfer is inherently faster (no gradients):

```bash
python scripts/main.py --turbo --mode style
```

Or press `2` during runtime to switch to style mode.

### 7. Enable TensorRT (Style Transfer Only)

TensorRT gives 2-4× additional speedup for style transfer:

```bash
# Check if TensorRT is available
python -c "import tensorrt; import torch_tensorrt; print('TRT ready')"

# Turbo mode auto-enables TensorRT when available
python scripts/main.py --turbo --mode style
```

### 8. Maximize Jetson Clocks

```bash
# Set maximum performance mode
sudo nvpmodel -m 0

# Lock clocks at maximum
sudo jetson_clocks

# Verify
sudo jetson_clocks --show
```

This alone can provide 20-50% improvement on Jetson platforms.

---

## Platform-Specific Recommendations

### Jetson Nano (4GB)

The original Jetson Nano has limited GPU (128 Maxwell cores, ~472 GFLOPS FP16). Use aggressive optimizations:

```bash
# Recommended command
python scripts/main.py --nano -c 0 -f

# Breakdown of what --nano does:
#   --turbo           → async pipeline + FP16
#   --process-res nano → 160×96 processing
#   process_frame_fast → single gradient step
#   --frame-skip 1    → process every other frame
#   layer_index=3     → shallow layer (faster)
#   display 640×480   → reasonable output
```

**Additional Nano tips:**
- Use style transfer mode (`--mode style`) for smoothest experience
- Close other GPU processes (`sudo tegrastats` to check)
- Use a USB camera at 320×240 native to reduce capture overhead
- Consider using the PS3 Eye at 320×240 for lowest capture latency

### Jetson Orin Nano (8GB)

The Orin Nano has much more GPU power (1024 Ampere cores, ~40 TOPS INT8). Default turbo settings work well:

```bash
# Default turbo (480×270 processing, 720p display)
python scripts/main.py --turbo -f

# For shows/performances (720p + best quality)
python scripts/main.py --turbo --process-res quality -c csi -f

# Maximum FPS for DeepDream
python scripts/main.py --turbo --process-res ultra_fast -f
```

### Desktop GPU (RTX 3060+)

Desktop GPUs are fast enough for quality mode:

```bash
# Full quality
python scripts/main.py --turbo --process-res native -f

# Or just standard mode (no downscaling needed)
python scripts/main.py -W 640 -H 480 -f
```

---

## Profiling and Diagnostics

### Built-in Profiler

Press `P` during turbo mode to print a timing breakdown:

```
Performance (18.2 FPS):
  capture             :    2.1ms
  process             :   48.3ms
  display             :    1.8ms
  budget              :  -19.2ms @ 30fps
```

A negative budget means you're over the 33ms frame budget for 30 FPS.

### Python Profiling

```bash
# Profile the main loop
python -m cProfile -s cumulative scripts/main.py --turbo --no-camera 2>&1 | head -40

# Line-by-line profiling of dream engine
pip install line_profiler
kernprof -l -v scripts/dream_engine.py
```

### GPU Monitoring

```bash
# Jetson GPU utilization
sudo tegrastats

# Desktop NVIDIA GPU
nvidia-smi dmon -s u
watch -n 0.5 nvidia-smi
```

---

## Tuning Parameters for Visual Quality vs Speed

### "Looks Good, Runs Fast" Settings

```python
# Strong effect despite single iteration
dream.intensity = 0.04      # Higher intensity compensates for fewer iterations
dream.feedback = 0.6        # High feedback accumulates dream over time
dream.layer_index = 3       # Cheap layer, still produces patterns
dream.zoom = 1.003          # Zoom enhances recursive dream effect
dream.hue_shift = 0.01      # Subtle color cycling adds interest
```

### "Maximum Dream, Damn the FPS" Settings

```python
dream.layer_index = 7       # Deep layer = complex hallucinations
dream.intensity = 0.03      # Moderate intensity
dream.octaves = 4           # Rich multi-scale detail
dream.iterations = 10       # Many gradient steps
dream.feedback = 0.4        # Moderate recursion
```

---

## Troubleshooting Low FPS

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| <3 FPS on Nano | Default settings too heavy | Use `--nano` |
| 6 FPS in turbo ultra_fast | DeepDream octave/iter overhead | Use `--nano` (single-pass fast dream) |
| Style transfer slow (<15 FPS) | TensorRT not available | Install `torch-tensorrt` or reduce resolution |
| FPS drops over time | Thermal throttling | Check `tegrastats`, add cooling |
| Sporadic stutter | GC pauses or background processes | Set `--frame-skip 1`, close other apps |
| Async pipeline doesn't help | Processing is the sole bottleneck | Expected — async smooths display, doesn't speed AI |
