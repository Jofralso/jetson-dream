# API Reference

Complete reference for all public classes and methods in the jetson-dream codebase.

---

## dream_engine.py

### `DeepDreamEngine`

Real-time DeepDream processor for video frames using PyTorch InceptionV3.

#### Constructor

```python
DeepDreamEngine(resolution: tuple[int, int] = (640, 480), turbo: bool = False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | `(width, height)` | `(640, 480)` | Processing resolution for AI inference |
| `turbo` | `bool` | `False` | Enable AMP FP16 for faster gradient computation |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `target_layer` | `str` | Current InceptionV3 layer name being targeted |
| `device` | `torch.device` | CUDA device (or CPU fallback) |

#### Controllable Parameters (Attributes)

| Attribute | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `layer_index` | `int` | 0–9 | 4 | InceptionV3 layer depth. 0=edges, 4=eyes/spirals, 9=hallucinations |
| `intensity` | `float` | 0.001–0.1 | 0.02 | Gradient ascent step size (dream strength) |
| `octaves` | `int` | 1–6 | 3 | Resolution scales in the pyramid |
| `octave_scale` | `float` | 1.1–2.0 | 1.4 | Downscale factor between octaves |
| `iterations` | `int` | 1–15 | 5 | Gradient ascent steps per octave |
| `feedback` | `float` | 0.0–1.0 | 0.3 | Previous frame bleed (0=independent, 1=full recursion) |
| `zoom` | `float` | 1.0–1.02 | 1.002 | Zoom-in factor per frame |
| `jitter` | `int` | 0–32 | 16 | Random pixel shift to reduce tiling |
| `hue_shift` | `float` | 0.0–0.1 | 0.0 | Color hue rotation per frame |
| `blur_amount` | `float` | 0.0–1.0 | 0.0 | Post-processing Gaussian blur |
| `frame_skip` | `int` | 0–5 | 0 | Skip N frames between processing (0=process every frame) |

#### Methods

##### `process_frame(frame: np.ndarray) -> np.ndarray`

Full multi-octave DeepDream processing pipeline.

- **Input**: BGR uint8 numpy array from camera
- **Output**: Dreamed BGR uint8 frame
- **Cost**: `octaves × iterations` backward passes through InceptionV3
- **In turbo mode**: Caps at 2 octaves, 3 iterations
- **Frame skip**: Returns cached result for skipped frames

##### `process_frame_fast(frame: np.ndarray) -> np.ndarray`

Ultra-fast single-pass dream: 1 gradient step, no octave pyramid.

- **Input**: BGR uint8 numpy array
- **Output**: Lightly dreamed BGR uint8 frame
- **Cost**: 1 backward pass through InceptionV3 (up to target layer)
- **Use case**: Jetson Nano where even turbo mode is too slow
- **Effect**: Lighter/subtler dream but 3-5× faster than full pipeline

##### `update_params(params: dict)`

Bulk-update dream parameters from a dictionary (typically from MIDI mapper).

Recognized keys: `layer_index`, `intensity`, `octaves`, `iterations`, `feedback`, `zoom`, `jitter`, `hue_shift`, `blur_amount`, `frame_skip`

##### `get_info() -> dict`

Returns current engine state as a dictionary for HUD overlay display.

---

### `InceptionDreamModel(nn.Module)`

Wrapper around InceptionV3 that returns activations at a chosen layer. Used internally by `DeepDreamEngine`.

```python
forward(x: torch.Tensor, target_layer: str) -> torch.Tensor
```

Performs forward pass through InceptionV3 layers sequentially, returning early when `target_layer` is reached. This means shallower layers are computationally cheaper.

#### Available Layers (shallow → deep)

| Index | Layer Name | Visual Effect |
|-------|-----------|---------------|
| 0 | `Conv2d_1a_3x3` | Edges, low-level patterns |
| 1 | `Conv2d_2b_3x3` | Textures |
| 2 | `Conv2d_3b_1x1` | Simple patterns |
| 3 | `Conv2d_4a_3x3` | Complex textures |
| 4 | `Mixed_5b` | Eyes, spirals |
| 5 | `Mixed_5c` | Animal features |
| 6 | `Mixed_5d` | Shapes, proto-objects |
| 7 | `Mixed_6a` | Abstract structures |
| 8 | `Mixed_6b` | Complex forms |
| 9 | `Mixed_7a` | High-level hallucinations |

---

## style_engine.py

### `StyleTransferEngine`

Real-time neural style transfer using feed-forward TransformNet models.

#### Constructor

```python
StyleTransferEngine(
    models_dir: str = "models",
    resolution: tuple[int, int] = (640, 480),
    turbo: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models_dir` | `str` | `"models"` | Directory containing `.pth` style model files |
| `resolution` | `(width, height)` | `(640, 480)` | Processing resolution |
| `turbo` | `bool` | `False` | Enable TensorRT/FP16 acceleration |

#### Controllable Parameters (Attributes)

| Attribute | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `style_blend` | `float` | 0.0–1.0 | 1.0 | Blend between original (0) and styled (1) |
| `feedback` | `float` | 0.0–1.0 | 0.0 | Previous frame recursion |
| `hue_shift` | `float` | 0.0–1.0 | 0.0 | Color rotation |
| `saturation` | `float` | 0.0–2.0 | 1.0 | Saturation multiplier |
| `contrast` | `float` | 0.5–2.0 | 1.0 | Contrast multiplier |

#### Methods

##### `load_style(style_name: str) -> bool`

Load a style model by name. Models are cached after first load.

Returns `True` if the style was loaded successfully, `False` if the model file is missing.

##### `select_style_by_index(index: int) -> bool`

Select a style by numeric index (0-7). Maps to the `STYLE_MODELS` dictionary order.

##### `process_frame(frame: np.ndarray) -> np.ndarray`

Apply style transfer to a single video frame.

- **Input**: BGR uint8 numpy array
- **Output**: Styled BGR uint8 frame
- **Cost**: 1 forward pass through TransformNet (~7MB model)

##### `update_params(params: dict)`

Recognized keys: `style_index`, `style_name`, `style_blend`, `feedback`, `hue_shift`, `saturation`, `contrast`

##### `get_info() -> dict`

Returns current engine state for HUD overlay.

#### Available Styles

| Name | Description | Filename |
|------|------------|----------|
| `mosaic` | Classic mosaic tiles | `mosaic.pth` |
| `candy` | Bright candy colors | `candy.pth` |
| `rain_princess` | Dreamy rain painting | `rain_princess.pth` |
| `udnie` | Abstract geometric (Francis Picabia) | `udnie.pth` |
| `starry_night` | Van Gogh swirls | `starry_night.pth` |
| `la_muse` | Picasso-esque abstraction | `la_muse.pth` |
| `the_scream` | Edvard Munch distortion | `the_scream.pth` |
| `feathers` | Peacock feather patterns | `feathers.pth` |

### `TransformNet(nn.Module)`

Feed-forward style transformation network (Johnson et al. 2016).

Architecture: `3 downsampling convs → 5 residual blocks → 2 upsampling convs → final conv`

- Input: `[B, 3, H, W]` float tensor in [0, 255] range
- Output: `[B, 3, H, W]` float tensor in [0, 255] range

---

## turbo_engine.py

### `TurboStyleEngine`

TensorRT-accelerated style transfer for maximum FPS.

#### Constructor

```python
TurboStyleEngine(
    model: nn.Module,
    input_shape: tuple[int, int],
    cache_dir: str = "models/trt_cache"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | TransformNet model to accelerate |
| `input_shape` | `(H, W)` | Fixed input resolution for TRT engine |
| `cache_dir` | `str` | Directory to cache serialized TRT engines |

#### Optimization Hierarchy

Automatically selects the best available backend:

1. **TensorRT FP16** — 3-5× speedup (requires `torch_tensorrt`)
2. **torch.compile + FP16** — 1.5-2× speedup (requires PyTorch 2.0+)
3. **Vanilla FP16** — ~2× speedup (baseline CUDA improvement)

#### Methods

##### `forward(input_tensor: torch.Tensor) -> torch.Tensor`

Run optimized forward pass. Handles FP16 conversion and CUDA stream management internally.

##### `optimization_info -> str`

Returns string describing current optimization level: `"tensorrt_fp16"`, `"torch_compile_fp16"`, `"fp16"`, or `"cpu"`.

---

### `TurboDreamOptimizer`

Optimizes DeepDream inference with AMP and reduced computation. DeepDream needs gradients so TensorRT doesn't work directly.

#### Methods

##### `dream_step(tensor, target_layer, intensity) -> torch.Tensor`

Single gradient ascent step with Automatic Mixed Precision.

---

### `ProcessingResolutionManager`

Manages downscale/upscale for the processing pipeline. This is the single most impactful optimization — AI cost scales quadratically with resolution.

#### Constructor

```python
ProcessingResolutionManager(
    display_res: tuple[int, int],
    process_preset: str = "balanced"
)
```

#### Presets

| Preset | Resolution | Description |
|--------|-----------|-------------|
| `nano` | 160×96 | Jetson Nano maximum FPS |
| `nano_fast` | 213×120 | Jetson Nano balanced |
| `ultra_fast` | 320×180 | Very fast, chunky dream |
| `fast` | 426×240 | Good balance |
| `balanced` | 480×270 | Nice detail (default) |
| `quality` | 640×360 | Best visual quality |
| `native` | Display res | No downscaling |

#### Methods

##### `downscale(frame: np.ndarray) -> np.ndarray`

Downscale from display to processing resolution using `INTER_AREA` (good for shrinking).

##### `upscale(frame: np.ndarray) -> np.ndarray`

Upscale from processing to display resolution using `INTER_CUBIC` (good for enlarging).

##### `downscale_gpu(frame) / upscale_gpu(frame)`

GPU-accelerated variants using `cv2.cuda`. Falls back to CPU versions if CUDA not available in OpenCV.

---

### `PerformanceProfiler`

Real-time per-section timing profiler with rolling window.

#### Methods

##### `start(name: str)` / `stop(name: str)`

Start/stop timing a named section. Call in pairs.

##### `tick()`

Mark one frame completion.

##### `avg_ms(name: str) -> float`

Rolling average time in milliseconds for a section.

##### `summary() -> dict`

Dictionary of all sections with average timings, for HUD display.

##### `report() -> str`

Formatted multi-line performance report.

---

## video_pipeline.py

### `VideoPipeline`

Camera capture and display management with GStreamer support.

#### Constructor

```python
VideoPipeline(
    camera: int | str = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fullscreen: bool = False,
    window_name: str = "jetson-dream",
    turbo: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera` | `int\|str` | `0` | Camera source: index, `"csi"`, `"ps3eye"`, or URL |
| `width` | `int` | `640` | Display width in pixels |
| `height` | `int` | `480` | Display height in pixels |
| `fps` | `int` | `30` | Target camera framerate |
| `fullscreen` | `bool` | `False` | Start in fullscreen |
| `turbo` | `bool` | `False` | Use GStreamer optimizations (drop frames, no sync) |

#### Methods

##### `start()`

Open camera and create display window. Raises `RuntimeError` if camera unavailable.

##### `read_frame() -> np.ndarray | None`

Capture a single BGR uint8 frame. Returns `None` if capture failed.

##### `show_frame(frame: np.ndarray, info: dict | None = None)`

Display frame with optional HUD overlay. Updates internal FPS counter.

##### `poll_key(wait_ms: int = 1) -> int`

Poll for keyboard input. Returns key code or -1 (255 masked).

##### `toggle_fullscreen()` / `toggle_hud()`

Toggle display modes.

##### `stop()`

Release camera and close window.

##### `generate_test_frame() -> np.ndarray`

Generate a synthetic test pattern (when no camera available).

### `build_gst_pipeline(...) -> str | int`

Build a GStreamer pipeline string for various camera sources.

- CSI cameras use `nvarguscamerasrc` with hardware-accelerated `nvvidconv`
- USB cameras in turbo mode use `v4l2src` with optimized buffer settings
- Returns device index for standard USB camera usage

---

## async_pipeline.py

### `FrameBuffer`

Thread-safe single-frame buffer with drop semantics. Always stores only the latest frame.

#### Methods

| Method | Description |
|--------|-------------|
| `put(frame)` | Store new frame, overwriting previous |
| `get(timeout=0.1)` | Block until new frame available, return it |
| `peek()` | Return latest frame without blocking or clearing |
| `frame_id` | Monotonically increasing frame counter |

### `CaptureThread(Thread)`

Daemon thread that reads camera frames at full speed into a FrameBuffer.

### `ProcessThread(Thread)`

Daemon thread that reads from input buffer, applies AI processing (with optional downscale/upscale), and writes to output buffer.

#### Methods

| Method | Description |
|--------|-------------|
| `update_process_fn(fn)` | Hot-swap the AI processing function |
| `fps` | Current processing frame rate |
| `process_ms` | Latest frame processing time in milliseconds |

### `AsyncPipeline`

Complete async pipeline: capture → process → display.

#### Constructor

```python
AsyncPipeline(
    cap: cv2.VideoCapture,
    process_fn: Callable[[np.ndarray], np.ndarray],
    display_w: int = 1280,
    display_h: int = 720,
    downscale_fn: Callable | None = None,
    upscale_fn: Callable | None = None
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Launch capture + process threads |
| `stop()` | Stop all threads |
| `get_display_frame(timeout)` | Get latest processed frame (non-blocking) |
| `get_raw_frame()` | Get latest raw camera frame |
| `update_process_fn(fn)` | Swap AI engine live |
| `get_info()` | Dict with `cap_fps`, `ai_fps`, `ai_ms` |

---

## midi_launchpad.py

### `LaunchpadMidi`

Novation Launchpad MK3 MIDI controller interface (Programmer Mode).

#### Constructor

```python
LaunchpadMidi(port_name: str | None = None)
```

| Parameter | Description |
|-----------|-------------|
| `port_name` | Optional substring to match MIDI port name. `None` = auto-detect |

#### State Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pads_held` | `dict[int, float]` | Currently held pad notes → velocity [0-1] |
| `pad_triggers` | `dict[int, float]` | One-shot triggers (cleared each `get_state()`) |
| `top_buttons` | `dict[int, bool]` | CC 91-98 button states |
| `top_triggers` | `dict[int, bool]` | One-shot button triggers |
| `available` | `bool` | Whether MIDI connection is active |

#### Methods

| Method | Description |
|--------|-------------|
| `start()` | Open MIDI ports, enter Programmer mode |
| `stop()` | Close ports, clear LEDs |
| `get_state() -> dict` | Thread-safe snapshot (clears triggers) |
| `set_pad_color(note, color)` | Set pad LED by palette index |
| `set_pad_rgb(note, r, g, b)` | Set pad LED to exact RGB (SysEx) |
| `set_all_pads(color)` | Set all 64 pads to same color |
| `clear_pads()` | Turn off all LEDs |
| `set_row_colors(row, colors)` | Set entire row of LEDs |
| `list_ports()` | Print available MIDI ports |

### Helper Functions

```python
note_to_grid(note: int) -> tuple[int, int] | None
grid_to_note(row: int, col: int) -> int
```

Convert between MIDI note numbers and (row, col) grid positions. Row 0 = bottom, Col 0 = left.

---

## param_mapper.py

### `ParamMapper`

Maps Launchpad MK3 state to AI engine parameters.

#### Constructor

```python
ParamMapper(launchpad=None)
```

Works with or without a connected Launchpad (keyboard-only mode).

#### Methods

| Method | Description |
|--------|-------------|
| `process(midi_state) -> dict` | Process Launchpad state, return parameter updates |
| `update_leds()` | Refresh Launchpad LED display |
| `get_info() -> dict` | Current mode/preset for HUD |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mode` | `int` | `MODE_DREAM` (0), `MODE_STYLE` (1), or `MODE_BLEND` (2) |
| `params` | `dict` | All current AI parameter values |
| `freeze` | `bool` | Frame freeze state |
| `preset_name` | `str` | Name of last applied preset |

---

## streaming.py

### `MJPEGServer`

Simple HTTP MJPEG server for network streaming.

#### Constructor

```python
MJPEGServer(port: int = 8080, quality: int = 80)
```

#### Methods

| Method | Description |
|--------|-------------|
| `put_frame(frame)` | Update the frame to stream |
| `start()` | Begin accepting connections |
| `stop()` | Shutdown server |

Clients connect via `http://<jetson-ip>:8080/` and receive a continuous MJPEG stream.

---

## download_styles.py

### `download_styles(output_dir: str = "models")`

Downloads all available pre-trained style models from PyTorch's example model repository. Models are ~7MB each. Skips files that already exist.
