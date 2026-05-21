# jetson-dream Documentation

## Table of Contents

| Document | Description |
|----------|-------------|
| [Setup Guide](setup-guide.md) | Installation, hardware requirements, camera/MIDI/display setup |
| [Architecture](architecture.md) | System design, data flow, module dependencies, thread model |
| [API Reference](api-reference.md) | Complete class and method reference for all modules |
| [Performance Tuning](performance-tuning.md) | FPS optimization strategies, platform-specific settings |
| [MIDI Controller](midi-controller.md) | Launchpad MK3 pad layout, parameter mapping, LED feedback |
| [DeepDream Explained](deepdream-explained.md) | How the algorithm works, layer selection, parameter interactions |

## Quick Navigation

**"I want to..."**

- **Get started** → [Setup Guide](setup-guide.md)
- **Understand the codebase** → [Architecture](architecture.md)
- **Look up a method or class** → [API Reference](api-reference.md)
- **Get more FPS** → [Performance Tuning](performance-tuning.md)
- **Learn the Launchpad controls** → [MIDI Controller](midi-controller.md)
- **Understand DeepDream** → [DeepDream Explained](deepdream-explained.md)

## CLI Quick Reference

```bash
# Standard mode
python scripts/main.py -c 0 -f

# Turbo mode (Orin Nano)
python scripts/main.py --turbo -f

# Nano mode (Jetson Nano — maximum FPS)
python scripts/main.py --nano -f

# Style transfer only (fastest)
python scripts/main.py --turbo --mode style -f

# No hardware (testing)
python scripts/main.py --no-camera --no-midi
```

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/main.py` | ~500 | Entry point, CLI args, main loop |
| `scripts/dream_engine.py` | ~400 | DeepDream (InceptionV3 gradient ascent) |
| `scripts/style_engine.py` | ~350 | Fast Neural Style Transfer |
| `scripts/turbo_engine.py` | ~400 | TensorRT/FP16 + resolution manager + profiler |
| `scripts/async_pipeline.py` | ~290 | Threaded capture→process→display pipeline |
| `scripts/video_pipeline.py` | ~250 | Camera I/O, GStreamer, HUD overlay |
| `scripts/midi_launchpad.py` | ~280 | Launchpad MK3 MIDI protocol |
| `scripts/param_mapper.py` | ~380 | MIDI pads → AI parameters + presets |
| `scripts/streaming.py` | ~100 | MJPEG HTTP server |
| `scripts/download_styles.py` | ~55 | Style model downloader |
