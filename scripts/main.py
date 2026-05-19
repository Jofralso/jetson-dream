#!/usr/bin/env python3
"""
jetson-dream — Live AI video dreaming with MIDI control.

Main entry point. Connects camera → AI engines → display,
controlled in real-time by a Novation Launchpad MK3.
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np

from scripts.dream_engine import DeepDreamEngine
from scripts.midi_launchpad import LaunchpadMidi
from scripts.param_mapper import MODE_BLEND, MODE_DREAM, MODE_STYLE, ParamMapper
from scripts.style_engine import StyleTransferEngine
from scripts.video_pipeline import VideoPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Live AI video dreaming with Launchpad MK3 control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # USB camera, auto-detect Launchpad
  python main.py -c 0                     # Explicit USB camera index
  python main.py -c csi                   # Jetson CSI camera
  python main.py -c csi -f               # CSI + fullscreen (for projector)
  python main.py --no-camera              # Test pattern (no camera needed)
  python main.py --no-midi               # Without MIDI controller
  python main.py --width 320 --height 240 # Lower res = faster dreaming
  python main.py --list-devices           # Show available cameras and MIDI
        """,
    )
    p.add_argument("-c", "--camera", default="0",
                   help="Camera source: device index, 'csi', or URL (default: 0)")
    p.add_argument("-W", "--width", type=int, default=480,
                   help="Processing width (default: 480, lower = faster)")
    p.add_argument("-H", "--height", type=int, default=360,
                   help="Processing height (default: 360, lower = faster)")
    p.add_argument("--fps", type=int, default=30,
                   help="Target camera FPS (default: 30)")
    p.add_argument("-f", "--fullscreen", action="store_true",
                   help="Start in fullscreen mode")
    p.add_argument("--no-camera", action="store_true",
                   help="Use test pattern instead of camera")
    p.add_argument("--no-midi", action="store_true",
                   help="Run without MIDI controller")
    p.add_argument("--midi-port", default=None,
                   help="MIDI port name substring to match")
    p.add_argument("--models-dir", default="models",
                   help="Directory containing style model .pth files")
    p.add_argument("--mode", choices=["dream", "style", "blend"], default="dream",
                   help="Starting engine mode (default: dream)")
    p.add_argument("--list-devices", action="store_true",
                   help="List available devices and exit")
    return p.parse_args()


def list_devices():
    """Print available cameras and MIDI ports."""
    print("\n=== Cameras ===")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{i}] USB camera {w}x{h}")
            cap.release()
    print("  [csi] Jetson CSI camera (if connected)")

    print("\n=== MIDI ===")
    lp = LaunchpadMidi()
    lp.list_ports()
    print()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    print("=" * 60)
    print("  jetson-dream — Live AI Video Dreaming")
    print("=" * 60)

    # ── Camera source ──
    camera = args.camera
    if camera.isdigit():
        camera = int(camera)

    # ── Initialize components ──
    print("\n[1/4] Video pipeline...")
    video = VideoPipeline(
        camera=camera if not args.no_camera else -1,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fullscreen=args.fullscreen,
    )

    print("[2/4] DeepDream engine...")
    dream = DeepDreamEngine(resolution=(args.width, args.height))

    print("[3/4] Style Transfer engine...")
    style = StyleTransferEngine(models_dir=args.models_dir, resolution=(args.width, args.height))
    # Try to load the first available style
    style.select_style_by_index(0)

    print("[4/4] MIDI (Launchpad MK3)...")
    launchpad = None
    mapper = None
    if not args.no_midi:
        launchpad = LaunchpadMidi(port_name=args.midi_port)
        launchpad.start()
        mapper = ParamMapper(launchpad=launchpad)

        # Set starting mode
        if args.mode == "style":
            mapper.mode = MODE_STYLE
        elif args.mode == "blend":
            mapper.mode = MODE_BLEND

        # Light up the pad grid
        mapper.update_leds()
    else:
        mapper = ParamMapper()
        print("  MIDI disabled — keyboard only")

    # ── Start video ──
    if args.no_camera:
        # Create window manually for test pattern mode
        cv2.namedWindow(video.window_name, cv2.WINDOW_AUTOSIZE)
        if args.fullscreen:
            cv2.setWindowProperty(
                video.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        video._running = True
    else:
        video.start()

    print("\n" + "=" * 60)
    print("  RUNNING — press Q or ESC to quit")
    print("  Keys: 1=Dream 2=Style 3=Blend F=Fullscreen H=HUD")
    print("=" * 60 + "\n")

    # ── Signal handling ──
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Main loop ──
    frozen_frame = None
    frame_time = time.time()

    while running:
        # Read MIDI state
        if launchpad and launchpad.available:
            midi_state = launchpad.get_state()
            updates = mapper.process(midi_state)

            # Apply parameter updates to engines
            if updates:
                dream.update_params(updates)
                style.update_params(updates)

            # Handle UI toggles
            if mapper.toggle_hud:
                video.toggle_hud()
            if mapper.toggle_fullscreen:
                video.toggle_fullscreen()

            # Update LED feedback periodically
            if int(time.time() * 2) % 2 == 0:
                mapper.update_leds()

        # Capture frame
        if mapper.freeze and frozen_frame is not None:
            frame = frozen_frame.copy()
        elif args.no_camera:
            frame = video.generate_test_frame()
        else:
            frame = video.read_frame()
            if frame is None:
                print("Camera read failed — using test pattern")
                frame = video.generate_test_frame()

        if mapper.freeze and frozen_frame is None:
            frozen_frame = frame.copy()
        elif not mapper.freeze:
            frozen_frame = None

        # Process through AI engine(s)
        mode = mapper.mode

        if mode == MODE_DREAM:
            output = dream.process_frame(frame)
        elif mode == MODE_STYLE:
            output = style.process_frame(frame)
        elif mode == MODE_BLEND:
            dreamed = dream.process_frame(frame)
            styled = style.process_frame(frame)
            # Blend both outputs
            blend = mapper.params.get("style_blend", 0.5)
            output = cv2.addWeighted(dreamed, 1 - blend, styled, blend, 0)
        else:
            output = frame

        # Build HUD info
        info = {}
        if mode in (MODE_DREAM, MODE_BLEND):
            info.update(dream.get_info())
        if mode in (MODE_STYLE, MODE_BLEND):
            info.update(style.get_info())
        info.update(mapper.get_info())

        # Display
        video.show_frame(output, info)

        # Keyboard controls
        key = video.poll_key(1)
        if key == ord("q") or key == 27:  # Q or ESC
            running = False
        elif key == ord("f"):
            video.toggle_fullscreen()
        elif key == ord("h"):
            video.toggle_hud()
        elif key == ord("1"):
            mapper.mode = MODE_DREAM
            print("Mode: DeepDream")
        elif key == ord("2"):
            mapper.mode = MODE_STYLE
            print("Mode: StyleTransfer")
        elif key == ord("3"):
            mapper.mode = MODE_BLEND
            print("Mode: Dream+Style")
        elif key == ord("r"):
            mapper._reset_params()
            dream.update_params(mapper.params)
            style.update_params(mapper.params)
            print("Parameters reset")
        elif key == ord(" "):
            mapper.freeze = not mapper.freeze
            print(f"Freeze: {'ON' if mapper.freeze else 'OFF'}")
        elif key == ord("+") or key == ord("="):
            dream.intensity = min(0.1, dream.intensity * 1.2)
            print(f"Intensity: {dream.intensity:.4f}")
        elif key == ord("-"):
            dream.intensity = max(0.001, dream.intensity / 1.2)
            print(f"Intensity: {dream.intensity:.4f}")

        # Frame timing
        now = time.time()
        frame_time = now

    # ── Cleanup ──
    print("\nShutting down...")
    if launchpad:
        launchpad.stop()
    video.stop()
    print("Done.")


if __name__ == "__main__":
    main()
