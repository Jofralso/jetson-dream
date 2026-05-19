#!/usr/bin/env python3
"""
Async threaded pipeline for 720p@30fps processing.

Decouples camera capture, AI processing, and display into separate threads
so each can run at its own pace. The display always shows the most recent
processed frame, even if the AI engine is slower than 30fps.

Architecture:
  CaptureThread → [frame_queue] → ProcessThread → [output_queue] → DisplayThread
       720p          maxsize=2       process_res        maxsize=2          720p

Frame dropping strategy:
  - Capture drops old frames (always gives freshest camera frame)
  - Process drops old frames (never falls behind)
  - Display always shows latest result (smooth at monitor refresh rate)
"""

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class FrameBuffer:
    """Thread-safe single-frame buffer with drop semantics.

    Always stores only the latest frame. Readers get the newest frame
    and block until a new frame is available.
    """

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame_id = 0

    def put(self, frame: np.ndarray):
        """Store a new frame, overwriting any previous."""
        with self._lock:
            self._frame = frame
            self._frame_id += 1
        self._event.set()

    def get(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get the latest frame, blocking until available."""
        if self._event.wait(timeout=timeout):
            with self._lock:
                frame = self._frame
                self._event.clear()
            return frame
        return None

    def peek(self) -> np.ndarray | None:
        """Get the latest frame without blocking or clearing."""
        with self._lock:
            return self._frame

    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id


class CaptureThread(threading.Thread):
    """Captures frames from camera at full speed into a FrameBuffer."""

    def __init__(self, cap: "cv2.VideoCapture", output: FrameBuffer,
                 target_w: int, target_h: int):
        super().__init__(daemon=True, name="capture")
        self._cap = cap
        self._output = output
        self._target_w = target_w
        self._target_h = target_h
        self._running = False
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Resize to target resolution if needed
            h, w = frame.shape[:2]
            if w != self._target_w or h != self._target_h:
                frame = cv2.resize(frame, (self._target_w, self._target_h))

            self._output.put(frame)

            # FPS tracking
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = now

    def stop(self):
        self._running = False

    @property
    def fps(self) -> float:
        return self._fps


class ProcessThread(threading.Thread):
    """Runs AI processing on frames from input buffer, writes to output buffer.

    The process_fn receives a frame and returns a processed frame.
    Resolution scaling (downscale before, upscale after) happens here.
    """

    def __init__(self, input_buf: FrameBuffer, output_buf: FrameBuffer,
                 process_fn: Callable[[np.ndarray], np.ndarray],
                 downscale_fn: Callable[[np.ndarray], np.ndarray] | None = None,
                 upscale_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        super().__init__(daemon=True, name="process")
        self._input = input_buf
        self._output = output_buf
        self._process_fn = process_fn
        self._downscale = downscale_fn
        self._upscale = upscale_fn
        self._running = False
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()
        self._process_ms = 0.0
        self._lock = threading.Lock()

    def run(self):
        self._running = True
        while self._running:
            frame = self._input.get(timeout=0.05)
            if frame is None:
                continue

            t0 = time.time()

            # Downscale for AI processing
            if self._downscale is not None:
                proc_frame = self._downscale(frame)
            else:
                proc_frame = frame

            # AI processing
            result = self._process_fn(proc_frame)

            # Upscale back to display resolution
            if self._upscale is not None:
                result = self._upscale(result)

            self._output.put(result)

            # Timing
            dt = time.time() - t0
            with self._lock:
                self._process_ms = dt * 1000

            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = now

    def stop(self):
        self._running = False

    def update_process_fn(self, fn: Callable[[np.ndarray], np.ndarray]):
        """Swap the processing function (e.g., mode change)."""
        self._process_fn = fn

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def process_ms(self) -> float:
        with self._lock:
            return self._process_ms


class AsyncPipeline:
    """Complete async pipeline: capture → process → display.

    Usage:
        pipeline = AsyncPipeline(cap, process_fn, display_res, process_res)
        pipeline.start()
        while running:
            frame = pipeline.get_display_frame()
            if frame is not None:
                cv2.imshow("dream", frame)
        pipeline.stop()
    """

    def __init__(
        self,
        cap: "cv2.VideoCapture",
        process_fn: Callable[[np.ndarray], np.ndarray],
        display_w: int = 1280,
        display_h: int = 720,
        downscale_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        upscale_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._capture_buf = FrameBuffer()
        self._display_buf = FrameBuffer()

        self._capture_thread = CaptureThread(
            cap, self._capture_buf, display_w, display_h
        )
        self._process_thread = ProcessThread(
            self._capture_buf, self._display_buf,
            process_fn, downscale_fn, upscale_fn,
        )

        self._started = False

    def start(self):
        """Start capture and processing threads."""
        if self._started:
            return
        self._capture_thread.start()
        self._process_thread.start()
        self._started = True
        print("AsyncPipeline: threads started")

    def stop(self):
        """Stop all threads."""
        self._capture_thread.stop()
        self._process_thread.stop()
        self._capture_thread.join(timeout=2.0)
        self._process_thread.join(timeout=2.0)
        self._started = False
        print("AsyncPipeline: stopped")

    def get_display_frame(self, timeout: float = 0.033) -> np.ndarray | None:
        """Get the latest processed frame for display.

        Non-blocking with short timeout. Returns None if no frame ready yet.
        """
        return self._display_buf.get(timeout=timeout)

    def get_raw_frame(self) -> np.ndarray | None:
        """Get the latest raw camera frame (for freeze/overlay)."""
        return self._capture_buf.peek()

    def update_process_fn(self, fn: Callable[[np.ndarray], np.ndarray]):
        """Swap the AI processing function live."""
        self._process_thread.update_process_fn(fn)

    @property
    def capture_fps(self) -> float:
        return self._capture_thread.fps

    @property
    def process_fps(self) -> float:
        return self._process_thread.fps

    @property
    def process_ms(self) -> float:
        return self._process_thread.process_ms

    def get_info(self) -> dict:
        return {
            "cap_fps": f"{self.capture_fps:.0f}",
            "ai_fps": f"{self.process_fps:.1f}",
            "ai_ms": f"{self.process_ms:.0f}ms",
        }
