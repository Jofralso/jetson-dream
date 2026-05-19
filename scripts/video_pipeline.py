#!/usr/bin/env python3
"""
Video capture and display pipeline.

Handles camera input (USB, CSI, IP stream) and fullscreen display output.
Provides frame capture, resolution management, and FPS counting.
Optimized for Jetson with GStreamer backend when available.
"""

import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def build_gst_pipeline(
    camera: int | str = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> str | int:
    """Build GStreamer pipeline string for Jetson CSI camera, or return device index."""
    if isinstance(camera, str):
        if camera.lower() == "csi":
            # NVIDIA CSI camera via nvarguscamerasrc
            return (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"framerate={fps}/1, format=NV12 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
            )
        if camera.startswith("http"):
            # IP camera / MJPEG stream
            return camera
        # Try as file path or GStreamer pipeline
        return camera

    return camera  # USB camera index


class VideoPipeline:
    """Camera capture and display management."""

    def __init__(
        self,
        camera: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fullscreen: bool = False,
        window_name: str = "jetson-dream",
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")

        self.camera = camera
        self.width = width
        self.height = height
        self.target_fps = fps
        self.fullscreen = fullscreen
        self.window_name = window_name

        self._cap = None
        self._frame_count = 0
        self._fps_time = time.time()
        self._current_fps = 0.0
        self._running = False

        # HUD overlay
        self.show_hud = True

    def start(self):
        """Open camera and create display window."""
        source = build_gst_pipeline(self.camera, self.width, self.height, self.target_fps)

        if isinstance(source, str) and ("!" in source or source.startswith("http")):
            # GStreamer pipeline or URL
            if "!" in source:
                self._cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
            else:
                self._cap = cv2.VideoCapture(source)
        else:
            # USB camera index
            self._cap = cv2.VideoCapture(int(source))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera}")

        # Read actual resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        # Create window
        if self.fullscreen:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self._running = True
        self._fps_time = time.time()
        print(f"Display: {'fullscreen' if self.fullscreen else f'{self.width}x{self.height}'}")

    def read_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera.

        Returns BGR uint8 numpy array, or None if capture failed.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None

        # Resize if needed
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def show_frame(self, frame: np.ndarray, info: dict | None = None):
        """Display a frame with optional HUD overlay."""
        # Update FPS counter
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

        # Draw HUD
        if self.show_hud and info:
            self._draw_hud(frame, info)

        cv2.imshow(self.window_name, frame)

    def _draw_hud(self, frame: np.ndarray, info: dict):
        """Draw heads-up display overlay."""
        y = 30
        line_height = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (0, 255, 0)
        thickness = 1

        # FPS
        cv2.putText(frame, f"FPS: {self._current_fps:.1f}", (10, y),
                     font, scale, color, thickness)
        y += line_height

        # Engine info
        for key, value in info.items():
            cv2.putText(frame, f"{key}: {value}", (10, y),
                         font, scale, color, thickness)
            y += line_height

    def poll_key(self, wait_ms: int = 1) -> int:
        """Poll for keyboard input. Returns key code or -1."""
        return cv2.waitKey(wait_ms) & 0xFF

    @property
    def fps(self) -> float:
        return self._current_fps

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )

    def toggle_hud(self):
        self.show_hud = not self.show_hud

    def stop(self):
        """Release camera and close window."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def generate_test_frame(self) -> np.ndarray:
        """Generate a test pattern frame (when no camera is available)."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = time.time()

        # Animated gradient
        for y in range(self.height):
            for x in range(0, self.width, 4):  # Skip pixels for speed
                r = int(127 + 127 * np.sin(x * 0.02 + t))
                g = int(127 + 127 * np.sin(y * 0.02 + t * 0.7))
                b = int(127 + 127 * np.sin((x + y) * 0.01 + t * 1.3))
                frame[y, x:x + 4] = (b, g, r)

        cv2.putText(frame, "NO CAMERA — TEST PATTERN", (10, self.height // 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
