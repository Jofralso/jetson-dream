#!/usr/bin/env python3
"""
Network streaming for jetson-dream output.

Supports:
  - MJPEG HTTP server (simple, low latency)
  - UDP/TCP raw frames
  - GStreamer pipelines
"""

import threading
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

try:
    import socket
except ImportError:
    socket = None


class MJPEGServer:
    """Simple HTTP MJPEG server for streaming video frames."""

    def __init__(self, port: int = 8080, quality: int = 80):
        """
        Args:
            port: HTTP server port
            quality: JPEG quality (0-100)
        """
        self.port = port
        self.quality = quality
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.server_thread = None

    def put_frame(self, frame: np.ndarray):
        """Update the frame to stream."""
        with self.frame_lock:
            self.frame = frame.copy()

    def _encode_frame(self) -> bytes:
        """Encode current frame as JPEG."""
        with self.frame_lock:
            if self.frame is None:
                return None
        ret, data = cv2.imencode(".jpg", self.frame.copy(), 
                                 [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return data if ret else None

    def _http_handler(self, client_socket, client_addr):
        """Handle a single HTTP client connection."""
        try:
            # Read HTTP request
            request = client_socket.recv(4096).decode("utf-8", errors="ignore")
            if "GET" not in request:
                client_socket.close()
                return

            # Send HTTP headers
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
            client_socket.sendall(response.encode())

            # Stream frames
            while self.running:
                jpeg_data = self._encode_frame()
                if jpeg_data is None:
                    time.sleep(0.01)
                    continue

                # Send MJPEG frame
                frame_header = (
                    "--FRAME\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_data)}\r\n"
                    "\r\n"
                )
                client_socket.sendall(frame_header.encode() + jpeg_data + b"\r\n")
                time.sleep(0.01)  # 100 FPS max per client

        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            client_socket.close()

    def _server_loop(self):
        """Main server loop."""
        import socket as sock_module

        server_socket = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
        server_socket.setsockopt(sock_module.SOL_SOCKET, sock_module.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", self.port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"MJPEG server listening on http://0.0.0.0:{self.port}/stream")

        while self.running:
            try:
                client_socket, client_addr = server_socket.accept()
                # Handle in thread to avoid blocking
                threading.Thread(
                    target=self._http_handler,
                    args=(client_socket, client_addr),
                    daemon=True,
                ).start()
            except sock_module.timeout:
                continue
            except Exception as e:
                print(f"Server error: {e}")

        server_socket.close()

    def start(self):
        """Start the streaming server."""
        if self.running:
            return
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Stop the streaming server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)


class UDPStreamer:
    """Stream frames via UDP."""

    def __init__(self, host: str = "localhost", port: int = 5000, quality: int = 85):
        """
        Args:
            host: Target host IP address
            port: Target UDP port
            quality: JPEG quality
        """
        self.host = host
        self.port = port
        self.quality = quality
        self.socket = None

    def connect(self) -> bool:
        """Connect to target host."""
        try:
            import socket as sock_module
            self.socket = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM)
            self.socket.setsockopt(sock_module.SOL_SOCKET, sock_module.SO_REUSEADDR, 1)
            print(f"UDP streamer: sending to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"UDP connect failed: {e}")
            return False

    def send_frame(self, frame: np.ndarray):
        """Send a frame via UDP."""
        if self.socket is None:
            return

        ret, jpeg_data = cv2.imencode(".jpg", frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ret:
            return

        try:
            # UDP max packet ~65KB, split if needed
            chunk_size = 60000
            for i in range(0, len(jpeg_data), chunk_size):
                chunk = jpeg_data[i : i + chunk_size]
                self.socket.sendto(chunk, (self.host, self.port))
        except Exception as e:
            print(f"UDP send failed: {e}")

    def close(self):
        """Close connection."""
        if self.socket:
            self.socket.close()


class GStreamerStreamer:
    """Stream via GStreamer pipeline."""

    def __init__(self, pipeline: str, width: int = 640, height: int = 480, fps: int = 30):
        """
        Args:
            pipeline: GStreamer pipeline string (e.g., "rtmpsink location=...")
            width: Frame width
            height: Frame height
            fps: Frame rate
        """
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None

    def start(self) -> bool:
        """Start GStreamer pipeline."""
        try:
            import subprocess
            full_pipeline = (
                f"fdsrc ! "
                f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! "
                f"{self.pipeline}"
            )
            self.proc = subprocess.Popen(
                ["gst-launch-1.0", "-e"] + full_pipeline.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"GStreamer pipeline started")
            return True
        except Exception as e:
            print(f"GStreamer failed: {e}")
            return False

    def send_frame(self, frame: np.ndarray):
        """Send frame to GStreamer."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"GStreamer send failed: {e}")

    def stop(self):
        """Stop GStreamer pipeline."""
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except:
                self.proc.kill()


class StreamingManager:
    """Manage multiple streaming outputs."""

    def __init__(self):
        self.mjpeg: Optional[MJPEGServer] = None
        self.udp: Optional[UDPStreamer] = None
        self.gstreamer: Optional[GStreamerStreamer] = None

    def enable_mjpeg(self, port: int = 8080, quality: int = 80) -> bool:
        """Enable MJPEG HTTP streaming."""
        self.mjpeg = MJPEGServer(port=port, quality=quality)
        self.mjpeg.start()
        return True

    def enable_udp(self, host: str, port: int = 5000, quality: int = 85) -> bool:
        """Enable UDP streaming."""
        self.udp = UDPStreamer(host=host, port=port, quality=quality)
        return self.udp.connect()

    def enable_gstreamer(self, pipeline: str, width: int = 640, height: int = 480, 
                         fps: int = 30) -> bool:
        """Enable GStreamer streaming."""
        self.gstreamer = GStreamerStreamer(pipeline, width, height, fps)
        return self.gstreamer.start()

    def put_frame(self, frame: np.ndarray):
        """Send frame to all active streams."""
        if self.mjpeg:
            self.mjpeg.put_frame(frame)
        if self.udp:
            self.udp.send_frame(frame)
        if self.gstreamer:
            self.gstreamer.send_frame(frame)

    def stop(self):
        """Stop all streams."""
        if self.mjpeg:
            self.mjpeg.stop()
        if self.udp:
            self.udp.close()
        if self.gstreamer:
            self.gstreamer.stop()
