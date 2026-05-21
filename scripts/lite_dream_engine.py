#!/usr/bin/env python3
"""
Lightweight DeepDream engine using MobileNetV2.

MobileNetV2 is ~10× smaller and ~5× faster than InceptionV3:
  - InceptionV3: 27.2M params, 5.7 GFLOPS
  - MobileNetV2:  3.4M params, 0.3 GFLOPS

This makes it practical for real-time DeepDream on Jetson Nano
at usable frame rates (15-30 FPS at 160×96).

The trade-off is that MobileNetV2's features are less "dreamy" than
InceptionV3's (fewer complex patterns), but the speed gain is massive.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError:
    torch = None
    nn = None

try:
    import cv2
except ImportError:
    cv2 = None


# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MobileNetV2 feature layers we can dream on
MOBILENET_LAYERS = [
    "features.0",    # 0 — first conv (edges)
    "features.1",    # 1 — first inverted residual
    "features.3",    # 2 — early features
    "features.6",    # 3 — mid features
    "features.10",   # 4 — mid-deep features
    "features.13",   # 5 — deep features
    "features.16",   # 6 — near-final features
    "features.18",   # 7 — final conv (most abstract)
]


# Base class for when torch is not available
_ModuleBase = nn.Module if nn is not None else object


class MobileNetDreamModel(_ModuleBase):
    """MobileNetV2 feature extractor for lightweight DeepDream."""

    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        mobilenet.eval()
        self.features = mobilenet.features

    def forward(self, x: "torch.Tensor", target_layer_idx: int) -> "torch.Tensor":
        """Forward pass up to target layer index, return activations."""
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self._resolve_layer_idx(target_layer_idx):
                return x
        return x

    @staticmethod
    def _resolve_layer_idx(dream_layer_idx: int) -> int:
        """Map our dream layer index (0-7) to MobileNetV2 feature index."""
        mapping = [0, 1, 3, 6, 10, 13, 16, 18]
        idx = max(0, min(dream_layer_idx, len(mapping) - 1))
        return mapping[idx]


class LiteDreamEngine:
    """Lightweight DeepDream using MobileNetV2 — optimized for Jetson Nano.

    ~5× faster than InceptionV3-based DeepDream at the same resolution.
    """

    def __init__(self, resolution: tuple[int, int] = (320, 240)):
        if torch is None:
            raise RuntimeError("PyTorch required: pip install torch torchvision")
        if cv2 is None:
            raise RuntimeError("OpenCV required: pip install opencv-python")

        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"LiteDream: loading MobileNetV2 on {self.device}...")
        self.model = MobileNetDreamModel().to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # AMP for CUDA
        self._amp_enabled = torch.cuda.is_available()
        if self._amp_enabled:
            self.model.half()
            print("LiteDream: FP16 enabled")

        print("LiteDream: ready (3.4M params)")

        # Parameters
        self.layer_index = 4
        self.intensity = 0.03
        self.iterations = 2
        self.feedback = 0.3
        self.zoom = 1.002
        self.jitter = 8
        self.hue_shift = 0.0
        self.blur_amount = 0.0
        self.frame_skip = 0

        # State
        self._prev_frame = None
        self._frame_counter = 0
        self._skip_result = None

        # Normalization
        self._normalize = transforms.Normalize(
            mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()
        )

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        """BGR uint8 → normalized tensor [1, 3, H, W]."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = self._normalize(tensor)
        tensor = tensor.to(self.device)
        if self._amp_enabled:
            tensor = tensor.half()
        return tensor

    def _postprocess(self, tensor: "torch.Tensor") -> np.ndarray:
        """Tensor → BGR uint8 frame."""
        img = tensor.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _dream_step(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Single gradient ascent step."""
        tensor = tensor.float().detach().requires_grad_(True)

        if self._amp_enabled:
            with torch.amp.autocast("cuda"):
                activations = self.model(tensor, self.layer_index)
                loss = activations.norm()
        else:
            activations = self.model(tensor, self.layer_index)
            loss = activations.norm()

        loss.backward()
        grad = tensor.grad.data
        grad /= grad.abs().mean() + 1e-8
        tensor = tensor.detach() + grad * self.intensity
        return tensor

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame through lightweight DeepDream."""
        # Frame skip
        if self.frame_skip > 0:
            self._frame_counter += 1
            if self._frame_counter % (self.frame_skip + 1) != 0:
                if self._skip_result is not None:
                    return self._skip_result

        h, w = frame.shape[:2]

        # Zoom
        if self.zoom > 1.0:
            dh = int(h * (1 - 1 / self.zoom) / 2)
            dw = int(w * (1 - 1 / self.zoom) / 2)
            if dh >= 1 and dw >= 1:
                frame = cv2.resize(frame[dh:h-dh, dw:w-dw], (w, h))

        # Feedback
        if self._prev_frame is not None and self.feedback > 0:
            prev = cv2.resize(self._prev_frame, (w, h))
            frame = cv2.addWeighted(frame, 1 - self.feedback, prev, self.feedback, 0)

        # Dream iterations
        tensor = self._preprocess(frame)
        for _ in range(self.iterations):
            with torch.enable_grad():
                if self.jitter > 0:
                    ox = np.random.randint(-self.jitter, self.jitter + 1)
                    oy = np.random.randint(-self.jitter, self.jitter + 1)
                    tensor = torch.roll(tensor, shifts=(oy, ox), dims=(2, 3))

                tensor = self._dream_step(tensor)

                if self.jitter > 0:
                    tensor = torch.roll(tensor, shifts=(-oy, -ox), dims=(2, 3))

        result = self._postprocess(tensor)

        # Post-processing
        if self.hue_shift > 0.001:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift * 180) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if self.blur_amount > 0.01:
            ksize = int(self.blur_amount * 15) * 2 + 1
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        self._prev_frame = result.copy()
        self._skip_result = result
        return result

    def update_params(self, params: dict):
        """Update from MIDI mapper."""
        if "layer_index" in params:
            self.layer_index = max(0, min(int(params["layer_index"]), 7))
        if "intensity" in params:
            self.intensity = float(params["intensity"])
        if "iterations" in params:
            self.iterations = max(1, min(int(params["iterations"]), 5))
        if "feedback" in params:
            self.feedback = float(np.clip(params["feedback"], 0, 1))
        if "zoom" in params:
            self.zoom = float(params["zoom"])
        if "jitter" in params:
            self.jitter = int(params["jitter"])
        if "hue_shift" in params:
            self.hue_shift = float(params["hue_shift"])
        if "blur_amount" in params:
            self.blur_amount = float(np.clip(params["blur_amount"], 0, 1))
        if "frame_skip" in params:
            self.frame_skip = max(0, int(params["frame_skip"]))

    def get_info(self) -> dict:
        return {
            "engine": "LiteDream (MobileNetV2)",
            "layer": f"{self.layer_index}/7",
            "intensity": f"{self.intensity:.3f}",
            "iterations": self.iterations,
            "feedback": f"{self.feedback:.2f}",
            "params": "3.4M",
        }
