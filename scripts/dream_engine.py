#!/usr/bin/env python3
"""
DeepDream engine — real-time dreamy video processing.

Uses PyTorch InceptionV3 to maximize activations in selected layers,
producing the classic psychedelic / hallucinatory DeepDream effect.
Optimized for Jetson Orin Nano with TensorRT when available.

Controllable parameters:
  - layer_index:  Which Inception layer to dream on (0-9). Shallow layers
                  produce textures/edges, deep layers produce eyes/faces/dogs.
  - intensity:    Gradient step size (dream strength per iteration).
  - octaves:      Number of resolution scales (more = richer detail, slower).
  - octave_scale: Downscale factor between octaves.
  - iterations:   Gradient ascent steps per octave.
  - feedback:     How much of the previous dreamed frame feeds into the next
                  (0 = no feedback / each frame independent, 1 = full recursion).
  - zoom:         Slow zoom into center each frame (for the classic zoom effect).
  - jitter:       Random shift before each gradient step (reduces tile artifacts).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Named layers we can dream on — ordered shallow → deep
DREAM_LAYERS = [
    "Conv2d_1a_3x3",       # 0 — edges, low-level patterns
    "Conv2d_2b_3x3",       # 1 — textures
    "Conv2d_3b_1x1",       # 2 — simple patterns
    "Conv2d_4a_3x3",       # 3 — complex textures
    "Mixed_5b",            # 4 — eyes, spirals
    "Mixed_5c",            # 5 — animal features
    "Mixed_5d",            # 6 — shapes, proto-objects
    "Mixed_6a",            # 7 — abstract structures
    "Mixed_6b",            # 8 — complex forms
    "Mixed_7a",            # 9 — high-level hallucinations
]


class InceptionDreamModel(nn.Module):
    """Wrapper around InceptionV3 that returns activations at a chosen layer."""

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Extract the sequential feature layers we care about
        self.layers = nn.ModuleList()
        self.layer_names = []

        # Walk the inception modules in order and store them
        layer_map = {
            "Conv2d_1a_3x3": inception.Conv2d_1a_3x3,
            "Conv2d_2a_3x3": inception.Conv2d_2a_3x3,
            "Conv2d_2b_3x3": inception.Conv2d_2b_3x3,
            "maxpool1": nn.MaxPool2d(kernel_size=3, stride=2),
            "Conv2d_3b_1x1": inception.Conv2d_3b_1x1,
            "Conv2d_4a_3x3": inception.Conv2d_4a_3x3,
            "maxpool2": nn.MaxPool2d(kernel_size=3, stride=2),
            "Mixed_5b": inception.Mixed_5b,
            "Mixed_5c": inception.Mixed_5c,
            "Mixed_5d": inception.Mixed_5d,
            "Mixed_6a": inception.Mixed_6a,
            "Mixed_6b": inception.Mixed_6b,
            "Mixed_6c": inception.Mixed_6c,
            "Mixed_6d": inception.Mixed_6d,
            "Mixed_6e": inception.Mixed_6e,
            "Mixed_7a": inception.Mixed_7a,
            "Mixed_7b": inception.Mixed_7b,
            "Mixed_7c": inception.Mixed_7c,
        }
        for name, module in layer_map.items():
            self.layers.append(module)
            self.layer_names.append(name)

    def forward(self, x: "torch.Tensor", target_layer: str) -> "torch.Tensor":
        """Forward pass up to target_layer, return activations."""
        for name, layer in zip(self.layer_names, self.layers):
            x = layer(x)
            if name == target_layer:
                return x
        return x


class DeepDreamEngine:
    """Real-time DeepDream processor for video frames."""

    def __init__(self, resolution: tuple[int, int] = (640, 480), turbo: bool = False):
        if torch is None:
            raise RuntimeError("PyTorch is required: pip install torch torchvision")
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")

        self.resolution = resolution
        self.turbo = turbo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"DeepDream: loading InceptionV3 on {self.device}...")
        self.model = InceptionDreamModel().to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # FP16 + AMP for turbo mode
        self._amp_enabled = turbo and torch.cuda.is_available()
        if self._amp_enabled:
            print("DeepDream: AMP FP16 enabled")

        print("DeepDream: model loaded")

        # Processing parameters (updated each frame from MIDI)
        self.layer_index = 4          # Start with Mixed_5b (eyes/spirals)
        self.intensity = 0.02         # Step size
        self.octaves = 3              # Number of octave scales
        self.octave_scale = 1.4       # Downscale factor
        self.iterations = 5           # Steps per octave
        self.feedback = 0.3           # Previous frame bleed [0-1]
        self.zoom = 1.002             # Zoom factor per frame (1.0 = none)
        self.jitter = 16              # Random shift to reduce tiling
        self.hue_shift = 0.0          # Color rotation [0-1]
        self.blur_amount = 0.0        # Post-blur [0-1]

        # State
        self._prev_frame = None       # Previous output for feedback loop

        # Normalization transform
        self._normalize = transforms.Normalize(
            mean=IMAGENET_MEAN.tolist(),
            std=IMAGENET_STD.tolist(),
        )

    @property
    def target_layer(self) -> str:
        idx = max(0, min(self.layer_index, len(DREAM_LAYERS) - 1))
        return DREAM_LAYERS[idx]

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        """BGR uint8 frame → normalized float32 tensor [1, 3, H, W]."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        tensor = self._normalize(tensor)
        return tensor.to(self.device)

    def _postprocess(self, tensor: "torch.Tensor") -> np.ndarray:
        """Tensor [1, 3, H, W] → BGR uint8 frame."""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _dream_step(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Single gradient ascent step — maximize activations at target layer."""
        tensor = tensor.detach().requires_grad_(True)

        if self._amp_enabled:
            with torch.amp.autocast("cuda"):
                activations = self.model(tensor, self.target_layer)
                loss = activations.norm()
        else:
            activations = self.model(tensor, self.target_layer)
            loss = activations.norm()

        loss.backward()

        grad = tensor.grad.data
        # Normalize gradient
        grad /= grad.abs().mean() + 1e-8
        tensor = tensor.detach() + grad * self.intensity

        return tensor

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Apply slow zoom toward center."""
        if self.zoom <= 1.0:
            return frame
        h, w = frame.shape[:2]
        # Crop a slightly smaller centered rectangle and scale back up
        dh = int(h * (1 - 1 / self.zoom) / 2)
        dw = int(w * (1 - 1 / self.zoom) / 2)
        if dh < 1 or dw < 1:
            return frame
        cropped = frame[dh:h - dh, dw:w - dw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _apply_jitter(self, tensor: "torch.Tensor") -> tuple["torch.Tensor", int, int]:
        """Random shift to reduce tiling artifacts."""
        if self.jitter <= 0:
            return tensor, 0, 0
        ox = np.random.randint(-self.jitter, self.jitter + 1)
        oy = np.random.randint(-self.jitter, self.jitter + 1)
        return torch.roll(tensor, shifts=(oy, ox), dims=(2, 3)), ox, oy

    def _undo_jitter(self, tensor: "torch.Tensor", ox: int, oy: int) -> "torch.Tensor":
        return torch.roll(tensor, shifts=(-oy, -ox), dims=(2, 3))

    def _apply_hue_shift(self, frame: np.ndarray) -> np.ndarray:
        """Rotate hue channel."""
        if self.hue_shift <= 0.001:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift * 180) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame through DeepDream.

        Args:
            frame: BGR uint8 numpy array from camera.

        Returns:
            Dreamed BGR uint8 frame.
        """
        h, w = frame.shape[:2]

        # Apply zoom to input
        frame = self._apply_zoom(frame)

        # Mix with previous output for feedback loop
        if self._prev_frame is not None and self.feedback > 0:
            prev = cv2.resize(self._prev_frame, (w, h))
            frame = cv2.addWeighted(frame, 1 - self.feedback, prev, self.feedback, 0)

        # In turbo mode, cap octaves/iterations for speed
        octaves = self.octaves
        iterations = self.iterations
        if self.turbo:
            octaves = min(octaves, 2)
            iterations = min(iterations, 3)

        # Multi-octave DeepDream
        base_tensor = self._preprocess(frame)
        detail_layers = []

        # Build octave pyramid
        octave_shapes = []
        for i in range(octaves):
            scale = self.octave_scale ** (octaves - 1 - i)
            oh = int(h / scale)
            ow = int(w / scale)
            octave_shapes.append((oh, ow))

        with torch.no_grad():
            for octave_idx, (oh, ow) in enumerate(octave_shapes):
                # Resize input to this octave's resolution
                if oh != h or ow != w:
                    octave_tensor = torch.nn.functional.interpolate(
                        base_tensor, size=(oh, ow), mode="bilinear", align_corners=False
                    )
                else:
                    octave_tensor = base_tensor.clone()

                # Add back detail from previous octave
                if detail_layers:
                    upscaled_detail = torch.nn.functional.interpolate(
                        detail_layers[-1], size=(oh, ow), mode="bilinear", align_corners=False
                    )
                    octave_tensor = octave_tensor + upscaled_detail

                # Gradient ascent iterations
                for _ in range(iterations):
                    with torch.enable_grad():
                        octave_tensor, ox, oy = self._apply_jitter(octave_tensor)
                        octave_tensor = self._dream_step(octave_tensor)
                        octave_tensor = self._undo_jitter(octave_tensor, ox, oy)

                # Extract detail (difference from input at this scale)
                if oh != h or ow != w:
                    input_at_scale = torch.nn.functional.interpolate(
                        base_tensor, size=(oh, ow), mode="bilinear", align_corners=False
                    )
                else:
                    input_at_scale = base_tensor
                detail = octave_tensor - input_at_scale
                detail_layers.append(detail)

        # Final result: original + accumulated detail at full resolution
        if detail_layers:
            final_detail = torch.nn.functional.interpolate(
                detail_layers[-1], size=(h, w), mode="bilinear", align_corners=False
            )
            result_tensor = base_tensor + final_detail
        else:
            result_tensor = base_tensor

        result = self._postprocess(result_tensor)

        # Post-processing
        result = self._apply_hue_shift(result)

        if self.blur_amount > 0.01:
            ksize = int(self.blur_amount * 15) * 2 + 1
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        self._prev_frame = result.copy()
        return result

    def update_params(self, params: dict):
        """Update dream parameters from a dict (typically from MIDI mapper)."""
        if "layer_index" in params:
            self.layer_index = int(params["layer_index"])
        if "intensity" in params:
            self.intensity = float(params["intensity"])
        if "octaves" in params:
            self.octaves = max(1, int(params["octaves"]))
        if "iterations" in params:
            self.iterations = max(1, int(params["iterations"]))
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

    def get_info(self) -> dict:
        """Return current engine state for HUD overlay."""
        info = {
            "engine": "DeepDream",
            "layer": self.target_layer,
            "intensity": f"{self.intensity:.3f}",
            "octaves": self.octaves,
            "iterations": self.iterations,
            "feedback": f"{self.feedback:.2f}",
            "zoom": f"{self.zoom:.4f}",
        }
        if self.turbo:
            info["turbo"] = "ON (AMP)" if self._amp_enabled else "ON"
        return info
