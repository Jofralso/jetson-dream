#!/usr/bin/env python3
"""
Fast Neural Style Transfer engine.

Uses pre-trained transformation networks (Johnson et al. 2016) for real-time
artistic style transfer. Each style model is a feed-forward network that
transforms input frames in a single pass — much faster than optimization-based
NST, making it suitable for live video.

Supports:
  - Multiple pre-trained style models (Mosaic, Candy, Udnie, Rain Princess, etc.)
  - Live switching between styles via MIDI
  - Style blending (lerp between original and styled)
  - TensorRT acceleration when available

The models are small (~7MB each) and run at 15-30+ FPS on Jetson Orin Nano.
"""

import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


# Built-in styles and their download URLs (PyTorch fast-neural-style models)
STYLE_MODELS = {
    "mosaic": {
        "filename": "mosaic.pth",
        "description": "Classic mosaic tiles",
    },
    "candy": {
        "filename": "candy.pth",
        "description": "Bright candy colors",
    },
    "rain_princess": {
        "filename": "rain_princess.pth",
        "description": "Dreamy rain painting",
    },
    "udnie": {
        "filename": "udnie.pth",
        "description": "Abstract geometric (Francis Picabia)",
    },
    "starry_night": {
        "filename": "starry_night.pth",
        "description": "Van Gogh swirls",
    },
    "la_muse": {
        "filename": "la_muse.pth",
        "description": "Picasso-esque abstraction",
    },
    "the_scream": {
        "filename": "the_scream.pth",
        "description": "Edvard Munch distortion",
    },
    "feathers": {
        "filename": "feathers.pth",
        "description": "Peacock feather patterns",
    },
}


class ConvBlock(nn.Module):
    """Convolution + InstanceNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              padding_mode="reflect")
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect")
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect")
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsample + Conv + Norm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              padding_mode="reflect")
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.relu(self.norm(self.conv(x)))


class TransformNet(nn.Module):
    """Feed-forward style transformation network (Johnson et al. 2016).

    Architecture: 3 downsampling convs → 5 residual blocks → 3 upsampling convs
    """

    def __init__(self):
        super().__init__()
        # Downsampling
        self.down1 = ConvBlock(3, 32, 9, 1)
        self.down2 = ConvBlock(32, 64, 3, 2)
        self.down3 = ConvBlock(64, 128, 3, 2)

        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling
        self.up1 = UpsampleBlock(128, 64, 3, 1, upsample=2)
        self.up2 = UpsampleBlock(64, 32, 3, 1, upsample=2)

        # Final output
        self.final = nn.Conv2d(32, 3, 9, 1, 4, padding_mode="reflect")

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x


class StyleTransferEngine:
    """Real-time neural style transfer for video frames."""

    def __init__(self, models_dir: str = "models", resolution: tuple[int, int] = (640, 480),
                 turbo: bool = False):
        if torch is None:
            raise RuntimeError("PyTorch is required: pip install torch torchvision")
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")

        self.models_dir = Path(models_dir)
        self.resolution = resolution
        self.turbo = turbo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loaded models cache
        self._models: dict[str, TransformNet] = {}
        self._current_style = None
        self._current_model: TransformNet | None = None
        self._turbo_engine = None  # TurboStyleEngine when turbo=True

        # Parameters (updated from MIDI)
        self.style_blend = 1.0        # 0 = original, 1 = fully styled
        self.feedback = 0.0           # Previous frame bleed
        self.hue_shift = 0.0          # Color rotation [0-1]
        self.saturation = 1.0         # Saturation multiplier [0-2]
        self.contrast = 1.0           # Contrast multiplier [0.5-2]

        self._prev_frame = None

        # Available style names
        self.style_names = list(STYLE_MODELS.keys())
        print(f"StyleTransfer: {len(self.style_names)} styles available")

    def load_style(self, style_name: str) -> bool:
        """Load a style model by name. Returns True if successful."""
        if style_name == self._current_style and self._current_model is not None:
            return True

        # Check cache first
        if style_name in self._models:
            self._current_model = self._models[style_name]
            self._current_style = style_name
            print(f"StyleTransfer: switched to '{style_name}' (cached)")
            return True

        # Try to load from disk
        if style_name in STYLE_MODELS:
            model_path = self.models_dir / STYLE_MODELS[style_name]["filename"]
        else:
            # Custom model — look for .pth file
            model_path = self.models_dir / f"{style_name}.pth"

        if not model_path.exists():
            print(f"StyleTransfer: model not found: {model_path}")
            print(f"  Download style models to {self.models_dir}/")
            return False

        try:
            model = TransformNet()
            state_dict = torch.load(str(model_path), map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self._models[style_name] = model
            self._current_model = model
            self._current_style = style_name
            print(f"StyleTransfer: loaded '{style_name}' from {model_path}")

            # Build turbo engine if enabled
            if self.turbo and torch.cuda.is_available():
                try:
                    from scripts.turbo_engine import TurboStyleEngine
                    h = self.resolution[1] if len(self.resolution) > 1 else self.resolution[0]
                    w = self.resolution[0]
                    self._turbo_engine = TurboStyleEngine(
                        model, (h, w),
                        cache_dir=str(self.models_dir / "trt_cache"),
                    )
                    print(f"StyleTransfer: turbo engine ready ({self._turbo_engine.optimization_info})")
                except Exception as e:
                    print(f"StyleTransfer: turbo build failed ({e}), using standard FP16")
                    self._turbo_engine = None
                    if torch.cuda.is_available():
                        self._current_model.half()

            return True
        except Exception as e:
            print(f"StyleTransfer: failed to load '{style_name}': {e}")
            return False

    def select_style_by_index(self, index: int) -> bool:
        """Select style by numeric index (for MIDI pad mapping)."""
        if 0 <= index < len(self.style_names):
            return self.load_style(self.style_names[index])
        return False

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        """BGR uint8 → float32 tensor [1, 3, H, W] in [0, 255] range."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, tensor: "torch.Tensor") -> np.ndarray:
        """Tensor [1, 3, H, W] → BGR uint8."""
        img = tensor.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply style transfer to a single video frame.

        Args:
            frame: BGR uint8 numpy array.

        Returns:
            Styled BGR uint8 frame.
        """
        if self._current_model is None:
            return frame

        h, w = frame.shape[:2]

        # Mix with previous output for feedback
        if self._prev_frame is not None and self.feedback > 0:
            prev = cv2.resize(self._prev_frame, (w, h))
            frame = cv2.addWeighted(frame, 1 - self.feedback, prev, self.feedback, 0)

        # Run style transfer
        with torch.no_grad():
            input_tensor = self._preprocess(frame)
            if self._turbo_engine is not None:
                output_tensor = self._turbo_engine.forward(input_tensor)
            elif self.turbo and torch.cuda.is_available():
                # FP16 fallback
                output_tensor = self._current_model(input_tensor.half()).float()
            else:
                output_tensor = self._current_model(input_tensor)
            styled = self._postprocess(output_tensor)

        # Blend original with styled
        if self.style_blend < 0.99:
            styled = cv2.addWeighted(frame, 1 - self.style_blend, styled, self.style_blend, 0)

        # Post-processing: saturation + contrast
        if abs(self.saturation - 1.0) > 0.01 or abs(self.contrast - 1.0) > 0.01:
            hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation, 0, 255)
            hsv[:, :, 2] = np.clip((hsv[:, :, 2] - 128) * self.contrast + 128, 0, 255)
            if self.hue_shift > 0.001:
                hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift * 180) % 180
            styled = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        elif self.hue_shift > 0.001:
            hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift * 180) % 180
            styled = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        self._prev_frame = styled.copy()
        return styled

    def update_params(self, params: dict):
        """Update style parameters from a dict."""
        if "style_index" in params:
            self.select_style_by_index(int(params["style_index"]))
        if "style_name" in params:
            self.load_style(params["style_name"])
        if "style_blend" in params:
            self.style_blend = float(np.clip(params["style_blend"], 0, 1))
        if "feedback" in params:
            self.feedback = float(np.clip(params["feedback"], 0, 1))
        if "hue_shift" in params:
            self.hue_shift = float(params["hue_shift"])
        if "saturation" in params:
            self.saturation = float(np.clip(params["saturation"], 0, 2))
        if "contrast" in params:
            self.contrast = float(np.clip(params["contrast"], 0.5, 2))

    def get_info(self) -> dict:
        """Current engine state for HUD."""
        return {
            "engine": "StyleTransfer",
            "style": self._current_style or "none",
            "blend": f"{self.style_blend:.2f}",
            "feedback": f"{self.feedback:.2f}",
            "saturation": f"{self.saturation:.2f}",
        }
