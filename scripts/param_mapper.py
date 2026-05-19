#!/usr/bin/env python3
"""
Launchpad MK3 → Dream parameter mapper.

Maps the Launchpad's 8×8 grid and top buttons to AI engine parameters.
Designed for intuitive live performance control.

═══════════════════════════════════════════════════════════════
  LAUNCHPAD MK3 LAYOUT (Programmer Mode)
═══════════════════════════════════════════════════════════════

  Top buttons (CC 91-98):
  [▲] [▼] [◄] [►] [Session] [Drums] [Keys] [User]
   91   92  93  94    95       96      97     98

  Pad grid (notes):
  Row 8: 81 82 83 84 85 86 87 88    ← LAYER SELECT (dream layer 0-7)
  Row 7: 71 72 73 74 75 76 77 78    ← STYLE SELECT (style models 0-7)
  Row 6: 61 62 63 64 65 66 67 68    ← INTENSITY (8 levels)
  Row 5: 51 52 53 54 55 56 57 58    ← FEEDBACK (8 levels)
  Row 4: 41 42 43 44 45 46 47 48    ← HUE SHIFT (8 positions)
  Row 3: 31 32 33 34 35 36 37 38    ← ITERATIONS / BLUR (8 levels)
  Row 2: 21 22 23 24 25 26 27 28    ← ZOOM SPEED (8 levels)
  Row 1: 11 12 13 14 15 16 17 18    ← PRESETS / TRIGGERS

═══════════════════════════════════════════════════════════════
"""

from scripts.midi_launchpad import (
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_DIM_WHITE,
    COLOR_GREEN,
    COLOR_OFF,
    COLOR_ORANGE,
    COLOR_PINK,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    grid_to_note,
    note_to_grid,
)

# Engine modes
MODE_DREAM = 0
MODE_STYLE = 1
MODE_BLEND = 2  # Both engines mixed

MODE_NAMES = {
    MODE_DREAM: "DeepDream",
    MODE_STYLE: "StyleTransfer",
    MODE_BLEND: "Dream+Style",
}

# Row assignments (row 0 = bottom, row 7 = top)
ROW_PRESETS = 0       # Row 1: presets and triggers
ROW_ZOOM = 1          # Row 2: zoom speed
ROW_ITER_BLUR = 2     # Row 3: iterations / blur
ROW_HUE = 3           # Row 4: hue shift
ROW_FEEDBACK = 4      # Row 5: feedback amount
ROW_INTENSITY = 5     # Row 6: intensity / blend
ROW_STYLE = 6         # Row 7: style model select
ROW_LAYER = 7         # Row 8: dream layer select

# Row colors for LED feedback
ROW_COLORS = {
    ROW_PRESETS: COLOR_WHITE,
    ROW_ZOOM: COLOR_CYAN,
    ROW_ITER_BLUR: COLOR_BLUE,
    ROW_HUE: COLOR_PINK,
    ROW_FEEDBACK: COLOR_PURPLE,
    ROW_INTENSITY: COLOR_ORANGE,
    ROW_STYLE: COLOR_GREEN,
    ROW_LAYER: COLOR_RED,
}

# Top button assignments
BTN_PREV_MODE = 91    # ▲ — previous engine mode
BTN_NEXT_MODE = 92    # ▼ — next engine mode
BTN_DEC = 93          # ◄ — decrease octaves
BTN_INC = 94          # ► — increase octaves
BTN_RESET = 95        # Session — reset all params
BTN_FREEZE = 96       # Drums — freeze frame (toggle)
BTN_HUD = 97          # Keys — toggle HUD
BTN_FULLSCREEN = 98   # User — toggle fullscreen

# Preset definitions (row 1 pads)
PRESETS = {
    0: {  # Subtle dream
        "name": "Subtle",
        "mode": MODE_DREAM,
        "layer_index": 3,
        "intensity": 0.01,
        "octaves": 2,
        "iterations": 3,
        "feedback": 0.1,
        "zoom": 1.001,
    },
    1: {  # Deep dream
        "name": "Deep",
        "mode": MODE_DREAM,
        "layer_index": 6,
        "intensity": 0.03,
        "octaves": 4,
        "iterations": 8,
        "feedback": 0.4,
        "zoom": 1.003,
    },
    2: {  # Psychedelic
        "name": "Psychedelic",
        "mode": MODE_DREAM,
        "layer_index": 8,
        "intensity": 0.05,
        "octaves": 3,
        "iterations": 10,
        "feedback": 0.6,
        "zoom": 1.005,
        "hue_shift": 0.02,
    },
    3: {  # Gentle style
        "name": "Gentle Style",
        "mode": MODE_STYLE,
        "style_blend": 0.5,
        "feedback": 0.1,
    },
    4: {  # Full style
        "name": "Full Style",
        "mode": MODE_STYLE,
        "style_blend": 1.0,
        "feedback": 0.3,
    },
    5: {  # Dream + Style blend
        "name": "Dream+Style",
        "mode": MODE_BLEND,
        "layer_index": 4,
        "intensity": 0.015,
        "style_blend": 0.5,
        "feedback": 0.3,
    },
    6: {  # Infinite zoom
        "name": "Inf. Zoom",
        "mode": MODE_DREAM,
        "layer_index": 5,
        "intensity": 0.02,
        "feedback": 0.7,
        "zoom": 1.01,
    },
    7: {  # Chaos
        "name": "Chaos",
        "mode": MODE_BLEND,
        "layer_index": 9,
        "intensity": 0.06,
        "octaves": 5,
        "iterations": 15,
        "feedback": 0.8,
        "zoom": 1.008,
        "hue_shift": 0.05,
    },
}


class ParamMapper:
    """Maps Launchpad MK3 state to AI engine parameters."""

    def __init__(self, launchpad=None):
        self.launchpad = launchpad

        # Current engine mode
        self.mode = MODE_DREAM

        # Current parameter values
        self.params = {
            # DeepDream
            "layer_index": 4,
            "intensity": 0.02,
            "octaves": 3,
            "iterations": 5,
            "feedback": 0.3,
            "zoom": 1.002,
            "jitter": 16,
            "hue_shift": 0.0,
            "blur_amount": 0.0,
            # Style Transfer
            "style_index": 0,
            "style_blend": 1.0,
            "saturation": 1.0,
            "contrast": 1.0,
        }

        # UI state
        self.freeze = False
        self.toggle_hud = False
        self.toggle_fullscreen = False
        self.preset_name = ""

        # Track active selections per row for LED feedback
        self._row_selection = {row: -1 for row in range(8)}

    def _col_to_value(self, col: int, low: float, high: float) -> float:
        """Map column (0-7) to a value in [low, high]."""
        return low + (col / 7.0) * (high - low)

    def process(self, midi_state: dict) -> dict:
        """Process Launchpad state and return updated AI parameters.

        Args:
            midi_state: Output from LaunchpadMidi.get_state()

        Returns:
            Dict of parameter updates to apply to engines.
        """
        updates = {}

        # Reset one-shot UI flags
        self.toggle_hud = False
        self.toggle_fullscreen = False

        # Process top button triggers
        for cc, triggered in midi_state.get("top_triggers", {}).items():
            if not triggered:
                continue
            if cc == BTN_NEXT_MODE:
                self.mode = (self.mode + 1) % len(MODE_NAMES)
                print(f"Mode: {MODE_NAMES[self.mode]}")
            elif cc == BTN_PREV_MODE:
                self.mode = (self.mode - 1) % len(MODE_NAMES)
                print(f"Mode: {MODE_NAMES[self.mode]}")
            elif cc == BTN_INC:
                self.params["octaves"] = min(6, self.params["octaves"] + 1)
                updates["octaves"] = self.params["octaves"]
                print(f"Octaves: {self.params['octaves']}")
            elif cc == BTN_DEC:
                self.params["octaves"] = max(1, self.params["octaves"] - 1)
                updates["octaves"] = self.params["octaves"]
                print(f"Octaves: {self.params['octaves']}")
            elif cc == BTN_RESET:
                self._reset_params()
                updates = dict(self.params)
                print("Parameters reset")
            elif cc == BTN_FREEZE:
                self.freeze = not self.freeze
                print(f"Freeze: {'ON' if self.freeze else 'OFF'}")
            elif cc == BTN_HUD:
                self.toggle_hud = True
            elif cc == BTN_FULLSCREEN:
                self.toggle_fullscreen = True

        # Process pad triggers
        for note, velocity in midi_state.get("pad_triggers", {}).items():
            grid = note_to_grid(note)
            if grid is None:
                continue
            row, col = grid

            # Update row selection for LED feedback
            self._row_selection[row] = col

            if row == ROW_LAYER:
                # Dream layer select (0-7 maps to layer 0-9)
                idx = int(col * 9 / 7)
                self.params["layer_index"] = idx
                updates["layer_index"] = idx

            elif row == ROW_STYLE:
                # Style model select
                self.params["style_index"] = col
                updates["style_index"] = col

            elif row == ROW_INTENSITY:
                # Intensity / style blend
                if self.mode == MODE_STYLE:
                    val = self._col_to_value(col, 0.0, 1.0)
                    self.params["style_blend"] = val
                    updates["style_blend"] = val
                else:
                    val = self._col_to_value(col, 0.005, 0.08)
                    self.params["intensity"] = val
                    updates["intensity"] = val

            elif row == ROW_FEEDBACK:
                val = self._col_to_value(col, 0.0, 0.9)
                self.params["feedback"] = val
                updates["feedback"] = val

            elif row == ROW_HUE:
                val = self._col_to_value(col, 0.0, 0.1)
                self.params["hue_shift"] = val
                updates["hue_shift"] = val

            elif row == ROW_ITER_BLUR:
                if self.mode == MODE_STYLE:
                    val = self._col_to_value(col, 0.0, 1.0)
                    self.params["blur_amount"] = val
                    updates["blur_amount"] = val
                else:
                    val = int(self._col_to_value(col, 1, 15))
                    self.params["iterations"] = val
                    updates["iterations"] = val

            elif row == ROW_ZOOM:
                val = self._col_to_value(col, 1.0, 1.015)
                self.params["zoom"] = val
                updates["zoom"] = val

            elif row == ROW_PRESETS:
                if col in PRESETS:
                    self._apply_preset(col)
                    updates = dict(self.params)
                    self.preset_name = PRESETS[col]["name"]
                    print(f"Preset: {self.preset_name}")

        # Velocity-based intensity modulation: if pads are held,
        # the average velocity scales the current intensity
        if midi_state.get("num_pads_held", 0) > 1:
            avg_vel = sum(midi_state["pads_held"].values()) / len(midi_state["pads_held"])
            updates["_velocity_mod"] = avg_vel

        return updates

    def _apply_preset(self, preset_idx: int):
        """Load a preset into current params."""
        preset = PRESETS.get(preset_idx, {})
        if "mode" in preset:
            self.mode = preset["mode"]
        for key, value in preset.items():
            if key in ("name", "mode"):
                continue
            if key in self.params:
                self.params[key] = value

    def _reset_params(self):
        """Reset all parameters to defaults."""
        self.params = {
            "layer_index": 4,
            "intensity": 0.02,
            "octaves": 3,
            "iterations": 5,
            "feedback": 0.3,
            "zoom": 1.002,
            "jitter": 16,
            "hue_shift": 0.0,
            "blur_amount": 0.0,
            "style_index": 0,
            "style_blend": 1.0,
            "saturation": 1.0,
            "contrast": 1.0,
        }
        self.mode = MODE_DREAM
        self.freeze = False
        self.preset_name = ""

    def update_leds(self):
        """Update Launchpad LEDs to reflect current state."""
        if self.launchpad is None or not self.launchpad.available:
            return

        # Light up each row with dim color, active selection bright
        for row in range(8):
            base_color = ROW_COLORS.get(row, COLOR_DIM_WHITE)
            active_col = self._row_selection[row]

            for col in range(8):
                note = grid_to_note(row, col)
                if col == active_col:
                    self.launchpad.set_pad_color(note, base_color)
                else:
                    self.launchpad.set_pad_color(note, COLOR_DIM_WHITE)

    def get_info(self) -> dict:
        """State summary for HUD."""
        info = {
            "mode": MODE_NAMES.get(self.mode, "?"),
        }
        if self.preset_name:
            info["preset"] = self.preset_name
        if self.freeze:
            info["FROZEN"] = "YES"
        return info
