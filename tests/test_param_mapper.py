#!/usr/bin/env python3
"""Tests for the parameter mapper."""

import sys
from unittest.mock import MagicMock

import pytest

sys.modules["rtmidi"] = MagicMock()

from scripts.midi_launchpad import grid_to_note
from scripts.param_mapper import (
    MODE_BLEND,
    MODE_DREAM,
    MODE_STYLE,
    BTN_FREEZE,
    BTN_FULLSCREEN,
    BTN_HUD,
    BTN_NEXT_MODE,
    BTN_RESET,
    PRESETS,
    ROW_FEEDBACK,
    ROW_HUE,
    ROW_INTENSITY,
    ROW_LAYER,
    ROW_PRESETS,
    ROW_STYLE,
    ROW_ZOOM,
    ParamMapper,
)


def make_midi_state(**kwargs):
    """Build a minimal MIDI state dict."""
    state = {
        "pads_held": {},
        "pad_triggers": {},
        "top_buttons": {cc: False for cc in range(91, 99)},
        "top_triggers": {},
        "active_row": -1,
        "active_col": -1,
        "active_velocity": 0.0,
        "num_pads_held": 0,
    }
    state.update(kwargs)
    return state


class TestParamMapper:

    def setup_method(self):
        self.mapper = ParamMapper()

    def test_initial_mode(self):
        assert self.mapper.mode == MODE_DREAM

    def test_mode_cycle_next(self):
        state = make_midi_state(top_triggers={BTN_NEXT_MODE: True})
        self.mapper.process(state)
        assert self.mapper.mode == MODE_STYLE

        self.mapper.process(make_midi_state(top_triggers={BTN_NEXT_MODE: True}))
        assert self.mapper.mode == MODE_BLEND

        self.mapper.process(make_midi_state(top_triggers={BTN_NEXT_MODE: True}))
        assert self.mapper.mode == MODE_DREAM

    def test_layer_select_row(self):
        note = grid_to_note(ROW_LAYER, 3)  # col 3
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "layer_index" in updates

    def test_style_select_row(self):
        note = grid_to_note(ROW_STYLE, 2)
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "style_index" in updates
        assert updates["style_index"] == 2

    def test_intensity_row_dream_mode(self):
        self.mapper.mode = MODE_DREAM
        note = grid_to_note(ROW_INTENSITY, 7)  # max
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "intensity" in updates
        assert updates["intensity"] > 0.05

    def test_intensity_row_style_mode(self):
        self.mapper.mode = MODE_STYLE
        note = grid_to_note(ROW_INTENSITY, 4)
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "style_blend" in updates

    def test_feedback_row(self):
        note = grid_to_note(ROW_FEEDBACK, 5)
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "feedback" in updates
        assert 0 <= updates["feedback"] <= 1

    def test_hue_row(self):
        note = grid_to_note(ROW_HUE, 4)
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "hue_shift" in updates

    def test_zoom_row(self):
        note = grid_to_note(ROW_ZOOM, 6)
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert "zoom" in updates

    def test_preset_load(self):
        note = grid_to_note(ROW_PRESETS, 0)  # First preset
        state = make_midi_state(pad_triggers={note: 1.0})
        updates = self.mapper.process(state)
        assert self.mapper.preset_name == PRESETS[0]["name"]

    def test_freeze_toggle(self):
        assert self.mapper.freeze is False

        state = make_midi_state(top_triggers={BTN_FREEZE: True})
        self.mapper.process(state)
        assert self.mapper.freeze is True

        self.mapper.process(make_midi_state(top_triggers={BTN_FREEZE: True}))
        assert self.mapper.freeze is False

    def test_hud_toggle(self):
        state = make_midi_state(top_triggers={BTN_HUD: True})
        self.mapper.process(state)
        assert self.mapper.toggle_hud is True

    def test_fullscreen_toggle(self):
        state = make_midi_state(top_triggers={BTN_FULLSCREEN: True})
        self.mapper.process(state)
        assert self.mapper.toggle_fullscreen is True

    def test_reset(self):
        # Change some params first
        self.mapper.params["intensity"] = 0.99
        self.mapper.mode = MODE_BLEND

        state = make_midi_state(top_triggers={BTN_RESET: True})
        self.mapper.process(state)

        assert self.mapper.mode == MODE_DREAM
        assert self.mapper.params["intensity"] == 0.02

    def test_empty_state_no_crash(self):
        updates = self.mapper.process(make_midi_state())
        assert updates == {}

    def test_get_info(self):
        info = self.mapper.get_info()
        assert "mode" in info
        assert info["mode"] == "DeepDream"
