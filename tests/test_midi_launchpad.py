#!/usr/bin/env python3
"""Tests for Launchpad MK3 MIDI engine."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules["rtmidi"] = MagicMock()

from scripts.midi_launchpad import (
    COLOR_OFF,
    COLOR_RED,
    LaunchpadMidi,
    grid_to_note,
    note_to_grid,
)


class TestNoteGrid:
    """Test note ↔ grid coordinate conversion."""

    def test_note_to_grid_bottom_left(self):
        assert note_to_grid(11) == (0, 0)

    def test_note_to_grid_bottom_right(self):
        assert note_to_grid(18) == (0, 7)

    def test_note_to_grid_top_left(self):
        assert note_to_grid(81) == (7, 0)

    def test_note_to_grid_top_right(self):
        assert note_to_grid(88) == (7, 7)

    def test_note_to_grid_center(self):
        assert note_to_grid(55) == (4, 4)

    def test_note_to_grid_out_of_range(self):
        assert note_to_grid(0) is None
        assert note_to_grid(99) is None
        assert note_to_grid(10) is None  # col 0 is valid but row 0 is row index 0
        assert note_to_grid(19) is None  # col 9 is out of range

    def test_grid_to_note_roundtrip(self):
        for row in range(8):
            for col in range(8):
                note = grid_to_note(row, col)
                assert note_to_grid(note) == (row, col)

    def test_grid_to_note_values(self):
        assert grid_to_note(0, 0) == 11
        assert grid_to_note(7, 7) == 88
        assert grid_to_note(3, 5) == 46


class TestLaunchpadMidi:
    """Test Launchpad MIDI message handling."""

    def setup_method(self):
        self.lp = LaunchpadMidi()

    def test_initial_state(self):
        state = self.lp.get_state()
        assert state["pads_held"] == {}
        assert state["pad_triggers"] == {}
        assert state["num_pads_held"] == 0
        assert state["active_row"] == -1

    def test_note_on_pad_press(self):
        # Simulate Note On: note 55, velocity 100
        event = ([0x90, 55, 100], 0.0)
        self.lp._midi_callback(event)

        state = self.lp.get_state()
        assert 55 in state["pads_held"]
        assert abs(state["pads_held"][55] - 100 / 127) < 0.01
        assert 55 in state["pad_triggers"]
        assert state["active_row"] == 4
        assert state["active_col"] == 4
        assert state["num_pads_held"] == 1

    def test_note_off_pad_release(self):
        # Press then release
        self.lp._midi_callback(([0x90, 55, 100], 0.0))
        self.lp._midi_callback(([0x80, 55, 0], 0.0))

        state = self.lp.get_state()
        assert 55 not in state["pads_held"]
        assert state["num_pads_held"] == 0

    def test_note_on_velocity_zero_is_off(self):
        self.lp._midi_callback(([0x90, 55, 100], 0.0))
        self.lp._midi_callback(([0x90, 55, 0], 0.0))  # vel 0 = note off

        state = self.lp.get_state()
        assert 55 not in state["pads_held"]

    def test_cc_top_button(self):
        # CC 95 (Session button) pressed
        self.lp._midi_callback(([0xB0, 95, 127], 0.0))

        state = self.lp.get_state()
        assert state["top_buttons"][95] is True
        assert 95 in state["top_triggers"]

    def test_triggers_clear_after_read(self):
        self.lp._midi_callback(([0x90, 55, 100], 0.0))
        state1 = self.lp.get_state()
        assert 55 in state1["pad_triggers"]

        state2 = self.lp.get_state()
        assert state2["pad_triggers"] == {}

    def test_multiple_pads(self):
        self.lp._midi_callback(([0x90, 11, 80], 0.0))
        self.lp._midi_callback(([0x90, 55, 100], 0.0))
        self.lp._midi_callback(([0x90, 88, 60], 0.0))

        state = self.lp.get_state()
        assert state["num_pads_held"] == 3
        assert 11 in state["pads_held"]
        assert 55 in state["pads_held"]
        assert 88 in state["pads_held"]

    def test_out_of_grid_note_ignored(self):
        # Note 99 is outside the 8x8 grid
        self.lp._midi_callback(([0x90, 99, 100], 0.0))

        state = self.lp.get_state()
        assert state["num_pads_held"] == 0
        assert state["pad_triggers"] == {}
