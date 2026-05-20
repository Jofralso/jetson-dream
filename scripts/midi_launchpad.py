#!/usr/bin/env python3
"""
Novation Launchpad MK3 MIDI engine.

Handles the Launchpad MK3 in Programmer mode:
  - 8×8 RGB pad grid (notes 11-19, 21-29, ..., 81-89)
  - Top row function buttons (CC 91-98)
  - Logo button (CC 99)
  - Pad LED feedback via SysEx

Pad grid layout (Programmer Mode):
  Row 8: 81 82 83 84 85 86 87 88   (top row of pads)
  Row 7: 71 72 73 74 75 76 77 78
  Row 6: 61 62 63 64 65 66 67 68
  Row 5: 51 52 53 54 55 56 57 58
  Row 4: 41 42 43 44 45 46 47 48
  Row 3: 31 32 33 34 35 36 37 38
  Row 2: 21 22 23 24 25 26 27 28
  Row 1: 11 12 13 14 15 16 17 18   (bottom row of pads)
"""

import threading
import time

try:
    import rtmidi
except ImportError:
    rtmidi = None


# Novation Launchpad MK3 SysEx header
SYSEX_HEADER = [0x00, 0x20, 0x29, 0x02, 0x0D]

# Programmer mode enable SysEx
PROGRAMMER_MODE = [0xF0] + SYSEX_HEADER + [0x0E, 0x01, 0xF7]

# Pad color palette — Launchpad uses a 128-color index palette
# Key colors for visual feedback
COLOR_OFF = 0
COLOR_WHITE = 3
COLOR_RED = 5
COLOR_ORANGE = 9
COLOR_YELLOW = 13
COLOR_GREEN = 21
COLOR_CYAN = 37
COLOR_BLUE = 45
COLOR_PURPLE = 53
COLOR_PINK = 57
COLOR_DIM_WHITE = 1
COLOR_DIM_RED = 7
COLOR_DIM_GREEN = 23
COLOR_DIM_BLUE = 47


def note_to_grid(note: int) -> tuple[int, int] | None:
    """Convert MIDI note to (row, col) grid position.

    Row 0 = bottom, Row 7 = top.
    Col 0 = left, Col 7 = right.
    Returns None if note is outside the 8x8 grid.
    """
    row = (note // 10) - 1
    col = (note % 10) - 1
    if 0 <= row <= 7 and 0 <= col <= 7:
        return (row, col)
    return None


def grid_to_note(row: int, col: int) -> int:
    """Convert (row, col) to MIDI note number."""
    return (row + 1) * 10 + (col + 1)


class LaunchpadMidi:
    """Novation Launchpad MK3 MIDI controller interface."""

    def __init__(self, port_name: str | None = None):
        self.port_name = port_name

        # Pad state — which pads are held and their velocities
        self.pads_held: dict[int, float] = {}     # note → velocity [0-1]
        self.pad_triggers: dict[int, float] = {}  # note → velocity (one-shot per frame)

        # Top row buttons (CC 91-98)
        self.top_buttons: dict[int, bool] = {cc: False for cc in range(91, 99)}
        self.top_triggers: dict[int, bool] = {}

        # Continuous state derived from pad grid
        self.active_row = -1       # Which row has the most recent press
        self.active_col = -1       # Which col has the most recent press
        self.active_velocity = 0.0
        self.num_pads_held = 0

        self._midi_in = None
        self._midi_out = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._midi_in is not None and self._running

    def _find_port(self, midi_obj, direction: str) -> int | None:
        """Find Launchpad MK3 port index."""
        ports = midi_obj.get_ports()
        for i, name in enumerate(ports):
            name_lower = name.lower()
            # Launchpad MK3 shows as "Launchpad MK3" or "Launchpad Mini MK3"
            if "launchpad" in name_lower and ("mk3" in name_lower or "mk 3" in name_lower):
                # Prefer the MIDI port (port 1 is the DAW port, port 0 is the
                # "MIDI" port on MK3 — we want the second one for programmer mode)
                if self.port_name and self.port_name.lower() not in name_lower:
                    continue
                print(f"  Found Launchpad {direction}: [{i}] {name}")
                return i
            if self.port_name and self.port_name.lower() in name_lower:
                print(f"  Found MIDI {direction}: [{i}] {name}")
                return i
        return None

    def _midi_callback(self, event, data=None):
        """Process incoming MIDI from the Launchpad."""
        message, _ = event
        if len(message) < 2:
            return

        status = message[0] & 0xF0
        channel = message[0] & 0x0F

        with self._lock:
            if status == 0x90 and len(message) >= 3:
                # Note On — pad press
                note = message[1]
                velocity = message[2] / 127.0

                grid = note_to_grid(note)
                if grid is None:
                    return

                if velocity > 0:
                    self.pads_held[note] = velocity
                    self.pad_triggers[note] = velocity
                    self.active_row, self.active_col = grid
                    self.active_velocity = velocity
                else:
                    self.pads_held.pop(note, None)

                self.num_pads_held = len(self.pads_held)

            elif status == 0x80 and len(message) >= 3:
                # Note Off — pad release
                note = message[1]
                self.pads_held.pop(note, None)
                self.num_pads_held = len(self.pads_held)

            elif status == 0xB0 and len(message) >= 3:
                # Control Change — top row buttons
                cc = message[1]
                value = message[2]

                if 91 <= cc <= 98:
                    pressed = value > 0
                    self.top_buttons[cc] = pressed
                    if pressed:
                        self.top_triggers[cc] = True

    def get_state(self) -> dict:
        """Thread-safe snapshot of current Launchpad state."""
        with self._lock:
            state = {
                "pads_held": dict(self.pads_held),
                "pad_triggers": dict(self.pad_triggers),
                "top_buttons": dict(self.top_buttons),
                "top_triggers": dict(self.top_triggers),
                "active_row": self.active_row,
                "active_col": self.active_col,
                "active_velocity": self.active_velocity,
                "num_pads_held": self.num_pads_held,
            }
            # Reset one-shot triggers
            self.pad_triggers.clear()
            self.top_triggers.clear()
            return state

    def set_pad_color(self, note: int, color: int):
        """Set a single pad LED color by palette index."""
        if self._midi_out is None:
            return
        # Static LED: Note On on channel 1 (status 0x90)
        self._midi_out.send_message([0x90, note, color])

    def set_pad_rgb(self, note: int, r: int, g: int, b: int):
        """Set pad LED to exact RGB color via SysEx.

        r, g, b in range [0, 127].
        """
        if self._midi_out is None:
            return
        msg = [0xF0] + SYSEX_HEADER + [0x03, 0x03, note, r, g, b, 0xF7]
        self._midi_out.send_message(msg)

    def set_all_pads(self, color: int):
        """Set all 64 pads to the same palette color."""
        for row in range(1, 9):
            for col in range(1, 9):
                self.set_pad_color(row * 10 + col, color)

    def clear_pads(self):
        """Turn off all pad LEDs."""
        self.set_all_pads(COLOR_OFF)

    def set_row_colors(self, row: int, colors: list[int]):
        """Set colors for an entire row (0=bottom, 7=top)."""
        for col in range(min(8, len(colors))):
            note = grid_to_note(row, col)
            self.set_pad_color(note, colors[col])

    def _enter_programmer_mode(self):
        """Switch Launchpad to Programmer mode via SysEx."""
        if self._midi_out is None:
            return
        self._midi_out.send_message(PROGRAMMER_MODE)
        time.sleep(0.1)
        print("  Launchpad switched to Programmer mode")

    def start(self):
        """Open MIDI ports and start listening."""
        if rtmidi is None:
            print("Launchpad: python-rtmidi not installed — MIDI disabled")
            return

        # Open MIDI Input
        try:
            self._midi_in = rtmidi.MidiIn()
        except Exception as e:
            # ALSA sequencer may not be available over SSH
            error_str = str(e).lower()
            if "alsa" in error_str or "sequencer" in error_str:
                print("⚠ ALSA sequencer not available (SSH/headless environment)")
                print("  Trying direct USB MIDI access...")
                try:
                    # Try with dummy sequencer backend
                    self._midi_in = rtmidi.MidiIn()
                except Exception as e2:
                    print(f"  MIDI unavailable: {e2}")
                    print("  Run with --no-midi to skip")
                    return
            else:
                print(f"MIDI error: {e}")
                return

        in_ports = self._midi_in.get_ports()
        print(f"MIDI Input ports: {in_ports}")

        in_idx = self._find_port(self._midi_in, "input")
        if in_idx is None:
            print("Launchpad MK3 not found on MIDI input")
            print("  Available:", in_ports)
            return

        self._midi_in.open_port(in_idx)
        self._midi_in.set_callback(self._midi_callback)

        # Open MIDI Output (for LED feedback)
        try:
            self._midi_out = rtmidi.MidiOut()
        except Exception as e:
            print(f"⚠ MIDI output error: {e}")
            self._midi_out = None
            return

        out_ports = self._midi_out.get_ports()
        print(f"MIDI Output ports: {out_ports}")

        out_idx = self._find_port(self._midi_out, "output")
        if out_idx is not None:
            self._midi_out.open_port(out_idx)
            self._enter_programmer_mode()
        else:
            print("  Launchpad output not found — LED feedback disabled")
            self._midi_out = None

        self._running = True
        print("Launchpad MK3 MIDI engine started")

    def stop(self):
        """Close MIDI ports."""
        self._running = False
        if self._midi_out:
            self.clear_pads()
        if self._midi_in:
            self._midi_in.close_port()
            self._midi_in = None
        if self._midi_out:
            self._midi_out.close_port()
            self._midi_out = None

    def list_ports(self):
        """Print all available MIDI ports."""
        if rtmidi is None:
            print("python-rtmidi not installed")
            return
        tmp_in = rtmidi.MidiIn()
        tmp_out = rtmidi.MidiOut()
        print("MIDI Input ports:")
        for i, name in enumerate(tmp_in.get_ports()):
            print(f"  [{i}] {name}")
        print("MIDI Output ports:")
        for i, name in enumerate(tmp_out.get_ports()):
            print(f"  [{i}] {name}")
        del tmp_in, tmp_out
