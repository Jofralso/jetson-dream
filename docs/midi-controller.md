# MIDI Controller Reference

## Novation Launchpad MK3 Integration

jetson-dream uses the Launchpad MK3 in **Programmer Mode** for direct MIDI note/CC access to the 8×8 pad grid and top function buttons.

### Connection

The Launchpad is auto-detected on USB MIDI. The system searches for ports containing "Launchpad" and "MK3" in the name.

```bash
# List available MIDI ports
python scripts/main.py --list-devices

# Specify port name substring if multiple MIDI devices present
python scripts/main.py --midi-port "Launchpad MK3"

# Run without MIDI (keyboard-only control)
python scripts/main.py --no-midi
```

### Programmer Mode

On connection, the system sends a SysEx message to switch the Launchpad to Programmer Mode. In this mode:

- The 8×8 pad grid sends Note On/Off messages (notes 11-88)
- The top row sends Control Change messages (CC 91-98)
- All 64 pads can be individually lit via MIDI feedback

---

## Pad Grid Layout

```
         ┌─────────────────────────────────────────────────────────┐
         │  [▲]   [▼]   [◄]   [►]   [Ses]  [Drm]  [Key]  [Usr]  │
         │  CC91  CC92  CC93  CC94  CC95   CC96   CC97   CC98    │
         │  prev  next  oct-  oct+  reset  freeze HUD    F/S     │
         │  mode  mode                                            │
         ├─────────────────────────────────────────────────────────┤
         │                                                         │
Row 8    │  [81]  [82]  [83]  [84]  [85]  [86]  [87]  [88]      │  ← LAYER SELECT
         │  RED: dream layer 0──────────────────────────────9     │
         │                                                         │
Row 7    │  [71]  [72]  [73]  [74]  [75]  [76]  [77]  [78]      │  ← STYLE SELECT
         │  GREEN: style model 0────────────────────────────7     │
         │                                                         │
Row 6    │  [61]  [62]  [63]  [64]  [65]  [66]  [67]  [68]      │  ← INTENSITY
         │  ORANGE: intensity 0.005─────────────────────0.08      │
         │  (in Style mode: style_blend 0.0──────────────1.0)     │
         │                                                         │
Row 5    │  [51]  [52]  [53]  [54]  [55]  [56]  [57]  [58]      │  ← FEEDBACK
         │  PURPLE: feedback 0.0────────────────────────0.9       │
         │                                                         │
Row 4    │  [41]  [42]  [43]  [44]  [45]  [46]  [47]  [48]      │  ← HUE SHIFT
         │  PINK: hue shift 0.0─────────────────────────0.1       │
         │                                                         │
Row 3    │  [31]  [32]  [33]  [34]  [35]  [36]  [37]  [38]      │  ← ITERATIONS/BLUR
         │  BLUE: iterations 1──────────────────────────15        │
         │  (in Style mode: blur_amount 0.0──────────────1.0)     │
         │                                                         │
Row 2    │  [21]  [22]  [23]  [24]  [25]  [26]  [27]  [28]      │  ← ZOOM SPEED
         │  CYAN: zoom 1.0──────────────────────────────1.015     │
         │                                                         │
Row 1    │  [11]  [12]  [13]  [14]  [15]  [16]  [17]  [18]      │  ← PRESETS
         │  WHITE: preset 0─────────────────────────────7         │
         │                                                         │
         └─────────────────────────────────────────────────────────┘
```

---

## Top Button Functions

| Button | CC | Symbol | Action |
|--------|-----|--------|--------|
| Arrow Up | 91 | ▲ | Previous engine mode (Dream ← Style ← Blend) |
| Arrow Down | 92 | ▼ | Next engine mode (Dream → Style → Blend) |
| Arrow Left | 93 | ◄ | Decrease octaves (min 1) |
| Arrow Right | 94 | ► | Increase octaves (max 6) |
| Session | 95 | Ses | Reset all parameters to defaults |
| Drums | 96 | Drm | Toggle frame freeze |
| Keys | 97 | Key | Toggle HUD overlay |
| User | 98 | Usr | Toggle fullscreen |

---

## Row-by-Row Parameter Mapping

### Row 8 — Layer Select (Red LEDs)

Controls which InceptionV3 layer DeepDream targets. The column maps linearly to layer index 0-9:

| Pad | Layer Index | Layer Name | Effect |
|-----|------------|-----------|--------|
| Col 0 | 0 | Conv2d_1a_3x3 | Edges |
| Col 1 | 1 | Conv2d_2b_3x3 | Textures |
| Col 2 | 2-3 | Conv2d_3b/4a | Patterns |
| Col 3 | 3-4 | Mixed_5b | Eyes/spirals |
| Col 4 | 5 | Mixed_5c | Animal features |
| Col 5 | 6 | Mixed_5d | Shapes |
| Col 6 | 7-8 | Mixed_6a/6b | Abstract |
| Col 7 | 9 | Mixed_7a | Full hallucination |

**Performance note:** Shallower layers (left side) are faster because the model exits early.

### Row 7 — Style Select (Green LEDs)

Selects the active style transfer model (0-7):

| Pad | Style |
|-----|-------|
| Col 0 | Mosaic |
| Col 1 | Candy |
| Col 2 | Rain Princess |
| Col 3 | Udnie |
| Col 4 | Starry Night |
| Col 5 | La Muse |
| Col 6 | The Scream |
| Col 7 | Feathers |

### Row 6 — Intensity / Blend (Orange LEDs)

- **In Dream/Blend mode**: Controls `intensity` (gradient step size): 0.005 → 0.08
- **In Style mode**: Controls `style_blend` (original vs styled): 0.0 → 1.0

### Row 5 — Feedback (Purple LEDs)

Controls how much of the previous output feeds into the current frame: 0.0 → 0.9

- **0.0**: Each frame is independent
- **0.3**: Moderate recursion (default)
- **0.9**: Heavy recursion (psychedelic accumulation)

### Row 4 — Hue Shift (Pink LEDs)

Color rotation per frame: 0.0 → 0.1

Creates a cycling color palette effect. Even small values (0.01-0.02) produce noticeable color evolution.

### Row 3 — Iterations / Blur (Blue LEDs)

- **In Dream/Blend mode**: Gradient ascent iterations per octave: 1 → 15
- **In Style mode**: Post-processing Gaussian blur: 0.0 → 1.0

**Performance note:** Iterations multiply linearly with processing time. Use 3-5 for real-time.

### Row 2 — Zoom Speed (Cyan LEDs)

Zoom-in factor per frame: 1.0 → 1.015

| Value | Effect |
|-------|--------|
| 1.0 | No zoom |
| 1.002 | Subtle slow zoom (default) |
| 1.005 | Noticeable zoom |
| 1.01 | Fast infinite zoom |
| 1.015 | Very fast zoom |

### Row 1 — Presets (White LEDs)

Instant parameter presets that set mode + all parameters at once:

| Pad | Preset | Description |
|-----|--------|-------------|
| Col 0 | Subtle | Gentle dream, low feedback, slow zoom |
| Col 1 | Deep | Rich multi-octave dream |
| Col 2 | Psychedelic | Max hallucination + hue cycling |
| Col 3 | Gentle Style | Light style overlay |
| Col 4 | Full Style | Complete painterly style |
| Col 5 | Dream+Style | Both engines blended |
| Col 6 | Inf. Zoom | Classic deep dream zoom recursion |
| Col 7 | Chaos | Everything maxed (very slow!) |

---

## LED Feedback

The system provides visual feedback via pad colors:

- **Dim white**: Inactive pad (available for selection)
- **Bright color** (per row): Currently selected value for that parameter
- **Row colors**: Red (layer), Green (style), Orange (intensity), Purple (feedback), Pink (hue), Blue (iter), Cyan (zoom), White (presets)

LEDs update periodically during operation to reflect the current parameter state.

---

## Velocity Sensitivity

Pads respond to strike velocity (0.0 – 1.0). When multiple pads are held simultaneously, the average velocity modulates intensity as a performance expression feature.

---

## Keyboard Fallback

When no Launchpad is available (or using `--no-midi`):

| Key | Action |
|-----|--------|
| `1` | DeepDream mode |
| `2` | Style Transfer mode |
| `3` | Blend mode |
| `F` | Toggle fullscreen |
| `H` | Toggle HUD |
| `Space` | Freeze frame |
| `+` / `=` | Increase intensity ×1.2 |
| `-` | Decrease intensity ÷1.2 |
| `R` | Reset parameters |
| `P` | Print profiler report (turbo) |
| `Q` / `ESC` | Quit |

---

## MIDI Technical Details

### Protocol

- **Input**: Note On (0x90), Note Off (0x80), Control Change (0xB0)
- **Output**: Note On for pad colors, SysEx for RGB colors
- **Programmer Mode SysEx**: `F0 00 20 29 02 0D 0E 01 F7`

### Port Detection

The system searches MIDI port names for "launchpad" + ("mk3" or "mk 3"). On Jetson, the Launchpad typically appears as two ports — the system prefers the programmer/MIDI port over the DAW port.

### Error Handling

If MIDI is unavailable (SSH session, no ALSA, no Launchpad connected), the system gracefully falls back to keyboard-only control with a warning message.
