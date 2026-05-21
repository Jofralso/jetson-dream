# DeepDream: How It Works

## Overview

DeepDream is a computer vision technique that uses gradient ascent to amplify patterns that a neural network detects in an image. Instead of training the network to classify images, we reverse the process: we modify the *input image* to maximize the activations of a chosen layer.

The result is the network's "dreams" — the patterns it has learned to detect become amplified in the image, producing hallucinatory visuals of eyes, spirals, animal faces, and fractal textures.

---

## The Algorithm

### Core Concept: Gradient Ascent on Activations

Normal neural network training:
```
Input (fixed) → Network → Loss → Gradient → Update weights
```

DeepDream (our approach):
```
Input (modifiable) → Network → Activation norm → Gradient → Update INPUT
```

We freeze the network weights and instead modify the input pixels to maximize the L2 norm of activations at a target layer.

### Single Gradient Ascent Step

```python
def dream_step(image_tensor, model, target_layer, intensity):
    # 1. Forward pass to target layer
    image_tensor.requires_grad_(True)
    activations = model(image_tensor, target_layer)
    
    # 2. Loss = how much the layer "fires"
    loss = activations.norm()  # L2 norm of all activations
    
    # 3. Backpropagate to get gradient w.r.t. input pixels
    loss.backward()
    gradient = image_tensor.grad
    
    # 4. Normalize gradient and update input
    gradient /= gradient.abs().mean() + 1e-8
    image_tensor = image_tensor + gradient * intensity
    
    return image_tensor
```

### Multi-Octave Pipeline

Processing at a single resolution produces repetitive tile-like artifacts. The multi-octave approach processes at multiple scales:

```
Octave 1 (smallest):  160×120  → dream → extract detail
Octave 2 (medium):    320×240  → dream → extract detail  
Octave 3 (largest):   640×480  → dream → final output
```

At each octave:
1. Resize image to the octave's resolution
2. Add detail accumulated from previous octaves
3. Run N iterations of gradient ascent
4. Extract the "detail" (difference from original at this scale)
5. Pass detail to the next octave

This produces **multi-scale patterns** — large structures from small octaves, fine detail from large octaves.

### Anti-Tiling: Random Jitter

Before each gradient step, the image is randomly shifted by a few pixels (`jitter` parameter). This prevents the gradient from forming repetitive grid patterns along the edges of the network's receptive field.

```python
# Shift image randomly
shifted = torch.roll(image, shifts=(random_y, random_x), dims=(2, 3))
# Dream step
shifted = dream_step(shifted, ...)
# Shift back
result = torch.roll(shifted, shifts=(-random_y, -random_x), dims=(2, 3))
```

---

## Layer Selection: What the Network Dreams About

InceptionV3 is organized from low-level features (early layers) to high-level concepts (deep layers):

### Early Layers (index 0-2): Edges and Textures

The network first detects simple features — edges, color gradients, simple textures. Dreaming on these layers produces:
- Wavy edge enhancement
- Fabric-like textures
- Low-frequency color bands

**Visual character:** Subtle, painterly, texture-focused

### Middle Layers (index 3-5): Eyes, Spirals, Patterns

These layers detect repeating patterns, circular structures, and proto-objects:
- Spiral and circular forms
- Eye-like structures (the network has learned to detect eyes)
- Animal-like features (dogs, birds)

**Visual character:** Classic "DeepDream" hallucinations

### Deep Layers (index 6-9): Complex Hallucinations

The deepest layers encode high-level concepts:
- Animal faces and bodies
- Architectural structures
- Complex composite objects
- Abstract geometric constructions

**Visual character:** Surreal, highly detailed hallucinations

---

## Feedback Loop

The `feedback` parameter controls temporal recursion:

```python
current_input = (1 - feedback) * camera_frame + feedback * previous_output
```

- **feedback = 0**: Each frame is processed independently. Clean input, no accumulation.
- **feedback = 0.3**: Moderate recursion. Dreams build gently over time.
- **feedback = 0.7+**: Heavy recursion. The dream feeds on itself, producing increasingly complex patterns.

High feedback creates the "infinite regression" effect where patterns emerge, amplify, and evolve over time — even from a static input.

---

## Zoom Effect

The `zoom` parameter applies a slight zoom-in to each frame before processing:

```python
# Crop center slightly and resize back to original dimensions
crop_margin = (1 - 1/zoom) / 2
cropped = frame[margin:-margin, margin:-margin]
zoomed = resize(cropped, original_size)
```

Combined with feedback, this creates the classic **infinite zoom** effect where the dream appears to zoom endlessly into itself, revealing new patterns at every scale.

---

## Real-Time Optimizations

### AMP (Automatic Mixed Precision)

The gradient computation uses FP16 where possible:
```python
with torch.amp.autocast("cuda"):
    activations = model(tensor, target_layer)
    loss = activations.norm()
loss.backward()  # Gradients computed in mixed precision
```

This roughly halves memory bandwidth requirements and doubles throughput on Tensor Core-equipped GPUs.

### Fast Dream Mode (Nano)

For maximum FPS, `process_frame_fast()` eliminates the octave pyramid entirely:
- 1 forward pass + 1 backward pass
- No multi-scale processing
- No octave detail accumulation

The dream effect is lighter but processing is 3-6× faster. The `feedback` parameter compensates by accumulating the effect over time.

### Early Exit

The `InceptionDreamModel.forward()` method returns early once it reaches the target layer. Targeting layer 0 (Conv2d_1a_3x3) only computes 1 layer; targeting layer 9 (Mixed_7a) computes all 17 layers.

---

## Parameter Interaction Guide

### For Stronger Dreams (at the cost of FPS)
- Increase `intensity` (0.03 – 0.06)
- Increase `iterations` (8 – 15)
- Increase `octaves` (4 – 5)
- Target deeper layers (index 6 – 9)

### For Faster Processing (at the cost of dream strength)
- Decrease resolution (use `--nano` or `--process-res ultra_fast`)
- Use `process_frame_fast` (1 iteration, no octaves)
- Target shallower layers (index 0 – 3)
- Increase `frame_skip`

### For Trippy Evolution Without High Cost
- High `feedback` (0.5 – 0.8) — the dream accumulates cheaply
- Moderate `zoom` (1.003 – 1.005) — creates infinite regression
- Moderate `hue_shift` (0.01 – 0.02) — colors cycle naturally
- Low `intensity` with high feedback — subtle per-frame, strong over time

---

## Comparison: DeepDream vs Style Transfer

| Aspect | DeepDream | Style Transfer |
|--------|-----------|---------------|
| **Mechanism** | Gradient ascent on activations | Single forward pass through TransformNet |
| **Requires gradients** | Yes (backward pass) | No |
| **Cost per frame** | O(octaves × iterations × backward) | O(1 forward pass) |
| **GPU memory** | Higher (stores gradients) | Lower (inference only) |
| **Effect** | Hallucinatory patterns emerge from image | Artistic style applied uniformly |
| **Temporal** | Evolves with feedback (living dream) | Consistent style (static painting) |
| **TensorRT compatible** | No (needs gradients) | Yes (3-5× speedup) |
| **Typical FPS** | 5-15 (depending on settings) | 15-30+ |

Both engines can be blended in **Blend Mode** for combined effects.
