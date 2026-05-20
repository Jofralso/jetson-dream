#!/usr/bin/env python3
"""
Demo: Process a static image through DeepDream and Style Transfer.

Shows how the AI engines work without needing camera/MIDI.
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import cv2
import numpy as np

from scripts.dream_engine import DeepDreamEngine
from scripts.style_engine import StyleTransferEngine


def demo_image(image_path: str, output_dir: str = "demo_output"):
    """Process an image through both engines and save results."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"  Resolution: {w}×{h}")
    
    # Resize to processing resolution for speed
    proc_w, proc_h = 480, 360
    if (w, h) != (proc_w, proc_h):
        img = cv2.resize(img, (proc_w, proc_h))
        print(f"  Resized to: {proc_w}×{proc_h} for processing")
    
    orig = img.copy()
    
    print("\n" + "="*60)
    print("DEEPDREAM ENGINE")
    print("="*60)
    
    # Initialize DeepDream
    dream = DeepDreamEngine(resolution=(proc_w, proc_h), turbo=False)
    
    # Process with different layers
    print("\nProcessing DeepDream at different layers:")
    print("  (Layer 0=textures, 4=balanced, 9=extreme)")
    
    for layer in [0, 4, 9]:
        print(f"\n  Layer {layer}...", end=" ", flush=True)
        dream.layer_index = layer
        dream.intensity = 0.03
        dream.octaves = 3
        dream.iterations = 5
        
        result = dream.process_frame(orig.copy())
        
        output_path = f"{output_dir}/deepdream_layer{layer}.jpg"
        cv2.imwrite(output_path, result)
        print(f"✓ → {output_path}")
    
    print("\n" + "="*60)
    print("STYLE TRANSFER ENGINE")
    print("="*60)
    
    # Initialize Style Transfer
    style = StyleTransferEngine(models_dir="models", resolution=(proc_w, proc_h), turbo=False)
    
    print(f"\nAvailable styles: {style.style_names}")
    print("\nProcessing styles:")
    
    for i, style_name in enumerate(style.style_names[:4]):  # First 4 styles
        print(f"\n  Style {i}: {style_name}...", end=" ", flush=True)
        
        if not style.load_style(style_name):
            print("✗ Model not found (download first)")
            continue
        
        result = style.process_frame(orig.copy())
        
        # Sanitize filename
        safe_name = style_name.replace("/", "_").replace(" ", "_")
        output_path = f"{output_dir}/style_{safe_name}.jpg"
        cv2.imwrite(output_path, result)
        print(f"✓ → {output_path}")
    
    print("\n" + "="*60)
    print("BLEND MODE (50% Dream + 50% Style)")
    print("="*60)
    
    # Blend mode demo
    print("\nProcessing blend (InceptionV3 + Style Transfer)...")
    
    dream.layer_index = 6
    dream.intensity = 0.03
    dream.octaves = 2
    dream.iterations = 3
    
    dream_result = dream.process_frame(orig.copy())
    
    style.select_style_by_index(0)
    style_result = style.process_frame(orig.copy())
    
    blend_result = cv2.addWeighted(dream_result, 0.5, style_result, 0.5, 0)
    
    output_path = f"{output_dir}/blend_50_50.jpg"
    cv2.imwrite(output_path, blend_result)
    print(f"  ✓ → {output_path}")
    
    # Also try 70% domain 30% style
    blend_result_70 = cv2.addWeighted(dream_result, 0.7, style_result, 0.3, 0)
    output_path = f"{output_dir}/blend_70_30.jpg"
    cv2.imwrite(output_path, blend_result_70)
    print(f"  ✓ → {output_path}")
    
    print("\n" + "="*60)
    print("EFFECTS DEMO")
    print("="*60)
    
    # Test parameter variations
    print("\nTesting feedback loop (previous frame recursion)...")
    
    dream.layer_index = 4
    dream.intensity = 0.02
    dream.octaves = 2
    dream.iterations = 3
    dream.feedback = 0.3  # Accumulate 30% of previous
    
    feedback_frame = orig.copy()
    for i in range(5):
        feedback_frame = dream.process_frame(feedback_frame)
        if i == 4:  # Save final
            output_path = f"{output_dir}/feedback_recursive.jpg"
            cv2.imwrite(output_path, feedback_frame)
            print(f"  ✓ Iteration 5 → {output_path}")
    
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY")
    print("="*60)
    
    # Show how octaves affect output
    print("\nOctave impact (Dream mode):")
    dream.feedback = 0.0  # No feedback
    dream.layer_index = 4
    dream.intensity = 0.03
    dream.iterations = 5
    
    for octaves in [1, 2, 4]:
        dream.octaves = octaves
        result = dream.process_frame(orig.copy())
        output_path = f"{output_dir}/octaves_{octaves}.jpg"
        cv2.imwrite(output_path, result)
        print(f"  {octaves} octaves → {output_path}")
    
    # Show intensity impact
    print("\nIntensity impact (Dream mode, 2 octaves):")
    dream.octaves = 2
    
    for intensity in [0.01, 0.03, 0.08]:
        dream.intensity = intensity
        result = dream.process_frame(orig.copy())
        output_path = f"{output_dir}/intensity_{intensity:.3f}.jpg"
        cv2.imwrite(output_path, result)
        print(f"  intensity={intensity} → {output_path}")
    
    print("\n" + "="*60)
    print(f"✓ ALL DEMOS COMPLETE")
    print(f"  Output saved to: {output_dir}/")
    print("="*60)
    
    # List all outputs
    import os
    outputs = sorted(os.listdir(output_dir))
    print(f"\nGenerated {len(outputs)} images:")
    for fname in outputs:
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath) / 1024
        print(f"  • {fname} ({size:.1f} KB)")


def generate_test_image(output_path: str = "test_image.jpg", width: int = 640, height: int = 480):
    """Generate a test image with shapes and text."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Gradient background
    for y in range(height):
        img[y, :] = [int(255 * (1 - y/height)), 100, int(255 * y/height)]
    
    # Circles and shapes
    cv2.circle(img, (width//4, height//4), 80, (0, 255, 255), -1)
    cv2.circle(img, (3*width//4, height//4), 60, (255, 0, 255), -1)
    cv2.rectangle(img, (100, height-150), (width-100, height-50), (0, 255, 0), -1)
    
    # Text
    cv2.putText(img, "JETSON DREAM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "DeepDream & Style Transfer Demo", (40, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, img)
    print(f"Generated test image: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo AI engines on static image")
    parser.add_argument("image", nargs="?", default=None,
                       help="Input image path (or auto-generate test image)")
    parser.add_argument("-o", "--output", default="demo_output",
                       help="Output directory (default: demo_output/)")
    parser.add_argument("--generate", action="store_true",
                       help="Generate a test image if none provided")
    
    args = parser.parse_args()
    
    # Use provided image or generate one
    if args.image:
        image_path = args.image
    elif args.generate or True:  # Auto-generate by default
        image_path = generate_test_image()
    else:
        print("ERROR: Provide an image or use --generate")
        sys.exit(1)
    
    demo_image(image_path, args.output)
