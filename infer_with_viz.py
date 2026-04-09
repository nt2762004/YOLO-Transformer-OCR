"""
OCR Inference with Visualization
Runs end-to-end OCR and saves results with bounding boxes as PNG
"""

from pathlib import Path
import argparse

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

from src.infer import load_recognition_checkpoint, recognize_text
from src.config import get_project_paths


def visualize_results(image_path, results, output_path="ocr_results.png"):
    """
    Visualize OCR results with bounding boxes and save as PNG
    
    Args:
        image_path: Path to input image
        results: List of OCR results from recognize_text()
        output_path: Path to save output visualization
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes and text
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for i, item in enumerate(results):
        box = item['box']
        text = item['text']
        confidence = item['confidence']
        
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text label
        label = f"{text} ({confidence:.2f})"
        ax.text(
            x1, y1 - 5,
            label,
            fontsize=9,
            color=colors[i % len(colors)],
            bbox=dict(facecolor='white', alpha=0.7, pad=2),
            verticalalignment='bottom'
        )
    
    ax.set_title(f"OCR Results - {len(results)} text regions detected", fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="OCR Inference with Visualization")
    parser.add_argument("--image", type=str, required=True, help="Path to receipt image")
    parser.add_argument("--detector", type=str, default="artifacts/detector_runs/yolo_textdet/weights/best.pt")
    parser.add_argument("--recognizer", type=str, default="artifacts/checkpoints/recognition_best.pt")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename from input image
    image_name = Path(args.image).stem
    output_path = output_dir / f"{image_name}_result.png"
    
    # Verify files exist
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    if not Path(args.detector).exists():
        print(f"❌ Detector not found: {args.detector}")
        return
    
    if not Path(args.recognizer).exists():
        print(f"❌ Recognizer not found: {args.recognizer}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load models
    print("📦 Loading YOLO detector...")
    yolo = YOLO(args.detector)
    print("✅ YOLO loaded")
    
    print("📦 Loading recognition model...")
    model, tokenizer = load_recognition_checkpoint(Path(args.recognizer), device)
    print("✅ Recognition model loaded")
    
    # Run inference
    print(f"🔍 Running inference on: {args.image}")
    results = recognize_text(
        args.image,
        yolo,
        model.encoder,
        model.decoder,
        tokenizer,
        device,
        max_len=100
    )
    
    print(f"✅ Found {len(results)} text regions")
    
    # Print results
    print("\n" + "="*80)
    print("OCR RESULTS")
    print("="*80)
    for i, item in enumerate(results, 1):
        box = [int(x) for x in item['box']]
        text = item['text']
        conf = item['confidence']
        print(f"{i:3d}. [{box[0]:4d}, {box[1]:4d}, {box[2]:4d}, {box[3]:4d}] | Conf: {conf:.4f} | Text: {text}")
    print("="*80)
    
    # Visualize and save
    print(f"\n📊 Creating visualization...")
    output_path = visualize_results(args.image, results, output_path)
    print(f"✅ Visualization saved: {output_path}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
