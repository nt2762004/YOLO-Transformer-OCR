from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    """
    Plot training metrics and save to output directory.
    
    Args:
        history: List of dictionaries with metrics from each epoch
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(history)
    
    # Plot 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2.5, label='Train Loss', markersize=5)
    ax.plot(df['epoch'], df['val_loss'], marker='s', linewidth=2.5, label='Validation Loss', markersize=5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'loss_plot.png'}")
    
    # Plot 2: Character Error Rate (CER) and Word Error Rate (WER)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['cer'] * 100, marker='o', linewidth=2.5, label='CER (%)', markersize=5, color='#FF6B6B')
    ax.plot(df['epoch'], df['wer'] * 100, marker='s', linewidth=2.5, label='WER (%)', markersize=5, color='#4ECDC4')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Character Error Rate (CER) & Word Error Rate (WER)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cer_wer_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'cer_wer_plot.png'}")
    
    # Plot 3: Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], marker='o', linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val Loss
    axes[0, 1].plot(df['epoch'], df['val_loss'], marker='s', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_title('Validation Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # CER
    axes[1, 0].plot(df['epoch'], df['cer'] * 100, marker='o', linewidth=2, color='#FF6B6B')
    axes[1, 0].set_title('Character Error Rate (CER)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CER (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # WER
    axes[1, 1].plot(df['epoch'], df['wer'] * 100, marker='s', linewidth=2, color='#4ECDC4')
    axes[1, 1].set_title('Word Error Rate (WER)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('WER (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'combined_metrics.png'}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Epochs: {len(df)}")
    print(f"Best Val Loss: {df['val_loss'].min():.6f} (epoch {df['val_loss'].idxmin() + 1})")
    print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final Val Loss: {df['val_loss'].iloc[-1]:.6f}")
    print(f"Final CER: {df['cer'].iloc[-1]*100:.2f}%")
    print(f"Final WER: {df['wer'].iloc[-1]*100:.2f}%")
    print("="*50)
