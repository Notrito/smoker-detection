"""
Utility functions for the smoker detection project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to {seed}")


def get_device():
    """
    Get the device to use (cuda or cpu).
    
    Returns:
        torch.device: Device to use for training/inference
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU")
    
    return device


def save_checkpoint(model, optimizer, epoch, val_acc, path='checkpoint.pth'):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        val_acc: Validation accuracy
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint file
    
    Returns:
        tuple: (epoch, val_acc)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Checkpoint loaded from {path}")
    print(f"   Epoch: {epoch}, Val Acc: {val_acc:.2f}%")
    
    return epoch, val_acc


def visualize_samples(dataset, num_samples=8, class_names=['Not Smoking', 'Smoking']):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: SmokerDataset instance
        num_samples: Number of samples to display
        class_names: List of class names
    
    Returns:
        matplotlib figure
    """
    # Get random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Calculate grid size
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in zip(indices, axes):
        # Get image (without transform for visualization)
        img_path = dataset.image_paths[idx]
        from PIL import Image
        img = Image.open(img_path)
        label = dataset.labels[idx]
        
        # Display
        ax.imshow(img)
        ax.set_title(f'{class_names[label]}\n{img.size[0]}x{img.size[1]}', 
                     fontsize=10, fontweight='bold',
                     color='red' if label == 1 else 'green')
        ax.axis('off')
    
    # Hide extra subplots
    for ax in axes[num_samples:]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def print_dataset_info(train_loader, val_loader, test_loader):
    """
    Print information about the datasets.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    print("\n" + "="*60)
    print("📊 Dataset Information")
    print("="*60)
    
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + val_size + test_size
    
    print(f"\nDataset Splits:")
    print(f"   Training:   {train_size:4d} images ({100*train_size/total_size:.1f}%)")
    print(f"   Validation: {val_size:4d} images ({100*val_size/total_size:.1f}%)")
    print(f"   Test:       {test_size:4d} images ({100*test_size/total_size:.1f}%)")
    print(f"   Total:      {total_size:4d} images")
    
    print(f"\nBatch Information:")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    print("="*60 + "\n")


def create_directories(dirs):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories created: {', '.join(dirs)}")


def count_dataset_images(data_path):
    """
    Count images in dataset folders.
    
    Args:
        data_path: Path to dataset root
    
    Returns:
        dict: Image counts per split
    """
    data_path = Path(data_path)
    counts = {}
    
    for split in ['Training', 'Validation', 'Testing']:
        folder = data_path / split / split
        if folder.exists():
            smoking = len(list(folder.glob('smoking_*.jpg')))
            not_smoking = len(list(folder.glob('notsmoking_*.jpg')))
            counts[split] = {
                'smoking': smoking,
                'not_smoking': not_smoking,
                'total': smoking + not_smoking
            }
    
    return counts