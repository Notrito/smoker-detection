"""
Main training script for Smoker Detection with LoRA.

Usage:
    python train.py --data_path /path/to/data --epochs 15 --lr 1e-4 --rank 8
"""

import argparse
from pathlib import Path
import torch

from src.model import get_model, apply_lora_to_model, count_parameters
from src.dataset import create_dataloaders
from src.train import train_model, get_optimizer_and_criterion
from src.evaluate import (
    evaluate_model, 
    print_classification_report, 
    plot_confusion_matrix, 
    plot_training_history
)
from src.utils import set_seed, get_device, create_directories, print_dataset_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Smoker Detection Model with LoRA')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/kaggle/input/smoking',
                        help='Path to dataset root directory')
    
    # Model arguments
    parser.add_argument('--rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--target_layers', nargs='+', default=['layer3', 'layer4'],
                        help='Layers to apply LoRA to (default: layer3 layer4)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save outputs (default: results)')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth',
                        help='Path to save best model (default: best_model.pth)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    print("\n" + "="*60)
    print("🚀 Smoker Detection Training with LoRA")
    print("="*60 + "\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    create_directories([args.output_dir])
    
    # Get device
    device = get_device()
    if args.no_cuda:
        device = torch.device('cpu')
        print("CUDA disabled by user, using CPU")
    
    # Data paths
    data_path = Path(args.data_path)
    train_path = data_path / 'Training' / 'Training'
    val_path = data_path / 'Validation' / 'Validation'
    test_path = data_path / 'Testing' / 'Testing'
    
    # Create dataloaders
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Print dataset info
    print_dataset_info(train_loader, val_loader, test_loader)
    
    # Create model
    print("\n🏗️  Building model...")
    model = get_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Apply LoRA
    print(f"\n🔧 Applying LoRA (rank={args.rank})...")
    num_lora_layers = apply_lora_to_model(
        model, 
        target_layers=args.target_layers, 
        rank=args.rank
    )
    print(f"✅ LoRA applied to {num_lora_layers} convolutional layers")
    
    # Count parameters
    total_params, trainable_params, trainable_pct = count_parameters(model)
    print(f"\n📊 Parameter Count:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"   Frozen: {total_params - trainable_params:,} ({100 - trainable_pct:.2f}%)")
    
    # Get optimizer and criterion
    print("\n⚙️  Setting up training...")
    optimizer, criterion = get_optimizer_and_criterion(
        model, 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("\n" + "="*60)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_path=args.model_save_path
    )
    
    # Plot training curves
    print("\n📊 Plotting training history...")
    fig = plot_training_history(
        history, 
        save_path=f'{args.output_dir}/training_curves.png'
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("🧪 Testing on held-out test set...")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load(args.model_save_path))
    
    # Get predictions
    predictions, labels, test_acc = evaluate_model(
        model, test_loader, device
    )
    
    # Print classification report
    print_classification_report(predictions, labels)
    
    # Plot confusion matrix
    print("\n📊 Plotting confusion matrix...")
    fig = plot_confusion_matrix(
        predictions, 
        labels,
        save_path=f'{args.output_dir}/confusion_matrix.png'
    )
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📁 Outputs saved to: {args.output_dir}/")
    print(f"   - Training curves: {args.output_dir}/training_curves.png")
    print(f"   - Confusion matrix: {args.output_dir}/confusion_matrix.png")
    print(f"   - Best model: {args.model_save_path}")
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%\n")


if __name__ == '__main__':
    main()