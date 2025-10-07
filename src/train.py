"""
Training and validation functions for the smoker detection model.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=15, save_path='best_model.pth'):
    """
    Complete training loop with validation and model checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs (default: 15)
        save_path: Path to save best model (default: 'best_model.pth')
    
    Returns:
        dict: Training history with losses and accuracies
    """
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("🚀 Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Device: {device}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"\nResults:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"🎉 Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {save_path}")
    
    return history


def get_optimizer_and_criterion(model, lr=1e-4, weight_decay=1e-4):
    """
    Create optimizer and loss criterion with standard hyperparameters.
    
    Args:
        model: PyTorch model
        lr: Learning rate (default: 1e-4, conservative for fine-tuning)
        weight_decay: L2 regularization (default: 1e-4)
    
    Returns:
        tuple: (optimizer, criterion)
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    print("✅ Training configuration ready")
    print(f"   Loss: CrossEntropyLoss")
    print(f"   Optimizer: AdamW")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Optimizing {trainable_params:,} parameters")
    
    return optimizer, criterion