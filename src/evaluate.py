"""
Evaluation functions for model testing and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_loader, device, class_names=['Not Smoking', 'Smoking']):
    """
    Evaluate model on test set and return predictions and labels.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        class_names: List of class names for reporting
    
    Returns:
        tuple: (all_predictions, all_labels, test_accuracy)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("🧪 Evaluating on Test Set...")
    print(f"   Test batches: {len(test_loader)}\n")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    test_acc = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    return all_preds, all_labels, test_acc


def print_classification_report(predictions, labels, class_names=['Not Smoking', 'Smoking']):
    """
    Print detailed classification metrics.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        class_names: List of class names
    """
    print(f"\n{'='*60}")
    print(f"📊 TEST SET RESULTS")
    print(f"{'='*60}")
    
    # Overall accuracy
    test_acc = 100. * sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    print(f"\n   Overall Accuracy: {test_acc:.2f}%\n")
    
    # Detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=class_names, digits=4))
    print(f"{'='*60}")


def plot_confusion_matrix(predictions, labels, class_names=['Not Smoking', 'Smoking'], 
                          save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        class_names: List of class names
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    return fig


def get_predictions_with_confidence(model, dataloader, device):
    """
    Get predictions along with confidence scores.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for data
        device: Device to run inference on
    
    Returns:
        tuple: (predictions, confidences, labels)
    """
    model.eval()
    all_preds = []
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Get softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_confidences), np.array(all_labels)


def analyze_errors(model, dataloader, device, dataset, num_samples=10):
    """
    Analyze misclassified samples.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for data
        device: Device to run inference on
        dataset: Original dataset to access images
        num_samples: Number of error samples to display
    
    Returns:
        List of dictionaries with error information
    """
    predictions, confidences, labels = get_predictions_with_confidence(model, dataloader, device)
    
    # Find misclassified samples
    errors = []
    for idx, (pred, conf, label) in enumerate(zip(predictions, confidences, labels)):
        if pred != label:
            errors.append({
                'index': idx,
                'true_label': label,
                'predicted_label': pred,
                'confidence': conf,
                'image_path': dataset.image_paths[idx]
            })
    
    print(f"\n🔍 Error Analysis:")
    print(f"   Total errors: {len(errors)}")
    print(f"   Error rate: {100 * len(errors) / len(labels):.2f}%")
    
    # Sort by confidence (highest confidence errors are most interesting)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    return errors[:num_samples]