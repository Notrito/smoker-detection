"""
Custom Dataset class for Smoker Detection.
Handles loading images and labels from folder structure.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SmokerDataset(Dataset):
    """
    Custom Dataset for Smoker Detection.
    
    Expects folder structure with images named:
    - smoking_XXX.jpg for positive class
    - notsmoking_XXX.jpg for negative class
    
    Args:
        folder_path: Path to folder containing images
        transform: Optional torchvision transforms to apply
    """
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load smoking images (label = 1)
        for img_path in self.folder_path.glob('smoking_*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(1)
        
        # Load not smoking images (label = 0)
        for img_path in self.folder_path.glob('notsmoking_*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(0)
        
        # Verify dataset is not empty
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Loaded {len(self.image_paths)} images from {folder_path.name}")
        print(f"  - Smoking: {sum(self.labels)}")
        print(f"  - Not Smoking: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: {'smoking': count, 'not_smoking': count}
        """
        smoking_count = sum(self.labels)
        not_smoking_count = len(self.labels) - smoking_count
        
        return {
            'smoking': smoking_count,
            'not_smoking': not_smoking_count,
            'total': len(self.labels),
            'balance': smoking_count / len(self.labels)
        }


def get_transforms(img_size=224, augment=True):
    """
    Get image transformations for training or validation.
    
    Args:
        img_size: Target image size (default: 224 for ImageNet models)
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    # ImageNet normalization (standard for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),      # Smoking can occur on either side
            transforms.RandomRotation(degrees=10),       # Slight rotations for robustness
            transforms.ColorJitter(                      # Lighting variations
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def create_dataloaders(train_path, val_path, test_path, batch_size=32, 
                       img_size=224, num_workers=2):
    """
    Create DataLoaders for training, validation, and test sets.
    
    Args:
        train_path: Path to training data folder
        val_path: Path to validation data folder
        test_path: Path to test data folder
        batch_size: Batch size for DataLoader (default: 32)
        img_size: Image size for resizing (default: 224)
        num_workers: Number of parallel workers for data loading (default: 2)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transforms = get_transforms(img_size=img_size, augment=True)
    val_transforms = get_transforms(img_size=img_size, augment=False)
    
    # Create datasets
    train_dataset = SmokerDataset(train_path, transform=train_transforms)
    val_dataset = SmokerDataset(val_path, transform=val_transforms)
    test_dataset = SmokerDataset(test_path, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomize batch composition each epoch
        num_workers=num_workers,
        pin_memory=True         # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Keep order for reproducible evaluation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n✅ DataLoaders created")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader