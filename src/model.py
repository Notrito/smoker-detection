"""
LoRA (Low-Rank Adaptation) implementation for convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for convolutional layers.
    
    Args:
        original_layer: The Conv2d layer to adapt
        rank: LoRA rank (default=8)
              - Lower rank (4): Fewer parameters, less overfitting risk, less capacity
              - Medium rank (8-16): Balanced trade-off (recommended for most tasks)
              - Higher rank (32+): More capacity but approaches full fine-tuning
              
              For small datasets (<1000 images), rank=8 provides sufficient
              adaptation capacity while keeping parameters low (~2% of original layer).
    """
    
    def __init__(self, original_layer, rank=8):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        # Get dimensions from original layer
        out_channels = original_layer.out_channels
        in_channels = original_layer.in_channels
        kernel_size = original_layer.kernel_size
        
        # LoRA matrices: A (down-projection) and B (up-projection)
        # A reduces dimensions: in_channels -> rank
        # Initialized with small random values to break symmetry
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_channels, *kernel_size) * 0.01
        )
        
        # B expands dimensions: rank -> out_channels
        # Initialized to zeros so LoRA starts as identity (preserves pretrained weights)
        # This initialization strategy follows the original LoRA paper
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels, rank, 1, 1)
        )
        
        # Freeze original weights (preserve ImageNet knowledge)
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass combining original frozen weights with LoRA adaptation.
        
        Mathematical formulation:
        output = W_frozen * x + (B * (A * x))
        
        where * denotes convolution operation.
        """
        # Original forward pass (frozen pretrained weights)
        original_output = self.original_layer(x)
        
        # LoRA adaptation pathway (low-rank decomposition)
        # Step 1: Down-project with A (in_channels → rank)
        lora_output = F.conv2d(
            x,
            self.lora_A,
            stride=self.original_layer.stride,
            padding=self.original_layer.padding
        )
        
        # Step 2: Up-project with B (rank → out_channels)
        # These two sequential convolutions approximate a low-rank adaptation
        lora_output = F.conv2d(lora_output, self.lora_B)
        
        # Combine: W*x + (B*(A*x)) where * denotes convolution
        return original_output + lora_output


def get_model(num_classes=2, pretrained=True):
    """
    Load ResNet34 with optional pretrained weights.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
    
    Returns:
        ResNet34 model
    """
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)
    
    # Modify last layer for classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def apply_lora_to_model(model, target_layers=['layer3', 'layer4'], rank=8):
    """
    Apply LoRA adapters to specific layers in ResNet34.
    
    Strategy: We target layer3 and layer4 (high-level feature extractors) because:
    - layer1 & layer2: Extract low-level features (edges, textures) that are 
      universal across tasks → keep frozen, no adaptation needed
    - layer3 & layer4: Extract high-level semantic features (objects, contexts)
      that are task-specific → need slight adaptation for smoking detection
    - fc: Brand new classifier head → fully trainable
    
    This approach gives us the sweet spot:
    - Full fine-tuning: 21.8M params (overfitting risk with small datasets)
    - Only fc training: ~1K params (may underfit, features not adapted)
    - LoRA on layer3+layer4: ~465K params (2.14% of model, balanced approach)
    
    Args:
        model: ResNet34 model
        target_layers: List of layer names to apply LoRA to
        rank: LoRA rank (default=8, adds ~2% params per adapted layer)
    
    Returns:
        Number of convolutional layers where LoRA was applied
    """
    # Freeze ALL layers first (preserve ImageNet features)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the new classification head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    lora_count = 0
    
    for layer_name in target_layers:
        # Get the layer dynamically (e.g., model.layer3)
        layer = getattr(model, layer_name)
        
        # Iterate through all blocks in this layer
        for block in layer:
            # Find all Conv2d layers in this block dynamically
            for name, module in block.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Get parent module and attribute name to replace it
                    parent = block
                    attr_names = name.split('.')
                    
                    # Navigate to parent of the conv layer
                    for attr in attr_names[:-1]:
                        parent = getattr(parent, attr)
                    
                    # Check if not already wrapped
                    current_module = getattr(parent, attr_names[-1])
                    if not isinstance(current_module, LoRALayer):
                        # Replace with LoRA-wrapped version
                        setattr(parent, attr_names[-1], LoRALayer(current_module, rank=rank))
                        lora_count += 1
    
    return lora_count


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Returns:
        tuple: (total_params, trainable_params, trainable_percentage)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100. * trainable_params / total_params
    
    return total_params, trainable_params, trainable_pct