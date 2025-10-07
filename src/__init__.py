"""
Smoker Detection with LoRA Fine-Tuning

A parameter-efficient approach to binary image classification using 
Low-Rank Adaptation (LoRA) on pretrained ResNet34.
"""

from .model import (
    LoRALayer,
    get_model,
    apply_lora_to_model,
    count_parameters
)

from .dataset import (
    SmokerDataset,
    get_transforms,
    create_dataloaders
)

from .train import (
    train_one_epoch,
    validate,
    train_model,
    get_optimizer_and_criterion
)

from .evaluate import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_history,
    get_predictions_with_confidence,
    analyze_errors
)

from .utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    visualize_samples,
    print_dataset_info,
    create_directories,
    count_dataset_images
)

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    # Model
    'LoRALayer',
    'get_model',
    'apply_lora_to_model',
    'count_parameters',
    
    # Dataset
    'SmokerDataset',
    'get_transforms',
    'create_dataloaders',
    
    # Training
    'train_one_epoch',
    'validate',
    'train_model',
    'get_optimizer_and_criterion',
    
    # Evaluation
    'evaluate_model',
    'print_classification_report',
    'plot_confusion_matrix',
    'plot_training_history',
    'get_predictions_with_confidence',
    'analyze_errors',
    
    # Utils
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'visualize_samples',
    'print_dataset_info',
    'create_directories',
    'count_dataset_images',
]