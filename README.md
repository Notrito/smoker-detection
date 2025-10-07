# 🚬 Smoker Detection using LoRA Fine-Tuning

Fine-tuning a pretrained ResNet34 model with LoRA (Low-Rank Adaptation) for binary smoking detection in images.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This project demonstrates **parameter-efficient fine-tuning** using LoRA on a small dataset (1,120 images). By training only 2.14% of the model's parameters, we achieved **89.73% test accuracy** while preserving pretrained ImageNet features.

### Key Features:
- ✅ LoRA implementation from scratch for convolutional layers
- ✅ Only 465K trainable parameters (vs 21.8M full fine-tuning)
- ✅ Balanced performance across both classes (~89% F1-score)
- ✅ Efficient training on free Kaggle GPU

## 🎯 Results

| Metric | Validation | Test |
|--------|------------|------|
| **Accuracy** | 94.44% | 89.73% |
| **Precision (Smoking)** | - | 88.03% |
| **Recall (Smoking)** | - | 91.96% |
| **F1-Score (Smoking)** | - | 89.96% |

**Training Efficiency:**
- Trainable parameters: 465K (2.14%)
- Training time: ~15 minutes (15 epochs)
- GPU: Kaggle T4 (free tier)

## 🏗️ Model Architecture

- **Base Model:** ResNet34 (pretrained on ImageNet)
- **LoRA Adaptation:** Applied to layer3 and layer4 (high-level features)
- **LoRA Rank:** 8
- **Classification Head:** Linear(512 → 2)

### Parameter Distribution:
Total: 21.7M parameters
├─ Frozen (ImageNet): 21.3M (97.86%)
└─ Trainable: 465K (2.14%)
├─ LoRA adapters: 464K
└─ Classification head: 1K

## 📊 Dataset

**Source:** [Kaggle - Smoking Detection Dataset](https://www.kaggle.com/datasets/sujaykapadnis/smoking)

- **Total Images:** 1,120 (perfectly balanced)
- **Training:** 716 images (64%)
- **Validation:** 180 images (16%)
- **Test:** 224 images (20%)

**Challenges:**
- Low resolution (250×250 pixels)
- Small cigarettes in frame
- High variability in framing and lighting

## 🚀 Quick Start

### Installation
git clone https://github.com/YOUR_USERNAME/smoker-detection-lora.git
cd smoker-detection-lora
pip install -r requirements.txt
Training
bash# Run the Jupyter notebook
jupyter notebook notebooks/smoker_detection.ipynb
Or use the training script:
bashpython src/train.py --epochs 15 --lr 1e-4 --rank 8

## 🔧 Technologies Used

PyTorch - Deep learning framework
torchvision - Pretrained models and transforms
scikit-learn - Evaluation metrics
matplotlib/seaborn - Visualization
tqdm - Progress bars

## 📈 Training Details
Hyperparameters:

Learning Rate: 1e-4 (conservative for fine-tuning)
Optimizer: AdamW with weight decay 1e-4
Batch Size: 32
Epochs: 15
LoRA Rank: 8

Data Augmentation:

Random horizontal flip (p=0.5)
Random rotation (±10°)
Color jitter (brightness, contrast, saturation, hue)

## 🧠 What is LoRA?
LoRA (Low-Rank Adaptation) modifies pretrained weights by adding small trainable matrices:
Output = W_frozen × input + (B × A) × input
                             └─────┬─────┘
                          Low-rank adapter
Where:

W = Original frozen weights (from ImageNet)
A, B = Small trainable matrices (rank << original dimensions)

Benefits:

🔹 Massively reduced trainable parameters (~98% reduction)
🔹 Prevents overfitting on small datasets
🔹 Preserves pretrained knowledge
🔹 Faster training and lower memory

## 📁 Project Structure
smoker-detection-lora/
├── notebooks/
│   └── smoker_detection.ipynb    # Main training notebook
├── src/
│   ├── model.py                  # LoRA implementation
│   ├── dataset.py                # Custom dataset class
│   └── train.py                  # Training functions
├── results/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── requirements.txt
└── README.md

## 🎓 Key Learnings

LoRA is highly effective for small dataset fine-tuning
Transfer learning requires careful hyperparameter selection
Data augmentation is crucial with limited data
Parameter efficiency enables training on free/limited resources

## 🔮 Future Improvements

 Experiment with different LoRA ranks (4, 12, 16)
 Implement learning rate scheduling
 Try other base models (EfficientNet, Vision Transformer)
 Deploy model with FastAPI/Gradio demo
 Collect more diverse training data


Dataset: Sujay Kapadnis on Kaggle
LoRA Paper: Hu et al., 2021
Pretrained model: torchvision ResNet34

📧 Contact
Noel Triguero - noel.triguero@gmail.com
Project Link: https://www.kaggle.com/code/notrito/smoker-detection-with-lora
