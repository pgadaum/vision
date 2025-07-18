![](UTA-DataScience-Logo.png)

# Facial Emotion Recognition with CNNs

This repository contains a deep learning pipeline for classifying facial emotions using the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

---

## Overview

**Task Definition**:  
The objective is to classify grayscale face images (48x48 pixels) into one of 7 emotion categories: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`, as defined in the FER-2013 challenge.

**Approach**:  
We use Convolutional Neural Networks (CNNs) to learn visual features from raw pixel data. The model is trained on the official training split, validated during training, and evaluated on the test set. The pipeline includes data augmentation, normalization, early stopping, and learning rate scheduling.

**Summary of Performance**:  
After 50 epochs, the model achieved:
- Training accuracy: ~52%
- Validation accuracy: ~57.5%

---

## Summary of Work

### Data

- **Source**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Type**: 48x48 grayscale facial images
- **Format**: Folder structure with images in `train/` and `test/`, organized by label
- **Size**: ~35,000 images
  - Train: 28,709
  - Test: 7,178

### Preprocessing

- Images resized to 48x48 if necessary
- Normalized to pixel values in `[0, 1]`
- Labels encoded as integers
- Augmented training images using horizontal flips and slight rotations

### Data Visualization

- Class distribution visualized with a bar chart
- Sample images shown per emotion label to confirm correctness

---

## Problem Formulation

- **Input**: Grayscale image (48x48)
- **Output**: Emotion class (0–6)

### Model

- Model: CNN with Conv2D, MaxPooling, BatchNorm, Dropout, and Dense layers
- Loss: `categorical_crossentropy`
- Optimizer: `Adam` with learning rate scheduling
- Metrics: Accuracy

---

## Training

- Environment: Google Colab
- Runtime: T4 GPU (~14s/epoch)
- Total Epochs: 50
- Training Time: ~12–15 minutes
- Model saved using `ModelCheckpoint`

### Training Curves

- Training/Validation loss and accuracy plotted
- Validation accuracy peaked ~57.5%

### Early Stopping

- Used patience of 5 epochs to avoid overfitting

---

## Performance Summary

| Metric       | Train Set | Validation Set |
|--------------|-----------|----------------|
| Accuracy     | 52.1%     | 57.5%           |
| Loss         | 1.2101    | 1.1343          |

Classes: angry, disgust, fear, happy, neutral, sad, surprise

---

## Conclusions

- The CNN model effectively learns from raw FER-2013 data
- Validation performance outperformed training, indicating generalization
- Disgust class was underrepresented, leading to potential class imbalance issues

---

## Future Work

- Apply more advanced architectures (ResNet, MobileNet)
- Use pre-trained models with fine-tuning
- Address class imbalance with oversampling or weighted loss
- Explore multi-label emotion detection
- Improve performance using emotion bounding boxes

---

## Reproducibility

To reproduce:

1. Install `kagglehub` and download dataset
2. Execute all cells in `image2.ipynb` sequentially
3. Training will auto-start and save best model to disk

---

## Repository Structure

```
.
├── image2.ipynb              # Full notebook: download, preprocess, train, evaluate
├── utils.py                  # (Optional) Utility functions for preprocessing
├── /models                   # Directory to store saved models
├── /plots                    # Accuracy/loss training curves
└── README.md                 # This file
```

---

## Software Setup

```bash
pip install -q kagglehub opencv-python tensorflow matplotlib seaborn
```

---

## Data Access

- Dataset: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Download via:

```python
import kagglehub
path = kagglehub.dataset_download("msambare/fer2013")
```

---

## Training & Evaluation

Run `image2.ipynb` top to bottom:
- Model is trained for 50 epochs
- Automatically evaluates on test set
- Graphs are generated for accuracy/loss

---

## Credits

Dataset: Kaggle contributor `msambare`  
Model: CNN designed in Keras/TensorFlow  
Notebook by: *Pawan Gadaum*
