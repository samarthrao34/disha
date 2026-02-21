---
description: Repository Information Overview
alwaysApply: true
---

# Disha - Emotion Analysis Project

## Summary
Disha is a Python-based emotion analysis project that uses computer vision and deep learning to detect and analyze emotions in facial images. It combines multiple models including DeepFace and optionally FER for emotion detection, and includes a sentiment analysis component using transformers.

## Structure
- **main.py**: Core application with emotion analysis functionality
- **train/**: Contains training data organized by emotion categories and training script
- **test/**: Test dataset organized by emotion categories
- **outputs/**: Directory for storing analysis results
- **models/**: Generated during training to store trained models (referenced in code)

## Language & Runtime
**Language**: Python
**Version**: 3.10
**Build System**: Standard Python
**Package Manager**: Likely pip/uv (uv.lock file present)

## Dependencies
**Main Dependencies**:
- deepface (>=0.0.95): For facial analysis
- opencv-python (>=4.11.0.86): For image processing
- tf-keras (>=2.20.1): For deep learning models
- torch: For PyTorch-based models
- numpy: For numerical operations
- transformers: For sentiment analysis
- fer: For facial emotion recognition

**Development Dependencies**:
- torchvision: Used in training script for image transformations

## Build & Installation
```bash
# Install dependencies
pip install -e .

# Optional: Install additional dependencies not in pyproject.toml
pip install torch torchvision transformers fer
```

## Training
**Framework**: PyTorch
**Training Script**: train/train_emotion.py
**Dataset Structure**: Images organized in directories by emotion label
**Model Architecture**: ResNet18 (fine-tuned)
**Run Command**:
```bash
python train/train_emotion.py --epochs 10 --batch-size 32
```

## Inference
**Main Function**: analyze_emotion_ensemble in main.py
**Input**: Face image
**Output**: Emotion label, confidence score, and probability distribution
**Configuration**: Environment variables (EMO_MIN_TOP, EMO_T, USE_FER, DEBUG_EMO)

## Testing
**Test Data**: Located in test/ directory, organized by emotion categories
**Evaluation**: Manual testing with test images

## Additional Features
**Sentiment Analysis**: Text-based sentiment analysis using transformers
**Probability Normalization**: Custom softmax and normalization functions
**Ensemble Approach**: Combines multiple models for improved accuracy