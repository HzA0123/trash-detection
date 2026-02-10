# Comparative Analysis of Faster R-CNN and YOLOv8 for Trash Detection ♻️

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the official source code and research materials for the undergraduate thesis titled **"Comparative Analysis of Deep Learning Models for Waste Classification"**. This project evaluates and compares two state-of-the-art object detection architectures—**Faster R-CNN (Two-Stage)** and **YOLOv8 (One-Stage)**—to determine the most effective model for automated trash sorting systems.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Performance Results](#-performance-results)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)

## 🔍 Overview
Waste management is a critical global challenge. This research proposes an automated computer vision solution to classify waste into **6 categories**: `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, and `Trash`. 

The study focuses on the trade-off between **Accuracy** (Faster R-CNN) and **Speed** (YOLOv8) to provide recommendations for different implementation scenarios (e.g., static waste audit vs. robotic sorting).

## ✨ Key Features
*   **Dual-Model Implementation**: Complete training pipelines for both Faster R-CNN (ResNet50-FPN backbone) and YOLOv8m.
*   **Custom Dataset**: Enhanced TrashNet dataset with auto-labeling and manual verification.
*   **Comprehensive Evaluation**: Metrics include mAP@50, Recall, F1-Score, and Inference Speed (FPS).
*   **Analysis Tools**: Scripts for confusion matrix generation, loss visualization, and side-by-side inference comparison.

## 📂 Dataset
*   **Source**: Modified TrashNet Dataset.
*   **Classes**: 6 Classes (`Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`).
*   **Preprocessing**:
    *   **YOLOv8**: Mosaic Augmentation, HSV Color Jitter, Resize to 512x512.
    *   **Faster R-CNN**: Random Horizontal Flip, Normalization, Resize to 800px (min-size).

## 🧠 Model Architectures
| Component | Faster R-CNN | YOLOv8m |
| :--- | :--- | :--- |
| **Type** | Two-Stage Detector | One-Stage Detector |
| **Backbone** | ResNet-50-FPN | CSPDarknet53 |
| **Input Size** | 800 x 800 px | 512 x 512 px |
| **Optimizer** | SGD (lr=0.005) | SGD (Auto) |
| **Epochs** | 30 (Early Stop at 19) | 50 |

## 📊 Performance Results
The research successfully identified clear distinct advantages for each model:

| Metric | Faster R-CNN | YOLOv8m | Winner |
| :--- | :--- | :--- | :--- |
| **mAP@50** | 92.69% | **93.20%** | 🏆 YOLOv8 |
| **Recall** | **97.87%** | 88.12% | 🏆 FRCNN |
| **F1-Score** | **95.21%** | 90.59% | 🏆 FRCNN |
| **Speed (FPS)** | 4.7 FPS | **29.6 FPS** | 🏆 YOLOv8 |

> **Conclusion**: 
> *   **Faster R-CNN** is recommended for high-precision tasks where missing an object is critical (High Recall).
> *   **YOLOv8** is ideal for real-time applications requiring instant feedback (High Speed).

## 💻 Installation & Usage

### Prerequisites
*   Python 3.10+
*   CUDA 11.8+ (Recommended for GPU acceleration)

### 1. Clone Repository
```bash
git clone https://github.com/HzA0123/trash-detection.git
cd trash-detection
```

### 2. Install Dependencies
```bash
# For Faster R-CNN
pip install -r fasterRCNN/requirements.txt

# For YOLOv8
pip install ultralytics
```

### 3. Run Inference Comparison
To test both models on a single image:
```bash
python comparison/compare_single.py
```
*Note: Ensure you have placed your trained models in the respective directories.*

## 📁 Project Structure
```
├── fasterRCNN/         # Faster R-CNN Training & Config Source
│   ├── train.py        # Main training script
│   ├── dataset.py      # Custom Dataset Loader
│   └── utils.py        # Helper functions
├── yolo/               # YOLOv8 Training Source
│   ├── train.py        # YOLO Training script
│   └── dataset.yaml    # Ultralytics dataset config
├── comparison/         # Evaluation & Comparison Scripts
│   ├── compare_models.py # Generates metrics comparison
│   └── results/        # Output charts and logs
└── README.md           # Project Documentation
```

---
*Created by Hafid for Undergraduate Thesis Research (2025).*
