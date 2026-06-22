# BirdCLEF+ 2026 — Class Imbalance & Rare Species Learning

**Student:** Saanvi Upadhyay (2024120091)  
**Role:** Class Imbalance & Rare Species Learning  
**Course:** COSE47400 Deep Learning, Spring 2026

---

## Overview

This repository contains my individual experiments for the BirdCLEF+ 2026 Kaggle competition. The task involves classifying 234 bird species recorded in the Pantanal region of Brazil under CPU-only inference constraints. My contribution focused on addressing severe class imbalance and improving recognition of rare bird species through different loss functions, sampling strategies, and multi-label learning.

---

## Experiments

| Notebook | Method | Key Result |
|----------|---------|------------|
| `baseline_training.ipynb` | EfficientNetV2-B0 + Mel Spectrogram + Mixup | Public Leaderboard: **0.749** |
| `exp01_focal_loss_training.ipynb` | Focal Loss (α = 1, γ = 2) | Best single loss-function experiment |
| `exp02_weighted_loss_training.ipynb` | Weighted Binary Cross-Entropy | Performed below baseline |
| `exp03_sampler_training.ipynb` | Rare-class oversampling | Performed below baseline |
| `exp04_combined.ipynb` | Multi-label learning using secondary labels | Macro-F1 improved from **0.51 → 0.60** |
| `exp05_focal_loss_soundscape.ipynb` | Focal Loss + Soundscape Data + Multi-label Learning | **Macro-F1: 0.669**, **Validation AUC: 0.966** |

---

## Key Findings

- **Focal Loss (γ = 2)** consistently outperformed weighted loss and oversampling by emphasizing difficult minority-class examples during training.
- **Weighted BCE** and **oversampling** alone did not improve performance over the baseline.
- Integrating **soundscape recordings** introduced training examples for **28 bird species** that were absent from the curated training dataset.
- Using **secondary labels as multi-hot vectors** enabled effective multi-label learning and substantially improved rare-species recognition.
- The combined approach implemented in **Experiment 5** achieved the strongest overall validation performance (**Macro-F1 = 0.669**, **Validation AUC = 0.966**).

---

## Repository Contents

- Baseline implementation
- Five experimental notebooks
- Training logs
- Performance comparison across different imbalance-handling techniques
- Final implementation for rare-species learning

---

## Training Log

`exp05_training_log.csv` contains epoch-by-epoch metrics, including:

- Training Loss
- Validation Loss
- Validation AUC
- Validation Macro-F1

---

## Team Project Outcome

Our final ensemble combined the models developed by all team members using soft voting, achieving:

- **Public Leaderboard:** **0.858**
- **Private Leaderboard:** **0.855**
- **≈14.5% improvement over the baseline model**

---

## Author

**Saanvi Upadhyay**  
Korea University  
COSE47400 Deep Learning — Spring 2026
