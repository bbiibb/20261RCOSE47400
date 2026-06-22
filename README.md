# BirdCLEF+ 2026 — Class Imbalance & Rare Species Learning

**Student:** Saanvi Upadhyay (2024120091)  
**Role:** Class Imbalance & Rare Species Learning  
**Course:** COSE47400 Deep Learning, Spring 2026

---

## Overview

This repository contains my individual experiments for the BirdCLEF+ 2026 Kaggle competition. The objective is to classify 234 bird species recorded in the Pantanal region of Brazil. My contribution focuses on addressing class imbalance and improving the recognition of rare bird species using different deep learning strategies.

---

## Repository Structure

| Notebook | Description |
|----------|-------------|
| `baseline_training.ipynb` | Baseline EfficientNetV2-B0 model with Mel-spectrogram features and Mixup augmentation. |
| `exp01_focal_loss_training.ipynb` | Investigates the use of Focal Loss to improve learning for minority classes. |
| `exp02_weighted_loss_training.ipynb` | Evaluates Weighted Binary Cross-Entropy based on class frequencies. |
| `exp03_sampler_training.ipynb` | Explores oversampling techniques for underrepresented bird species. |
| `exp04_combined.ipynb` | Implements multi-label learning using secondary labels encoded as multi-hot vectors. |
| `exp05_focal_loss_soundscape.ipynb` | Combines Focal Loss, soundscape recordings, and multi-label learning for improved performance. |

---

## Key Contributions

- Investigated multiple techniques for handling severe class imbalance.
- Compared Focal Loss, Weighted BCE, and oversampling methods.
- Integrated soundscape recordings to improve recognition of rare species.
- Implemented multi-label learning using secondary labels.
- Evaluated each approach using validation metrics and performance comparisons.

---

## Main Findings

- **Focal Loss** provided the most effective strategy for learning underrepresented classes by focusing training on difficult examples.
- **Weighted BCE** and **oversampling** did not outperform the baseline model in this project.
- Incorporating **soundscape recordings** together with **multi-label encoding** produced the strongest overall results among the conducted experiments.

---

## Files

- `baseline_training.ipynb`
- `exp01_focal_loss_training.ipynb`
- `exp02_weighted_loss_training.ipynb`
- `exp03_sampler_training.ipynb`
- `exp04_combined.ipynb`
- `exp05_focal_loss_soundscape.ipynb`

---

## 
