# BirdCLEF+ 2026 — Class Imbalance & Rare Species Learning

**Student:** Saanvi Upadhyay (2024120091)  
**Role:** Class Imbalance & Rare Species Learning  
**Course:** COSE47400 Deep Learning, Spring 2026

---

## Overview

This repository contains my individual contribution to the BirdCLEF+ 2026 Kaggle competition. The task is to classify 234 bird species recorded in the Pantanal region of Brazil under CPU-only inference constraints. My work focused on addressing severe class imbalance and improving recognition of rare bird species through different loss functions, sampling strategies, and multi-label learning.

---

## Repository Contents

| Notebook | Method | Result |
|----------|---------|--------|
| `baseline_training.ipynb` | EfficientNetV2-B0 + Mel Spectrogram + Mixup | Public LB: **0.749** |
| `exp01_focal_loss_training.ipynb` | Focal Loss (α = 1, γ = 2) | Best individual loss-function experiment |
| `exp02_weighted_loss_training.ipynb` | Weighted Binary Cross-Entropy | Performed below baseline |
| `exp03_sampler_training.ipynb` | Rare-class oversampling | Performed below baseline |
| `exp04_combined.ipynb` | Multi-label learning using secondary labels | Macro-F1 improved from **0.51 → 0.60** |
| `exp05_focal_loss_soundscape.ipynb` | Focal Loss + Soundscape Data + Multi-label Learning | **Macro-F1: 0.669**, **Validation AUC: 0.966** |

---

## Key Findings

- **Focal Loss** was the most effective loss function for learning minority classes by focusing training on difficult examples.
- **Weighted BCE** and **oversampling** did not outperform the baseline model.
- Integrating **soundscape recordings** with **multi-label learning** improved recognition of rare bird species.
- The combined approach implemented in **Experiment 5** achieved the strongest validation performance with a **Macro-F1 of 0.669** and a **Validation AUC of 0.966**.

---

## Training Logs

Training metrics for Experiment 5 are included in:

- `exp05_training_log.csv`

The log records:

- Training Loss
- Validation Loss
- Validation AUC
- Validation Macro-F1

---

## Team Outcome

The final team ensemble combined all members' models using soft voting and achieved:

- **Public Leaderboard:** **0.858**
- **Private Leaderboard:** **0.855**
- **≈14.5% improvement over the baseline model**

---

## Author

**Saanvi Upadhyay**  
Korea University  
COSE47400 Deep Learning — Spring 2026# BirdCLEF+ 2026 — Class Imbalance & Rare Species Learning

**Student:** Saanvi Upadhyay (2024120091)  
**Role:** Class Imbalance & Rare Species Learning  
**Course:** COSE47400 Deep Learning, Spring 2026

---

## Overview

This repository contains my individual contribution to the BirdCLEF+ 2026 Kaggle competition. The task is to classify 234 bird species recorded in the Pantanal region of Brazil under CPU-only inference constraints. My work focused on addressing severe class imbalance and improving recognition of rare bird species through different loss functions, sampling strategies, and multi-label learning.

---

## Repository Contents

| Notebook | Method | Result |
|----------|---------|--------|
| `baseline_training.ipynb` | EfficientNetV2-B0 + Mel Spectrogram + Mixup | Public LB: **0.749** |
| `exp01_focal_loss_training.ipynb` | Focal Loss (α = 1, γ = 2) | Best individual loss-function experiment |
| `exp02_weighted_loss_training.ipynb` | Weighted Binary Cross-Entropy | Performed below baseline |
| `exp03_sampler_training.ipynb` | Rare-class oversampling | Performed below baseline |
| `exp04_combined.ipynb` | Multi-label learning using secondary labels | Macro-F1 improved from **0.51 → 0.60** |
| `exp05_focal_loss_soundscape.ipynb` | Focal Loss + Soundscape Data + Multi-label Learning | **Macro-F1: 0.669**, **Validation AUC: 0.966** |

---

## Key Findings

- **Focal Loss** was the most effective loss function for learning minority classes by focusing training on difficult examples.
- **Weighted BCE** and **oversampling** did not outperform the baseline model.
- Integrating **soundscape recordings** with **multi-label learning** improved recognition of rare bird species.
- The combined approach implemented in **Experiment 5** achieved the strongest validation performance with a **Macro-F1 of 0.669** and a **Validation AUC of 0.966**.

---

## Training Logs

Training metrics for Experiment 5 are included in:

- `exp05_training_log.csv`

The log records:

- Training Loss
- Validation Loss
- Validation AUC
- Validation Macro-F1

---

## Team Outcome

The final team ensemble combined all members' models using soft voting and achieved:

- **Public Leaderboard:** **0.858**
- **Private Leaderboard:** **0.855**
- **≈14.5% improvement over the baseline model**

---

## Author

**Saanvi Upadhyay**  
Korea University  
COSE47400 Deep Learning — Spring 2026V
