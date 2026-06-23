# BirdCLEF+ 2026 — Bird Species Recognition

**Course:** COSE47400 Deep Learning, Spring 2026 · Korea University

A team project for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition: classifying **234 bird species** from audio recordings of the Pantanal region of Brazil, under **CPU-only inference** constraints. Each member explored a different strategy to improve over a shared baseline; the final submission is a soft-voting ensemble of all members' models.

---

## Results

| Model | Public LB | Private LB |
|-------|-----------|------------|
| Baseline (EfficientNetV2-B0 + Mel Spectrogram + Mixup) | 0.749 | — |
| **Final team ensemble (soft voting)** | **0.858** | **0.855** |

≈ **14.5% improvement** over the baseline.

---

## Team & Contributions

| Member | Focus area | Directory |
|--------|-----------|-----------|
| **Saanvi Upadhyay** | Class imbalance & rare-species learning (focal loss, weighted BCE, sampling, multi-label, soundscape) | [`focal_loss/`](focal_loss/) · `baseline_training.ipynb` |
| **Siun** | Data augmentation & multi-label learning; final ensemble | [`data_aug&multi_label/`](data_aug%26multi_label/) · `ensemble_inference.ipynb` |
| **Younseo** | Spectrogram augmentation & zero-shot learning (mixup, spec-aug, Perch features) | [`spectrogram_aug&zero_shot/`](spectrogram_aug%26zero_shot/) |

> Track details and per-experiment results are documented in each directory's own README where available (e.g. [`focal_loss/README.md`](focal_loss/README.md)).

---

## Repository Structure

```
.
├── baseline_training.ipynb          # Shared baseline model
├── ensemble_inference.ipynb         # Final soft-voting ensemble
├── focal_loss/                      # Class imbalance & rare-species experiments
├── data_aug&multi_label/            # Data augmentation & multi-label experiments
└── spectrogram_aug&zero_shot/       # Spectrogram augmentation & zero-shot experiments
```

---

## Notes

- Notebooks are written for the Kaggle environment (BirdCLEF+ 2026 dataset attached) and CPU-only inference.
- Each experiment notebook is self-contained and records its own training/validation metrics.
