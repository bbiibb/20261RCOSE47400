# BirdCLEF+ 2026: Acoustic Species Identification in Pantanal Soundscapes

**Team 1** | COSE47400 Deep Learning, Spring 2026, Korea University

| Student | ID | Role |
|---|---|---|
| Siun Lim | 2023250028 | Class Imbalance & Rare Species Learning |
| Younseo Choi | 2024320378 | Domain Shift Adaptation |
| Saanvi Upadhyay | 2024120091 | Spectrogram Regularization & Zero-shot |

> **Baseline LB 0.749 → Final Ensemble LB 0.858 (+14.5%)**

---

## Task Overview

Classify 234 bird species from 60-second Pantanal soundscapes under **CPU-only inference constraints**, evaluated by class-mean Average Precision (cmAP). Three core challenges:

1. **Domain shift** — Clean Xeno-canto training clips vs. noisy real-world soundscapes at test time
2. **Class imbalance** — Severe long-tail distribution; many species appear in fewer than 10 clips
3. **Multi-label overlap** — Multiple species frequently vocalize simultaneously

---

## Repository Structure

```
20261RCOSE47400/
├── baseline_training.ipynb              # EfficientNetV2-B0 baseline (LB 0.749)
├── ensemble_inference.ipynb             # Weighted soft-voting ensemble (LB 0.858)
├── README.md
│
├── data_aug&multi_label/                # Domain Shift (Younseo Choi)
│   ├── exp1_audio_augmentation.ipynb        # Exp1: Waveform-level augmentation
│   ├── exp2_soundscape_noise_injection.ipynb # Exp2: Background noise injection
│   ├── exp3_multilabel_soundscape_timeshift.ipynb # Exp3: Multi-label + Soundscape Time Shift
│   └── exp3_inference.ipynb                 # Inference for Exp3 model
│
├── focal_loss/                          # Class Imbalance (Siun Lim)
│   ├── exp01_focal_loss_training.ipynb      # Exp C1: Focal Loss (α=1, γ=2)
│   ├── exp02_weighted_loss_training.ipynb   # Exp C2: Weighted BCE (pos_weight)
│   ├── exp03_sampler_training.ipynb         # Exp C3: WeightedRandomSampler
│   ├── exp04_combined.ipynb                 # Exp C4: Weighted BCE + Sampler combined
│   ├── exp05_focal_loss_soundscape.ipynb    # Exp C5: Focal Loss + Soundscape + Multi-label
│   ├── exp05_training_log.csv               # Epoch-by-epoch training metrics for Exp C5
│   └── README.md
│
└── spectrogram_aug&zero_shot/           # Spectrogram Regularization (Saanvi Upadhyay)
    ├── Exp1_mixup.py                        # Exp B1: Waveform-level Mixup (before mel)
    ├── Exp1_inference.ipynb
    ├── Exp2_spectrogram_augmentation.py     # Exp B2: SpecAugment + Spectrogram-level Mixup
    ├── Exp2_inference.ipynb
    ├── Exp3_perch_feature_zeroshot_experiment.py  # Exp B3: Perch v2 zero-shot probe
    └── Exp3__inference_perch-onnx-feature.ipynb
```

---

## Baseline

**File:** `baseline_training.ipynb`

| Component | Detail |
|---|---|
| Architecture | EfficientNetV2-B0 (~6M params), 234-class sigmoid head |
| Input | 5-second clips @ 32 kHz → 256×256 log-mel spectrogram (n_mels=256, n_fft=2048, f_min=20 Hz) |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW (lr=5×10⁻⁴) + CosineAnnealingLR, 32 epochs |
| Augmentation | Spectrogram-level Mixup (α=0.5, θ=0.8) |
| Inference | 60s soundscape → 12 non-overlapping 5s windows |
| **Result** | **Val AUC 0.985 / Public LB 0.749** |

The gap between Val AUC and LB motivates all downstream experiments.

---

## Experiments

### 1. Domain Shift Adaptation (`data_aug&multi_label/`)

**Goal:** Bridge the gap between clean Xeno-canto training data and noisy Pantanal soundscapes.  
All experiments use: EfficientNetV2-B0, BCEWithLogitsLoss, AdamW lr=5×10⁻⁴, batch size 32.

#### Exp1 — Waveform-level Augmentation (32 epochs)
**File:** `exp1_audio_augmentation.ipynb`

Waveform augmentation pipeline applied before mel-spectrogram conversion (`use_audio_aug=True`):

| Technique | Parameters | Purpose |
|---|---|---|
| AddGaussianNoise | p=0.5, amplitude 0.001–0.015 | Broadband ambient noise |
| TimeStretch | p=0.3, rate 0.8–1.2× | Recording device variation |
| PitchShift | p=0.3, ±2 semitones | Distance and temperature effects |
| Gain | p=0.3, ±6 dB | Microphone sensitivity differences |

Labels remain single-label (primary label only). **→ Public LB 0.788 (+5.2%)**

---

#### Exp2 — Soundscape Background Noise Injection (25 epochs)
**File:** `exp2_soundscape_noise_injection.ipynb`

Builds on Exp1 with additional `AddBackgroundNoise` (p=0.5, SNR 5–20 dB) using `train_soundscapes` recordings as noise sources. Training reduced to 25 epochs due to GPU quota.

**→ Public LB 0.769 (↓ degraded)** — Soundscape files contain unlabeled co-occurring species, causing contradictory supervision signals. This failure directly motivated Exp3.

---

#### Exp3 — Multi-label Encoding + Soundscape Integration + Time Shift (32 epochs)
**File:** `exp3_multilabel_soundscape_timeshift.ipynb`

Two simultaneous changes:
1. **Multi-hot label encoding** via `make_labels_row`: encodes both `primary_label` and `secondary_labels` (parsed via `ast.literal_eval`) across 4,372 samples (12.3% of train audio)
2. **Soundscape data as labeled training samples**: 28 of 234 species appear *exclusively* in `train_soundscapes` and cannot be learned otherwise. All valid soundscape segments from `train_soundscapes_labels.csv` are incorporated, with Time Shift augmentation (`soundscape_shift_sec=1.0`, 3 offsets per segment) expanding 1,478 → 4,434 samples. Soundscape samples excluded from validation to prevent distributional leakage.

**→ Public LB 0.832 / Private LB 0.842 — best individual score (+8.1% from multi-label alone)**

---

### 2. Class Imbalance (`focal_loss/`)

**Goal:** Improve recognition of rare species where BCE loss is dominated by frequent classes.  
All experiments: EfficientNetV2-B0, AdamW lr=5×10⁻⁴, CosineAnnealingLR, fold 0, **16 epochs**.

| File | Method | Val AUC | Macro-F1 |
|---|---|---|---|
| `exp02_weighted_loss_training.ipynb` | Weighted BCE (`pos_weight = N / (class_count × K)`) | 0.980 | 0.512 |
| `exp03_sampler_training.ipynb` | WeightedRandomSampler (inverse frequency per sample) | 0.982 | 0.531 |
| `exp04_combined.ipynb` | Weighted BCE + WeightedRandomSampler | 0.977 | 0.620 |
| **`exp01_focal_loss_training.ipynb`** | **Focal Loss (α=1, γ=2)** | **0.989** | **0.710** |

Focal Loss applies instance-level modulation `(1 − p_t)^γ`, concentrating gradients on hard/ambiguous examples without requiring prior knowledge of class frequencies. It was the **only method to surpass the baseline** on both metrics.

Weighted BCE and oversampling individually underperformed the baseline; their combination degraded further (AUC 0.977), showing the two strategies interfere rather than complement.

#### Exp C5 — Focal Loss + Soundscape + Multi-label (16 epochs)
**File:** `exp05_focal_loss_soundscape.ipynb` | **Log:** `exp05_training_log.csv`

Combines Focal Loss with soundscape data integration and multi-label encoding.

| Metric | Best Value | Epoch |
|---|---|---|
| Val AUC | 0.966 | 11 |
| Val Macro-F1 | 0.669 | 13 |

---

### 3. Spectrogram Regularization & Zero-shot (`spectrogram_aug&zero_shot/`)

**Goal:** Improve spectrogram-level regularization and zero-shot prediction for species absent from training audio.

#### Exp B1 — Waveform-level Mixup (32 epochs)
**File:** `Exp1_mixup.py` + `Exp1_inference.ipynb`

Mixup (α=0.5, θ=0.8) applied to **raw waveforms before mel-spectrogram conversion** (`mix(x, y)` → `self.mel(x)`). SpecAugment disabled (`spec_aug_enabled = False`).

**→ Public LB 0.787 / Private LB 0.794**

---

#### Exp B2 — SpecAugment + Spectrogram-level Mixup (32 epochs)
**File:** `Exp2_spectrogram_augmentation.py` + `Exp2_inference.ipynb`

Mixup applied **after mel-spectrogram conversion** (`self.mel(x)` → `mix(x, y)`). SpecAugment enabled (`spec_aug_enabled = True`) with frequency masking (`freq_mask_param=12`) and time masking (`time_mask_param=24`, p=0.3).

**→ Public LB 0.787 / Private LB 0.784**

Waveform-level Mixup (B1) outperforms spectrogram-level (B2) on Private LB, suggesting that mixing raw audio before feature extraction provides more acoustically meaningful variation.

---

#### Exp B3 — Perch v2 Zero-shot Hybrid (Zero-shot)
**File:** `Exp3_perch_feature_zeroshot_experiment.py` + `Exp3__inference_perch-onnx-feature.ipynb`

A separate probe MLP trained on **Google Perch v2 embeddings** targets the 28 zero-shot species absent from `train_audio`. At inference, `hybrid_routing=True` replaces only zero-shot class columns in the baseline prediction with probe predictions — seen-class scores remain unchanged.

- Zero-shot Macro AUC: **0.660 → 0.997**
- **→ Public LB 0.796 / Private LB 0.797** — highest individual score among B experiments

---

## Ensemble (`ensemble_inference.ipynb`)

Models targeting complementary challenges are combined via **weighted soft voting** using Public LB scores as weights:

$$\hat{y} = \frac{\sum_k w_k \cdot \sigma(f_k(x))}{\sum_k w_k}$$

| Ensemble Version | Public LB | Private LB |
|---|---|---|
| Domain Shift model (Exp3) solo | 0.831 | 0.831 |
| + Focal Loss | 0.840 | 0.840 |
| **Domain Shift + Spectrogram Aug (Final)** | **0.858** | **0.855** |

The two-model ensemble (Domain Shift + Spectrogram Augmentation) outperforms the three-model version that adds Focal Loss, confirming that **complementary objectives yield greater synergy than simply adding more models**.

---

## Key Takeaways

1. **Failure analysis reveals stronger directions**: Soundscape noise injection (Exp2) degraded performance, but diagnosing *why* (unlabeled co-occurring vocalizations → contradictory labels) directly led to treating soundscapes as labeled data (Exp3), the largest single gain (+8.1%).

2. **Val AUC is not a reliable proxy**: Multi-label integration reduced Val AUC (0.988 → 0.962) yet improved LB. The validation set consists of clean Xeno-canto recordings that do not reflect test conditions.

3. **Waveform-level Mixup > spectrogram-level Mixup**: Mixing raw audio before feature extraction (B1 Private 0.794) consistently outperforms mixing post-conversion (B2 Private 0.784).

4. **Complementary ensembles beat additive ones**: Two models trained on different challenges outperform three models trained on overlapping objectives.

---

## References

1. A. Masquelier. BirdCLEF 2026 – PyTorch Baseline [Training]. Kaggle Notebook, 2026.
2. D. S. Park et al. SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech*, 2019.
3. M. Tan and Q. V. Le. EfficientNetV2: Smaller models and faster training. *ICML*, 2021.
4. T. Y. Lin et al. Focal loss for dense object detection. *ICCV*, 2017.
