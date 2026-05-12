# NaturaViT + ALO: Healthy / Unhealthy Sleep Classification

## Problem Statement

Sleep disorders affect a large portion of the population and often go undiagnosed. Automated classification of sleep recordings into **Healthy** or **Unhealthy** categories can assist clinical decision-making by identifying patients who require further evaluation.

This work addresses the binary sleep classification problem using EEG/polysomnographic signal features extracted from overnight recordings. The goal is to classify each subject as:

- **Class 0** — Healthy sleeper
- **Class 1** — Unhealthy sleeper (presence of a sleep disorder)

---

## Dataset

| Property | Value |
|---|---|
| Source file | `healthy_unhealthy1.csv` |
| Total samples | 18,599 |
| Number of features | 1,025 |
| Label column | Last column (integer: 0 or 1) |
| Class balance | Binary, stratified splits used |

**Preprocessing:**
- Z-score normalization per feature: $\hat{x} = \frac{x - \mu}{\sigma + \epsilon}$
- 80/20 stratified train/test split (`random_state=42`)
- Training validation split: 20% of training data held for validation during model training

---

## Pipeline Overview

```
Raw Features (1025-dim)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1: Hybrid CNN-ViT     │  ← End-to-end training (baseline)
│  Multi-Scale CNN + ViT       │
└──────────────┬───────────────┘
               │  256-dim deep features (dual pooling output)
               ▼
┌──────────────────────────────┐
│  Stage 2: ALO Feature        │  ← Ant Lion Optimization
│  Selection                   │    selects 128 / 256 features
└──────────────┬───────────────┘
               │  ALO-selected deep features (128-dim)
               ▼
┌──────────────────────────────┐
│  Stage 3: Downstream         │  ← SVM, KNN, MLP classifiers
│  Classifiers                 │    trained on selected features
└──────────────────────────────┘
```

---

## Stage 1: Hybrid CNN-ViT Architecture (Baseline)

### Input

Raw normalized feature vector of shape `(1025,)` reshaped to `(1025, 1)` for 1-D convolution.

### Multi-Scale CNN Block

Three parallel `Conv1D` branches capture local patterns at different temporal scales:

| Branch | Kernel Size | Filters |
|--------|-------------|---------|
| Branch 1 | 3 | 32 |
| Branch 2 | 7 | 32 |
| Branch 3 | 15 | 32 |

Branches are concatenated → `(1025, 96)`, followed by `Dropout(0.2)`.

Two subsequent strided convolution + pooling layers further compress the sequence:

| Layer | Filters | Kernel | Pooling | Output Shape |
|-------|---------|--------|---------|--------------|
| Conv1D | 64 | 5 | MaxPool(2) | `(512, 64)` |
| Conv1D | 128 | 5 | MaxPool(2) | `(256, 128)` |

Each convolution is followed by `BatchNormalization` and `Dropout`.

### Positional Encoding

Learnable `Embedding(input_dim=seq_len, output_dim=128)` added element-wise to the CNN output to inject position information into the sequence before the Transformer blocks.

### Transformer Blocks (× 2)

Each `TransformerBlock` contains:

- **Multi-Head Self-Attention** — 8 heads, `embed_dim=128`, `dropout=0.1`
  - Scaled dot-product attention: $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Feed-Forward Network** — `Dense(256, GELU)` → `Dropout` → `Dense(128)`
- **Layer Normalization** (pre-residual) with $\epsilon = 10^{-6}$
- **Residual connections** around both sub-layers

### Dual Pooling → Deep Features

```
GlobalAveragePooling1D  (128-dim)
         +
GlobalMaxPooling1D      (128-dim)
         │
     Concatenate
         │
   256-dim deep feature vector   ← used for ALO feature selection
```

### Classification Head

```
Dense(256, ReLU) → BatchNorm → Dropout(0.4)
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
Dense(2, Softmax)
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (`lr = 0.001`) |
| Loss | Sparse Categorical Cross-Entropy |
| Epochs | 150 (with early stopping) |
| Batch size | 64 |
| Early stopping patience | 25 epochs (monitors `val_loss`) |
| LR reduction patience | 10 epochs (factor 0.5, min `1e-7`) |
| Model checkpoint | Best `val_accuracy` saved |

---

## Stage 2: Ant Lion Optimization (ALO) for Feature Selection

### Motivation

The 256-dimensional deep feature vector from the CNN-ViT pooling layer may contain redundant or noisy dimensions. ALO is applied to select the most discriminative subset, reducing dimensionality while preserving classification power.

### ALO Algorithm

ALO mimics the predator–prey hunting strategy of ant lions:

1. **Initialization** — `N=20` ant lion positions randomly sampled in `[0, 1]^256`
2. **Fitness function** — Each position `p` is decoded to a binary mask (`p > 0.5`). A `KNeighborsClassifier(k=5)` is evaluated using **3-fold stratified CV** on training data only (no data leakage):

$$\text{Fitness} = \alpha \cdot (1 - \overline{\text{CV Acc}}) + (1 - \alpha) \cdot \frac{|\text{selected}|}{|\text{total}|}, \quad \alpha = 0.99$$

3. **Random walk** with adaptive shrinkage — bounds tighten at 10%, 50%, 75%, 90%, 95% of iterations
4. **Elite guidance** — Best ant lion (elite) guides ant walks at each iteration
5. **Survival selection** — Top `N` positions from merged ant lion + ant population kept each iteration
6. **Iterations** — 25 iterations, 20 agents

### ALO Result

| Property | Value |
|---|---|
| Input feature dimension | 256 |
| Selected feature dimension | **128** |
| Reduction | 50% |
| Best CV Accuracy (ALO) | See convergence curve |

Selected features are the 128 indices where `best_position > 0.5`.

---

## Stage 3: Downstream Classifiers on ALO-Selected Features

Three classifiers trained on the `(N_train, 128)` ALO-selected scaled deep features:

| Classifier | Configuration |
|---|---|
| **SVM** | RBF kernel, `class_weight='balanced'`, `probability=True` |
| **KNN** | `k=5`, `weights='distance'`, Minkowski metric |
| **MLP** | Hidden layers `(128, 64)`, `max_iter=500` |

---

## Results

### Evaluation Metrics

All models evaluated on the held-out test set (20% of 18,599 = ~3,720 samples).

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | AUC | Kappa |
|---|---|---|---|---|---|---|---|
| **Hybrid CNN-ViT (Baseline)** | 86.40% | 84.02% | 89.89% | 82.90% | 86.86% | 0.9487 | 0.7280 |
| **ALO + SVM** | **87.66%** | **89.60%** | 85.22% | **90.11%** | **87.35%** | 0.9368 | **0.7532** |
| ALO + KNN | 87.20% | 87.94% | 86.24% | 88.17% | 87.08% | 0.9339 | 0.7441 |
| ALO + MLP | 86.45% | 88.26% | 84.09% | 88.82% | 86.12% | 0.9409 | 0.7290 |

> **Best overall model: ALO + SVM** with 87.66% accuracy — a **+1.26%** improvement over the baseline.

### Key Observations

- The **baseline CNN-ViT** achieves the highest AUC (0.9487), indicating strong probability calibration.
- **ALO + SVM** improves accuracy, precision, specificity, F1, and Kappa over the baseline by reducing noisy deep features.
- **ALO + KNN** provides the best recall balance while maintaining high specificity.
- **ALO + MLP** provides the most conservative predictions (highest specificity at 88.82%).
- ALO consistently improves **precision and specificity** across all classifiers, reducing false positives — clinically important for avoiding unnecessary diagnoses.

---

## Artifacts Saved

| File | Description |
|---|---|
| `artifacts/hybrid_cnn_vit_alo_best.h5` | Best CNN-ViT model (by val accuracy) |
| `artifacts/hybrid_cnn_vit_alo_final.h5` | Final trained CNN-ViT model |
| `artifacts/alo_best_mask.npy` | Binary feature selection mask (256-dim) |
| `artifacts/alo_best_position.npy` | Continuous ALO position vector |
| `artifacts/alo_convergence_curve.npy` | Fitness per iteration |
| `artifacts/training_history.npy` | CNN-ViT train/val accuracy & loss |
| `artifacts/comparison_summary.csv` | All model metrics in tabular form |
| `artifacts/baseline_predictions.csv` | CNN-ViT test predictions & probabilities |
| `artifacts/svm_alo_predictions.csv` | ALO+SVM predictions & probabilities |
| `artifacts/knn_alo_predictions.csv` | ALO+KNN predictions & probabilities |
| `artifacts/mlp_alo_predictions.csv` | ALO+MLP predictions & probabilities |
| `artifacts/training_history.png` | Accuracy & loss curves |
| `artifacts/baseline_evaluation.png` | Baseline confusion matrix & ROC |
| `artifacts/alo_convergence.png` | ALO convergence plot |
| `artifacts/svm_alo_evaluation.png` | SVM confusion matrix & ROC |
| `artifacts/knn_alo_evaluation.png` | KNN confusion matrix & ROC |
| `artifacts/mlp_alo_evaluation.png` | MLP confusion matrix & ROC |
| `artifacts/model_comparison.png` | Bar chart: all models vs all metrics |

---

## Summary

This work presents a two-stage classification framework for healthy/unhealthy sleep detection:

1. A **Hybrid CNN-ViT** model combines multi-scale local feature extraction (CNN) with global context modeling (Vision Transformer) to learn a compact 256-dimensional deep representation from 1025 raw signal features.

2. **Ant Lion Optimization** reduces the 256-dim deep feature space by 50% (to 128 features), retaining only the most discriminative dimensions while penalizing feature-set size.

3. Classical **SVM, KNN, and MLP** classifiers trained on ALO-selected features outperform or match the end-to-end deep model in accuracy and interpretability, with **ALO+SVM** achieving the best accuracy of 87.66% and a Cohen's Kappa of 0.753.
