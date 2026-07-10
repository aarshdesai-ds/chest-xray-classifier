# Phase 3: Baseline CNN Training

From-scratch convolutional baseline for 14-label chest X-ray classification, trained on the Phase 2 data pipeline and evaluated with per-class AUROC on both validation and the official held-out test set.

---

## Table of Contents

- [Overview](#overview)
- [Phase Questions](#phase-questions)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

Phase 3 trains the first model of the project: a compact CNN built and trained **from scratch** (no pretrained weights). Its purpose is not to be state of the art but to establish an **honest baseline** that later phases — starting with transfer learning — must beat.

The phase reuses the Phase 2 `Dataset`/`DataLoader` pipeline, defines a 4-block convolutional network with a 14-logit output, trains with validation-monitored early stopping, and then evaluates the best checkpoint on both the validation and the official test split. Every metric is reported **per disease**, and the model is deliberately judged on **macro-AUROC** rather than accuracy.

---

## Phase Questions

- Can a from-scratch CNN learn usable signal on this dataset at all?
- What is an honest baseline macro-AUROC to measure future models against?
- Which of the 14 diseases are learnable, and which are near-random?
- Does the model generalize from validation to the official held-out test set?
- Why do accuracy and F1 look misleading on this task, and what should be reported instead?

---

## Dataset Description

Phase 3 consumes the patient-safe splits produced in Phase 1 and loaded in Phase 2.

| File | Rows | Role |
|---|---|---|
| `train_data.csv` | 76,277 | Model fitting |
| `val_data.csv` | 10,247 | Model selection (early stopping) |
| `test_data.csv` | 25,596 | Final held-out evaluation (used once) |

### Disease Labels

The model predicts 14 independent binary targets:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
Pleural_Thickening, Pneumonia, Pneumothorax
```

Because a single X-ray can carry several findings, this is a **multi-label** problem, so each output uses an independent sigmoid (not a softmax).

---

## Model Architecture

`MyModel` — a compact 4-block convolutional network (~1.2M parameters):

```text
Input                                             [B, 3, 224, 224]
Block 1: Conv2d(3->32, 3x3, pad 1) -> BN -> ReLU -> MaxPool(2)   -> [B, 32, 112, 112]
Block 2: Conv2d(32->64,  3x3, pad 1) -> BN -> ReLU -> MaxPool(2) -> [B, 64,  56,  56]
Block 3: Conv2d(64->128, 3x3, pad 1) -> BN -> ReLU -> MaxPool(2) -> [B, 128, 28,  28]
Block 4: Conv2d(128->256,3x3, pad 1) -> BN -> ReLU -> MaxPool(2) -> [B, 256, 14,  14]
AdaptiveAvgPool2d(1)                                             -> [B, 256, 1, 1]
Flatten -> Dropout(0.3) -> Linear(256 -> 14)                     -> [B, 14] logits
```

Design choices:

- **Square 3x3 kernels with `padding=1`** preserve spatial size within a block; `MaxPool(2)` halves it — appropriate for 2D images (unlike 1D-signal-style asymmetric kernels).
- **`AdaptiveAvgPool2d(1)`** collapses the feature map to a fixed 256-vector, so the classifier input size is independent of input resolution.
- **No sigmoid in the model** — raw logits are returned and `BCEWithLogitsLoss` applies the sigmoid internally for numerical stability.

---

## Methodology

### 1. Reuse the Phase 2 data pipeline

`ChestImageDataset` + `DataLoader` build `train`, `val`, and `test` loaders with `batch_size=64`, `num_workers=4`. Images are resized to `224x224`, converted to tensors, and normalized with ImageNet statistics.

### 2. Configure the training objective

| Component | Setting |
|---|---|
| Loss | `BCEWithLogitsLoss` (multi-label) |
| Optimizer | `Adam`, `lr=1e-5`, `weight_decay=0.01` |
| Scheduler | `StepLR(step_size=2, gamma=0.95)` |

### 3. Train with validation monitoring

For up to 50 epochs the loop:

1. trains on all training batches (`zero_grad -> forward -> loss -> backward -> step`),
2. runs a `no_grad` validation pass,
3. accumulates all validation logits/labels into `[N, 14]` arrays,
4. computes **macro-AUROC** plus per-class AUROC, precision, recall, and F1,
5. checkpoints `baseline_model.pt` whenever validation macro-AUROC improves,
6. early-stops after 5 epochs without improvement.

### 4. Evaluate per class on validation

The best epoch's per-disease metrics are assembled into a table sorted by AUROC and exported to `per_class_val_baseline.csv`.

### 5. Evaluate once on the official test set

The best checkpoint is reloaded and run through the test loader a single time, producing `per_class_test_baseline.csv`. The test set is used only for this final, unbiased estimate — never for model selection.

---

## Key Findings

### 1. The baseline learns real signal

A from-scratch CNN reaches **0.694 macro-AUROC on validation** (best epoch 11 of 17, early-stopped at 16; ~247 minutes on a T4). Well above the 0.5 random floor, so the model genuinely ranks diseased images above healthy ones.

### 2. Accuracy is a trap on this dataset

Validation accuracy sat at **95.55% every epoch** — which is exactly the accuracy of predicting "no disease" for everything, because only ~4.5% of all label slots are positive. A near-chance model can look "95% accurate." This is why the project reports **macro-AUROC**, not accuracy.

### 3. Precision / recall / F1 are ~0 at threshold 0.5 — by calibration, not by bug

Despite AUROC up to 0.85, precision/recall/F1 are essentially **0** across all classes. Cross-checked against the all-negative accuracy above, this confirms the model never crosses the 0.5 threshold: BCE on heavily imbalanced data compresses probabilities below 0.5. The **ranking is good; the operating point is wrong.** This motivates threshold tuning in a later phase, not retraining.

### 4. A validation-to-test generalization gap

The model is meaningfully weaker on the official test split:

- **Validation macro-AUROC: 0.694**
- **Test macro-AUROC: 0.604**
- **Gap: ~0.09**

Because AUROC is prevalence-invariant, the higher test prevalence does not explain this — it reflects a genuine **distribution shift in the official NIH split** (a well-documented property), plus a small amount of selection bias from tuning on validation. The **test number (0.60) is the honest generalization estimate.**

### 5. Clear-signature diseases rank best; subtle findings rank worst

Findings with obvious radiographic patterns (Edema, Effusion, Pneumothorax) rank highest; small or subtle findings (Nodule, Mass, Cardiomegaly) sit near chance. These weak classes are the primary target for transfer learning.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- baseline_model.pt                    # Best checkpoint (val macro-AUROC 0.694)
|
|-- phase-1/
|   |-- phase-1.ipynb
|   |-- phase-1-README.md
|
|-- phase-2/
|   |-- chest-xray-detection-phase2.ipynb
|   |-- phase-2-README.md
|
|-- phase-3/
|   |-- chest-xray-detection-phase3.ipynb   # Baseline training + evaluation
|   |-- phase-3-README.md                     # This file
|
|-- results/
|   |-- train_data.csv            # Shared Phase 1 output
|   |-- val_data.csv              # Shared Phase 1 output
|   |-- test_data.csv             # Shared Phase 1 output
|   |-- final_metrics.csv         # Shared class-balance summary
|   |-- per_class_val_baseline.csv           # Per-disease validation metrics
|   |-- per_class_test_baseline.csv          # Per-disease test metrics
```

---

## Requirements

```text
Python 3.8+
pandas
numpy
torch
torchvision
scikit-learn
Pillow
kagglehub
jupyter
```

Install dependencies:

```bash
pip install pandas numpy torch torchvision scikit-learn pillow kagglehub jupyter
```

A CUDA GPU is strongly recommended for training (from-scratch training on CPU is impractical). A single evaluation pass will run on CPU but is faster on GPU.

---

## How to Run

1. Attach the NIH Chest X-ray image dataset and the Phase 1 CSV dataset as Kaggle inputs.
2. Enable a GPU accelerator (Settings -> Accelerator -> GPU).
3. Open `phase-3/chest-xray-detection-phase3.ipynb`.
4. Confirm the CSV paths point to the Phase 1 exported files.
5. Run all cells to train, or — to evaluate an existing checkpoint without retraining — run only the setup, model-definition, and dataset/loader cells, then load `baseline_model.pt` and run the evaluation cells.
6. Save `models/baseline_model.pt`, `results/per_class_val_baseline.csv`, and `results/per_class_test_baseline.csv` to the notebook output.

---

## Results Summary

### Headline metrics

| Metric | Validation | Test |
|---|---|---|
| Macro-AUROC | **0.694** | **0.604** |
| Accuracy @ 0.5 | 95.55% (all-negative) | — |
| Macro F1 @ 0.5 | ~0.00 | ~0.00 |
| Best epoch | 11 | — |
| Training time | ~247 min (T4) | — |

### Per-class AUROC (validation vs test)

| Disease | Val AUROC | Test AUROC | Test support |
|---|---|---|---|
| Edema | 0.846 | 0.696 | 925 |
| Effusion | 0.784 | 0.674 | 4,658 |
| Consolidation | 0.736 | 0.651 | 1,815 |
| Pneumothorax | 0.708 | 0.683 | 2,665 |
| Atelectasis | 0.694 | 0.594 | 3,279 |
| Emphysema | 0.690 | 0.624 | 1,093 |
| Pleural_Thickening | 0.683 | 0.637 | 1,143 |
| Fibrosis | 0.666 | 0.598 | 435 |
| Infiltration | 0.659 | 0.620 | 6,112 |
| Mass | 0.645 | 0.547 | 1,748 |
| Cardiomegaly | 0.640 | 0.544 | 1,069 |
| Pneumonia | 0.633 | 0.558 | 555 |
| Nodule | 0.608 | 0.569 | 1,623 |
| Hernia | 0.724 | 0.465* | 86 |
| **Macro** | **0.694** | **0.604** | — |

\* Hernia's test AUROC is computed on only 86 positives and is high-variance; interpret with a wide confidence interval.

---

## Ideas for Extension

### 1. Transfer learning (primary next step)

Replace the from-scratch CNN with an ImageNet-pretrained backbone (e.g., DenseNet-121). As a **controlled ablation**, change only the architecture first so any AUROC gain is attributable to transfer learning. Expect the largest lift on the subtle, low-AUROC classes, and ideally a **narrower validation-to-test gap**.

### 2. Handle class imbalance

Introduce `pos_weight` (weighted BCE) or focal loss to stop the rare classes (Hernia, Pneumonia) from being ignored during training.

### 3. Threshold tuning and calibration

Select per-class thresholds on validation (max-F1 or target-sensitivity) to recover usable precision/recall from the current 0-at-0.5 behavior, and add calibration curves.

### 4. Data augmentation

Add flips, rotations, and intensity jitter to regularize and improve generalization to the shifted test distribution.

### 5. Training efficiency and hygiene

Add mixed-precision training, an `EarlyStopping` `min_delta` (~0.005) so noise-level gains stop resetting patience, and fixed seeds for reproducibility.

### 6. Interpretability

Add Grad-CAM overlays to confirm the model attends to lung anatomy rather than dataset shortcuts (e.g., corner text or laterality markers).

---

*Phase 3 establishes an honest, fully-evaluated baseline: a from-scratch CNN at **0.694 macro-AUROC on validation** and **0.604 on the official held-out test set**, with a documented generalization gap and a calibration-driven explanation for its 0-at-threshold F1. These two numbers — and the per-class tables — are the "before" that transfer learning and later phases are measured against.*
